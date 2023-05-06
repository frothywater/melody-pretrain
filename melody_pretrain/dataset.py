import os
from glob import glob
from typing import Dict, List, Literal, NamedTuple, Optional, Tuple, Union

import lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .tokenizer import MIDITokenizer

AttentionKind = Union[Literal["full"], Literal["causal"], Literal["prefix"]]
positional_id_padding_index = 0


class DatasetItem(NamedTuple):
    """A single dataset item to be processed by data collator.
    Allow extra data to be attached to the item."""

    data: np.ndarray
    note_map: np.ndarray
    extra_data: Dict[str, np.ndarray]
    filename: str


class MaskedData(NamedTuple):
    """Used for MLM, the data is a pair of (inputs, labels).
    Allow extra data to be attached to the data pair.
    """

    inputs: torch.Tensor
    labels: torch.Tensor


class InfillingData(NamedTuple):
    """Used for text infilling, the data is a pair of (masked data, target).
    Allow extra data to be attached to the data pair.
    """

    nonnoise_spans: List[np.ndarray]
    noise_spans: List[np.ndarray]
    # Indicated that indices that the target spans will be after combined
    noise_span_indices: List[int]

    # Used for specific field prediction in our ngram masking
    field_padding_indices: Optional[List[int]] = None

    # Indicate whether the masked span is used for CLM
    is_long_mask: bool = False


class DataBatch(NamedTuple):
    """A batch of data for training. Allow extra data to be attached to the batch.
    Note that in PyTorch's padding mask and attention mask, True means to ignore."""

    input_ids: torch.Tensor
    label_ids: Optional[torch.Tensor] = None

    attention_kind: AttentionKind = "full"

    # Whole sequence length for CLM/MLM or infilling/recovery
    lengths: Optional[List[int]] = None
    # Prefix length for infilling/recovery
    source_lengths: Optional[List[int]] = None

    # Positional IDs for permutated blank infilling
    positional_ids: Optional[torch.Tensor] = None

    filenames: Optional[List[str]] = None


class Masking:
    # For MLM, this field indicates whether the strategy needs to perform masking on each sequence separately,
    # if not, the strategy will perform masking on the whole batch.
    # For text infilling, masking is always performed on each sequence separately.
    need_to_mask_per_data: bool = False
    tokenizer: MIDITokenizer

    def setup_tokenizer(self, tokenizer: MIDITokenizer):
        self.tokenizer = tokenizer
        self.pad_token_tensor = torch.from_numpy(self.tokenizer.pad_token_ids).long()
        self.mask_token_tensor = torch.from_numpy(self.tokenizer.mask_token_ids).long()

    def _get_length_mask(self, lengths: List[int], seq_len: int) -> torch.Tensor:
        """Get mask for given actual lengths.
        Args:
            lengths: list of actual lengths.
            seq_len: length of sequence.
        Returns:
            mask: (batch, seq_len)
        """
        lengths = torch.tensor(lengths)
        length_mask = torch.arange(seq_len)[None, :] >= lengths[:, None]
        return length_mask

    def _get_spans(self, mask_indices: np.ndarray) -> np.ndarray:
        """Get spans of True values in mask_indices.
        Args:
            mask_indices: (seq_len) bool array
        Returns:
            spans: (num_spans, 2) array of (start, end) indices
        """
        mask_indices = np.concatenate([[False], mask_indices, [False]])
        diff = np.diff(mask_indices.astype(int))
        start_indices = np.nonzero(diff == 1)
        end_indices = np.nonzero(diff == -1)
        spans = np.stack([start_indices, end_indices], axis=1)
        return spans

    def _mask_batch_with_noise_spans(
        self, inputs: torch.Tensor, noise_spans: List[List[Tuple[int, int]]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask input batch with givne noise spans.
        Args:
            inputs: (batch_size, seq_len, num_features)
            noise_spans: list of noise spans for each batch, where each noise span is a tuple of (start, end).
        Returns:
            inputs: (batch_size, seq_len, num_features)
            labels: (batch_size, seq_len, num_features)
        """
        labels = inputs.clone()
        batch_size, seq_len, _ = inputs.shape
        mask_shape = (batch_size, seq_len)
        assert (
            len(noise_spans) == batch_size
        ), f"Noise spans should have the same length as batch size, but got {len(noise_spans)} != {batch_size}"

        # See all noise spans in the batch as a whole, and choose spans to mask, replace, and keep unchanged
        span_positions = []
        for item_index, spans in enumerate(noise_spans):
            span_positions += [(item_index, start, end) for start, end in spans]

        num_total_spans = len(span_positions)
        span_positions = np.array(span_positions)
        np.random.shuffle(span_positions)

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        num_masking_spans = int(num_total_spans * 0.8)
        masking_span_positions = span_positions[:num_masking_spans]
        for item_index, start, end in masking_span_positions:
            inputs[item_index, start:end] = self.mask_token_tensor

        # 10% of the time, we replace masked input tokens with random word
        num_replacing_spans = int(num_total_spans * 0.1)
        replacing_span_positions = span_positions[num_masking_spans : num_masking_spans + num_replacing_spans]
        replacing_indices = torch.zeros(mask_shape, dtype=torch.bool, device=inputs.device)
        for item_index, start, end in replacing_span_positions:
            replacing_indices[item_index, start:end] = True
        random_words = torch.stack(
            [torch.randint(size, mask_shape, dtype=inputs.dtype) for size in self.tokenizer.vocab_sizes],
            dim=-1,
        )
        inputs[replacing_indices] = random_words[replacing_indices]

        # Build boolean mask for all noise spans, and set labels
        noise_indices = torch.zeros(mask_shape, dtype=torch.bool, device=inputs.device)
        for item_index, spans in enumerate(noise_spans):
            for start, end in spans:
                noise_indices[item_index, start:end] = True
        labels[~noise_indices] = self.pad_token_tensor

        return MaskedData(inputs, labels)

    def mask(self, data: np.ndarray, note_map: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def mask_batch(self, inputs: torch.Tensor, note_maps: List[np.ndarray], lengths: List[int], **kwargs) -> MaskedData:
        raise NotImplementedError


class InfillingMasking(Masking):
    def mask_for_infilling(self, data: np.ndarray, note_map: np.ndarray, **kwargs) -> InfillingData:
        raise NotImplementedError

    def get_estimated_num_noise_spans(self, seq_len: int) -> int:
        num_noise_tokens = int(round(seq_len * self.corruption_rate))
        num_noise_tokens = min(max(num_noise_tokens, 1), seq_len - 1)
        mean_span_length = getattr(self, "mean_span_length", 5)
        return int(round(num_noise_tokens / mean_span_length))

    def get_estimated_infilling_seq_length(self, seq_len: int) -> int:
        # ([MASK] + <SEP>) per noise span, with some extra space
        return seq_len + 2 * (self.get_estimated_num_noise_spans(seq_len) + 5)

    def get_estimated_recovery_seq_length(self, seq_len: int) -> int:
        num_noise_tokens = int(round(seq_len * self.corruption_rate))
        num_noise_tokens = min(max(num_noise_tokens, 1), seq_len - 1)
        # corrupted part + <SEP> + original part + (some extra space)
        return (seq_len - num_noise_tokens) + 1 + seq_len + 10


class MultiTargetInfillingMasking(InfillingMasking):
    """Randomly choose a masking strategy for each batch."""

    def __init__(self, maskings: Tuple[InfillingMasking, ...], probabilities: Tuple[float, ...]):
        assert len(maskings) == len(probabilities), "Number of maskings should be the same as number of probabilities"
        self.maskings = maskings
        # normalize probabilities
        self.probabilities = np.array(probabilities) / sum(probabilities)
        self.need_to_mask_per_data = any(masking.need_to_mask_per_data for masking in maskings)

    def setup_tokenizer(self, tokenizer: MIDITokenizer):
        for masking in self.maskings:
            masking.setup_tokenizer(tokenizer)

    def mask(self, data: np.ndarray, note_map: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        index = np.random.choice(len(self.maskings), p=self.probabilities)
        masking = self.maskings[index]
        return masking.mask(data, note_map, **kwargs)

    def mask_batch(self, inputs: torch.Tensor, note_maps: List[np.ndarray], lengths: List[int], **kwargs) -> MaskedData:
        index = np.random.choice(len(self.maskings), p=self.probabilities)
        masking = self.maskings[index]
        return masking.mask_batch(inputs, note_maps, lengths, **kwargs)

    def mask_for_infilling(self, data: np.ndarray, note_map: np.ndarray, **kwargs) -> InfillingData:
        index = np.random.choice(len(self.maskings), p=self.probabilities)
        masking = self.maskings[index]
        return masking.mask_for_infilling(data, note_map, **kwargs)

    def get_estimated_num_noise_spans(self, seq_len: int) -> int:
        return max(masking.get_estimated_num_noise_spans(seq_len) for masking in self.maskings)

    def get_estimated_infilling_seq_length(self, seq_len: int) -> int:
        return max(masking.get_estimated_infilling_seq_length(seq_len) for masking in self.maskings)

    def get_estimated_recovery_seq_length(self, seq_len: int) -> int:
        return max(masking.get_estimated_recovery_seq_length(seq_len) for masking in self.maskings)


class RandomTokenMasking(Masking):
    def __init__(self, corruption_rate: float = 0.15):
        super().__init__()
        self.corruption_rate = corruption_rate

    def mask_batch(
        self, inputs: torch.Tensor, note_maps: List[np.ndarray], lengths: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask input batch. (BERT style)
        Args:
            inputs: (batch_size, seq_len, num_features)
            lengths: (batch_size)
        Returns:
            inputs: (batch_size, seq_len, num_features)
            labels: (batch_size, seq_len, num_features)
        Note:
            This function is modified from `transformers.data.data_collator.DataCollatorForLanguageModeling`.
        """
        labels = inputs.clone()
        batch_size, seq_len, _ = inputs.shape
        mask_shape = (batch_size, seq_len)

        problability_matrix = torch.full(mask_shape, self.corruption_rate)
        # Only mask the tokens that are not padding
        problability_matrix.masked_fill_(self._get_length_mask(lengths, seq_len), value=0.0)
        mask_indices = torch.bernoulli(torch.full(mask_shape, self.corruption_rate)).bool()
        labels[~mask_indices] = self.pad_token_tensor

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(mask_shape, 0.8)).bool() & mask_indices
        inputs[indices_replaced] = self.mask_token_tensor

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(mask_shape, 0.5)).bool() & mask_indices & ~indices_replaced
        random_words = torch.stack(
            [torch.randint(size, mask_shape, dtype=inputs.dtype) for size in self.tokenizer.vocab_sizes],
            dim=-1,
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return MaskedData(inputs, labels)


class SingleSpanMasking(InfillingMasking):
    need_to_mask_per_data = True

    def __init__(self, corruption_rate: float = 0.5):
        super().__init__()
        self.corruption_rate = corruption_rate

    def _get_random_span(self, length: int):
        num_noise_tokens = int(round(length * self.corruption_rate))
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        start = np.random.randint(0, length - num_noise_tokens)
        end = start + num_noise_tokens
        return start, end

    def mask_for_infilling(self, data: np.ndarray, note_map: np.ndarray, **kwargs) -> InfillingData:
        """Mask data with single span for infilling. Put a single mask token for the span.
        Args:
            data: (seq_len, num_features)
        """
        seq_len, _ = data.shape
        start, end = self._get_random_span(seq_len)
        nonnoise_spans = [data[:start], data[end:]]
        noise_spans = [data[start:end]]
        # target span index is always 1, since the sequence is {nonnoise, noise, nonnoise}
        noise_span_indices = [1]
        return InfillingData(nonnoise_spans, noise_spans, noise_span_indices, is_long_mask=True)

    def get_estimated_num_noise_spans(self, seq_len: int) -> int:
        return 1

    def get_estimated_infilling_seq_length(self, seq_len: int) -> int:
        # [MASK] + <SEP>
        return seq_len + 2


class VariableSpanMasking(InfillingMasking):
    need_to_mask_per_data = True

    def __init__(self):
        super().__init__()

    def _get_random_span(self, length: int):
        num_noise_tokens = np.random.randint(1, length - 1)
        start = np.random.randint(0, length - num_noise_tokens)
        end = start + num_noise_tokens
        return start, end

    def _remove_bos_eos(self, data: np.ndarray) -> np.ndarray:
        bos_mask = np.all(data == self.tokenizer.bos_token_ids, axis=-1)
        eos_mask = np.all(data == self.tokenizer.eos_token_ids, axis=-1)
        return data[~bos_mask & ~eos_mask]

    def _align_bars_(self, data: np.ndarray):
        bar_field_index = self.tokenizer.field_indices["bar"]
        bar_vocab_size = self.tokenizer.vocab_sizes[bar_field_index]
        note_mask = data[:, bar_field_index] < bar_vocab_size
        min_bar = np.min(data[note_mask, bar_field_index])
        data[note_mask, bar_field_index] -= min_bar

    def mask_for_infilling(self, data: np.ndarray, note_map: np.ndarray, **kwargs) -> InfillingData:
        """Mask data with single span for infilling. Put a single mask token for the span.
        Args:
            data: (seq_len, num_features)
        """
        data = self._remove_bos_eos(data)
        self._align_bars_(data)
        
        seq_len, _ = data.shape
        start, end = self._get_random_span(seq_len)
        nonnoise_spans = [data[:start], data[end:]]
        noise_spans = [data[start:end]]
        # target span index is always 1, since the sequence is {nonnoise, noise, nonnoise}
        noise_span_indices = [1]
        return InfillingData(nonnoise_spans, noise_spans, noise_span_indices, is_long_mask=True)

    def get_estimated_num_noise_spans(self, seq_len: int) -> int:
        return 1

    def get_estimated_infilling_seq_length(self, seq_len: int) -> int:
        # [MASK] + <SEP>
        return seq_len + 2


class RandomSpanMasking(InfillingMasking):
    def __init__(self, corruption_rate: float = 0.15, mean_span_length: int = 5):
        super().__init__()
        self.corruption_rate = corruption_rate
        self.mean_span_length = mean_span_length

    def _get_random_spans(self, length: int) -> List[Tuple[int, int]]:
        """Get random spans for masking.
        Args:
            length: length of the sequence
        Returns:
            spans: list of (start, end) indices, where the odd ones are noise spans.
        Note:
            This function is modified from
            `transformers.examples.flax.language-modeling.run_t5_mlm_flax.FlaxDataCollatorForT5MLM`.
        """
        num_noise_tokens = int(round(length * self.corruption_rate))
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_nonnoise_tokens = length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / self.mean_span_length))
        # make sure that the number of span pairs is at least equal to the number of noise/non-noise tokens
        num_noise_spans = min(max(num_noise_spans, 1), min(num_noise_tokens, num_nonnoise_tokens))

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items: int, num_segments: int):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape (num_segments) containing positive integers that add up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_starts = np.concatenate([[0], span_starts])
        span_ends = span_starts + interleaved_span_lengths
        return list(zip(span_starts, span_ends))

    def mask_batch(
        self, inputs: torch.Tensor, note_maps: List[np.ndarray], lengths: List[int], **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        noise_spans_list = []
        for i, length in enumerate(lengths):
            spans = self._get_random_spans(length)
            noise_spans = [span for i, span in enumerate(spans) if i % 2 == 1]
            noise_spans_list.append(noise_spans)

        return self._mask_batch_with_noise_spans(inputs, noise_spans_list)

    def mask_for_infilling(self, data: np.ndarray, note_map: np.ndarray, **kwargs) -> InfillingData:
        """Mask data with random spans for infilling. Put a single mask token for each span. (T5-style)
        Args:
            data: (seq_len, num_features)
        """
        seq_len, _ = data.shape
        spans = self._get_random_spans(seq_len)
        # Collect masked data and target data. Maybe there is a better way to do this.
        nonnoise_spans, noise_spans = [], []
        noise_span_indices = []
        for i, (start, end) in enumerate(spans):
            if i % 2 == 0:
                # non-noise span
                nonnoise_spans.append(data[start:end])
            else:
                # noise span
                noise_spans.append(data[start:end])
                noise_span_indices.append(i)
        return InfillingData(nonnoise_spans, noise_spans, noise_span_indices)


class RandomBarMasking(InfillingMasking):
    # for estimate whole sequence length
    mean_span_length = 5

    def __init__(self, corruption_rate: float = 0.15, extra_data_field_name: str = "bar_spans"):
        super().__init__()
        self.corruption_rate = corruption_rate
        self.extra_data_field_name = extra_data_field_name

    def setup_tokenizer(self, tokenizer: MIDITokenizer):
        super().setup_tokenizer(tokenizer)

    def _get_bar_token_spans(
        self, bar_note_spans: np.ndarray, note_map: np.ndarray, token_start: int, token_end: int
    ) -> np.ndarray:
        """Pick bar spans within given range and shift them to start from 0."""
        # get range of notes within the range of tokens
        note_start = (note_map["start"] >= token_start).nonzero()[0][0]
        note_end = (note_map["end"] <= token_end).nonzero()[0][-1] + 1

        # get bar spans within the range of notes
        filtered_bar_note_spans = bar_note_spans[
            (bar_note_spans["start"] >= note_start) & (bar_note_spans["end"] <= note_end)
        ]
        assert len(filtered_bar_note_spans) > 0, "No bar spans found."

        # convert bar spans to token spans and shift them to start from 0
        bar_token_spans = np.zeros_like(filtered_bar_note_spans)
        note_starts = filtered_bar_note_spans["start"]
        note_ends = filtered_bar_note_spans["end"] - 1
        bar_token_spans["start"] = note_map["start"][note_starts] - token_start
        bar_token_spans["end"] = note_map["end"][note_ends] - token_start

        # ensure that token indices of notes at two ends cover the whole sequence
        bar_token_spans["start"][note_starts == 0] = 0
        bar_token_spans["end"][note_ends == len(note_map) - 1] = token_end - token_start

        return bar_token_spans

    def _get_random_noise_bars(self, bar_token_spans: np.ndarray, length: int) -> np.ndarray:
        """Get random noise bars.
        Args:
            bar_token_spans: structured array with fields "start" and "end".
            length: an integer scalar, the length of the sequence.
        Returns:
            noise_bars: (num_bars) bool array, where True means noise bar.
        """
        num_bars = len(bar_token_spans)

        # Randomly select bars until we have enough noise bars
        random_bar_indices = np.arange(num_bars)
        np.random.shuffle(random_bar_indices)
        noise_bars = np.zeros(num_bars, dtype=bool)
        current_noise_tokens = 0
        for index in random_bar_indices:
            start, end = bar_token_spans[index]
            noise_bars[index] = True
            current_noise_tokens += end - start
            if current_noise_tokens / length >= self.corruption_rate:
                break

        return noise_bars

    def mask_batch(
        self, inputs: torch.Tensor, note_maps: List[np.ndarray], lengths: List[int], offsets: List[int], **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        noise_spans_list = []
        bars_list = kwargs[self.extra_data_field_name]
        for length, offset, bar_note_spans, note_map in zip(lengths, offsets, bars_list, note_maps):
            bar_token_spans = self._get_bar_token_spans(bar_note_spans, note_map, offset, offset + length)
            noise_bars = self._get_random_noise_bars(bar_token_spans, length)
            noise_spans_list.append(bar_token_spans[noise_bars])
        return self._mask_batch_with_noise_spans(inputs, noise_spans_list)

    def mask_for_infilling(self, data: np.ndarray, note_map: np.ndarray, offset: int, **kwargs) -> InfillingData:
        """Mask data with random bars for infilling. Put a single mask token for each bar.
        Args:
            data: (seq_len, num_features)
            offset: offset of data in the original sequence, in case random cropping is used
        """
        seq_len, _ = data.shape
        bar_note_spans = kwargs[self.extra_data_field_name]
        bar_token_spans = self._get_bar_token_spans(bar_note_spans, note_map, offset, offset + seq_len)
        noise_bars = self._get_random_noise_bars(bar_token_spans, seq_len)

        nonnoise_spans, noise_spans = [], []
        noise_span_indices = []
        for i, (start, end) in enumerate(bar_token_spans):
            if noise_bars[i]:
                # noise bar
                noise_spans.append(data[start:end])
                noise_span_indices.append(i)
            else:
                # non-noise bar
                nonnoise_spans.append(data[start:end])
        return InfillingData(nonnoise_spans, noise_spans, noise_span_indices)


class FixedBarMasking(InfillingMasking):
    need_to_mask_per_data = True

    def __init__(
        self,
        num_past_bars: int,
        num_middle_bars: int,
        num_future_bars: int,
        extra_data_field_name: str = "bar_spans",
        random_crop: bool = False,
        force_bar_length: bool = False,
    ):
        super().__init__()
        self.num_past_bars = num_past_bars
        self.num_middle_bars = num_middle_bars
        self.num_future_bars = num_future_bars
        self.num_total_bars = num_past_bars + num_middle_bars + num_future_bars
        self.extra_data_field_name = extra_data_field_name
        self.random_crop = random_crop
        self.force_bar_length = force_bar_length

    def _get_fixed_bar_spans(self, bar_note_spans: np.ndarray, note_map: np.ndarray):
        """Get fixed noise bar spans based on given numbers of past, masking and future bars.
        Args:
            bar_note_spans: (num_bars, 2) array of (start, end) indices
        Returns:
            past_start, past_end, future_start, future_end: indices of past, masking and future bars
        """
        num_bars = len(bar_note_spans)
        if self.force_bar_length:
            assert num_bars >= self.num_total_bars, f"num_bars ({num_bars}) < num_total_bars ({self.num_total_bars})"
        start_bar_index = (
            np.random.randint(num_bars - self.num_total_bars)
            if self.random_crop and num_bars > self.num_total_bars
            else 0
        )
        past_start_note = bar_note_spans[start_bar_index]["start"]
        past_end_note = bar_note_spans[start_bar_index + self.num_past_bars - 1]["end"]
        middle_start_note = bar_note_spans[start_bar_index + self.num_past_bars]["start"]
        middle_end_note = bar_note_spans[start_bar_index + self.num_past_bars + self.num_middle_bars - 1]["end"]
        future_start_note = bar_note_spans[start_bar_index + self.num_past_bars + self.num_middle_bars]["start"]
        # if didn't force bar length, future end note can be the end of the song
        future_end_bar = min(start_bar_index + self.num_total_bars - 1, num_bars - 1)
        future_end_note = bar_note_spans[future_end_bar]["end"]

        # convert note indices to token indices
        past_start = note_map[past_start_note]["start"]
        past_end = note_map[past_end_note - 1]["end"]
        middle_start = note_map[middle_start_note]["start"]
        middle_end = note_map[middle_end_note - 1]["end"]
        future_start = note_map[future_start_note]["start"]
        future_end = note_map[future_end_note - 1]["end"]

        return past_start, past_end, middle_start, middle_end, future_start, future_end

    def mask_for_infilling(self, data: np.ndarray, note_map: np.ndarray, **kwargs) -> InfillingData:
        """Mask data with fixed noise bars based on given numbers of past, masking and future bars.
        Random cropping should be handled here rather than in the data collator.
        Args:
            data: (seq_len, num_features)
            bar_spans: (num_bars, 2) array of (start, end) indices
        Returns:
            masked_data: (masked_seq_len, num_features), with form of {tokens, [MASK], tokens}
            target: (infilling_seq_len, num_features), with form of {<SEP>, tokens}
        """
        bar_note_spans = kwargs[self.extra_data_field_name]
        past_start, past_end, middle_start, middle_end, future_start, future_end = self._get_fixed_bar_spans(
            bar_note_spans, note_map
        )

        past = data[past_start:past_end]
        middle = data[middle_start:middle_end]
        future = data[future_start:future_end]
        nonnoise_spans = [past, future]
        noise_spans = [middle]
        # target span index is always 1, since the sequence is {past, middle, future}
        noise_span_indices = [1]
        return InfillingData(nonnoise_spans, noise_spans, noise_span_indices, is_long_mask=True)

    def get_estimated_num_noise_spans(self, seq_len: int) -> int:
        return 1

    def get_estimated_infilling_seq_length(self, seq_len: int) -> int:
        # [MASK] + <SEP>
        return seq_len + 2

    def get_estimated_recovery_seq_length(self, seq_len: int) -> int:
        # corrupted part + <SEP> + original part + (some extra space)
        return (seq_len - self.num_middle_bars * 2) + 1 + seq_len + 5


class RandomNgramMasking(InfillingMasking):
    # for estimate whole sequence length
    mean_span_length = 5

    def __init__(
        self,
        corruption_rate: float = 0.15,
        extra_data_field_name: str = "ngrams",
        fallback_mean_span_length: int = 5,
    ):
        """Args:
        corruption_rate: corruption rate of ngram masking.
        extra_data_field_name: name of the extra data field containing ngram spans, `pitch_ngrams` or `rhythm_ngrams`.
        fallback_mean_span_length: mean span length for random span method when ngrams are not available.
        """
        super().__init__()
        self.corruption_rate = corruption_rate
        self.extra_data_field_name = extra_data_field_name

        self.random_span_masking = RandomSpanMasking(corruption_rate, fallback_mean_span_length)

    def setup_tokenizer(self, tokenizer: MIDITokenizer):
        super().setup_tokenizer(tokenizer)
        self.random_span_masking.setup_tokenizer(tokenizer)

    def _get_ngrams(
        self, ngram_note_spans: np.ndarray, note_map: np.ndarray, token_start: int, token_end: int
    ) -> np.ndarray:
        """Pick ngram spans within given range and shift them to start from 0."""

        # get range of notes within the range of tokens
        note_start = (note_map["start"] >= token_start).nonzero()[0][0]
        note_end = (note_map["end"] <= token_end).nonzero()[0][-1] + 1

        # get ngram spans within the range of notes
        filtered_ngram_note_spans = ngram_note_spans[
            (ngram_note_spans["start"] >= note_start) & (ngram_note_spans["end"] <= note_end)
        ]

        # convert ngram spans to token spans and shift them to start from 0
        ngram_token_spans = filtered_ngram_note_spans.copy()
        note_starts = filtered_ngram_note_spans["start"]
        note_ends = filtered_ngram_note_spans["end"] - 1
        ngram_token_spans["start"] = note_map["start"][note_starts] - token_start
        ngram_token_spans["end"] = note_map["end"][note_ends] - token_start

        # ensure that token indices of notes at two ends cover the whole sequence
        ngram_token_spans["start"][note_starts == 0] = 0
        ngram_token_spans["end"][note_ends == len(note_map) - 1] = token_end - token_start

        return ngram_token_spans

    def _get_random_noise_ngrams(self, num_tokens: int, ngrams: np.ndarray) -> np.ndarray:
        """Get random ngrams.
        Args:
            num_tokens: length of sequence.
            ngrams: structured array with fields ["start", "end", "length", "rank"].
        Returns:
            noise_spans: (num_noise_spans, 2) array of (start, end) indices.
            has_enough_noise_ngrams: whether there are enough noise ngrams.
        """
        if len(ngrams) == 0:
            return None, False

        # Randomly select ngrams until the corruption rate is reached
        permutation = np.random.permutation(len(ngrams))
        covered_indices = np.zeros(num_tokens, dtype=bool)
        for index in permutation:
            start, end = ngrams[index]["start"], ngrams[index]["end"]
            covered_indices[start:end] = True
            current_noise_tokens = covered_indices.sum()
            if current_noise_tokens / num_tokens >= self.corruption_rate:
                break

        # Turn covered indices into spans
        covered_indices = np.concatenate([[False], covered_indices, [False]])
        indices = np.where(covered_indices[:-1] != covered_indices[1:])[0]
        starts, ends = indices[::2], indices[1::2]
        noise_spans = np.stack([starts, ends], axis=1).tolist()

        has_enough_noise_ngrams = current_noise_tokens / num_tokens >= self.corruption_rate
        return noise_spans, has_enough_noise_ngrams

    def mask_batch(
        self, inputs: torch.Tensor, note_maps: List[np.ndarray], lengths: List[int], offsets: List[int], **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        noise_spans_list = []
        ngrams_list = kwargs[self.extra_data_field_name]
        for length, offset, ngram_note_spans, note_map in zip(lengths, offsets, ngrams_list, note_maps):
            ngrams = self._get_ngrams(ngram_note_spans, note_map, offset, offset + length)
            noise_ngram_spans, has_enough_noise_ngrams = self._get_random_noise_ngrams(length, ngrams)

            if not has_enough_noise_ngrams:
                noise_ngram_spans = self.random_span_masking._get_random_spans(length)[1::2]
            noise_spans_list.append(noise_ngram_spans)

        return self._mask_batch_with_noise_spans(inputs, noise_spans_list)

    def mask_for_infilling(self, data: np.ndarray, note_map: np.ndarray, offset: int, **kwargs) -> InfillingData:
        """Mask data with random n-grams. Put a single mask token for each n-gram.
        Args:
            data: (seq_len, num_features)
            offset: offset of data in the original sequence, in case random cropping is used
        """
        seq_len, _ = data.shape
        ngram_note_spans = kwargs[self.extra_data_field_name]
        ngrams = self._get_ngrams(ngram_note_spans, note_map, offset, offset + seq_len)
        noise_ngram_spans, has_enough_noise_ngrams = self._get_random_noise_ngrams(seq_len, ngrams)

        if not has_enough_noise_ngrams:
            return self.random_span_masking.mask_for_infilling(data, note_map)

        # Build masked data and target
        nonnoise_spans, noise_spans = [], []
        noise_span_indices = []
        num_spans = 0
        previous_end = 0
        for start, end in noise_ngram_spans:
            if previous_end < start:
                nonnoise_spans.append(data[previous_end:start])
                num_spans += 1
            noise_spans.append(data[start:end])
            noise_span_indices.append(num_spans)
            num_spans += 1
            previous_end = end
        if previous_end < seq_len:
            nonnoise_spans.append(data[previous_end:])

        return InfillingData(
            nonnoise_spans,
            noise_spans,
            noise_span_indices,
        )


class DataCollator:
    def __init__(self, seq_len: int, random_crop: bool = False):
        self.tokenizer: MIDITokenizer
        self.seq_len = seq_len
        self.random_crop = random_crop

    def setup_tokenizer(self, tokenizer: MIDITokenizer):
        self.tokenizer = tokenizer

    def truncate(
        self, batch: List[np.ndarray], max_length: Optional[int], random_crop: bool = False
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Truncate batch to given maximum length.
        Args:
            batch: list of (seq_len, num_features)
            max_length: maximum length of the batch, if None, no truncation is performed.
            random_crop: whether to crop the batch randomly
        Returns:
            batch: list of (max_length, num_features)
            offsets: list of offsets
        """
        if max_length is None:
            return batch, [0] * len(batch)
        if random_crop:
            # randomly crop a segment of max_length
            # if the length of the batch is longer than max_length
            result = []
            offsets = []
            for data in batch:
                if len(data) > max_length:
                    start = np.random.randint(0, len(data) - max_length)
                    data = data[start : start + max_length]
                else:
                    start = 0
                result.append(data)
                offsets.append(start)
        else:
            result = [data[:max_length] for data in batch]
            offsets = [0] * len(batch)
        return result, offsets

    def pad(self, batch: List[np.ndarray], max_length: Optional[int] = None) -> List[np.ndarray]:
        """Pad batch to given maximum length.
        Args:
            batch: list of (seq_len, num_features)
            max_length: maximum length of the batch, if None, use the length of the longest sequence
        Returns:
            batch: list of (max_length, num_features)
        """
        result = []
        max_length = max_length or max(len(data) for data in batch)
        for data in batch:
            assert (
                len(data) <= max_length
            ), f"length of data should be less than max_length, but got {len(data)} > {max_length}"
            if len(data) < max_length:
                pad = np.full((max_length - len(data), data.shape[1]), self.tokenizer.pad_token_ids, dtype=data.dtype)
                data = np.concatenate([data, pad], axis=0)
            result.append(data)
        return result


class DataCollatorForPaddingOnly(DataCollator):
    """Data collator for padding only, useful in inference stage."""

    def __init__(self, seq_len: int, empty: bool = False):
        super().__init__(seq_len)
        # Provide empty dummy data for from scratch generation
        self.empty = empty

    def __call__(self, batch: List[DatasetItem]) -> DataBatch:
        if self.empty:
            return DataBatch(input_ids=np.empty((1, 0, 0), dtype=np.int64))
        
        # TODO: Consider right side padding for batch inference?
        data_list = [item.data for item in batch]
        filenames = [item.filename for item in batch]
        data_list, _ = self.truncate(data_list, self.seq_len)
        input_ids = np.stack(self.pad(data_list, self.seq_len), axis=0)
        input_ids = torch.from_numpy(input_ids).long()
        lengths = [len(data) for data in data_list]

        return DataBatch(input_ids, label_ids=None, attention_kind="causal", lengths=lengths, filenames=filenames)


class DataCollatorForFixedInfilling(DataCollator):
    """Data collator for fixed infilling task, intended for prefix-style inference.
    Support fixed bar masking for now."""

    def __init__(self, masking: FixedBarMasking, seq_len: int):
        super().__init__(seq_len)
        self.masking = masking

    def setup_tokenizer(self, tokenizer: MIDITokenizer):
        super().setup_tokenizer(tokenizer)
        self.masking.setup_tokenizer(tokenizer)

    def __call__(self, batch: List[DatasetItem]) -> DataBatch:
        # Only collect masked data
        masked_data_list = []
        for item in batch:
            sources = self.masking.mask_for_infilling(item.data, item.note_map, **item.extra_data).nonnoise_spans
            assert len(sources) == 2, "Fixed bar masking should generate past and future spans."
            masked_data = np.concatenate([sources[0], [self.tokenizer.long_mask_token_ids], sources[1]], axis=0)
            masked_data_list.append(masked_data)
        input_ids = np.stack(self.pad(masked_data_list), axis=0)
        input_ids = torch.from_numpy(input_ids).long()

        lengths = [len(data) for data in masked_data_list]
        filenames = [item.filename for item in batch]
        return DataBatch(input_ids, label_ids=None, lengths=lengths, filenames=filenames)


class DataCollatorForCausalLanguageModeling(DataCollator):
    def __init__(self, seq_len: int, random_crop: bool = False):
        super().__init__(seq_len, random_crop)

    def __call__(self, batch: List[DatasetItem]) -> DataBatch:
        data_list = [item.data for item in batch]
        filenames = [item.filename for item in batch]

        # truncate to (seq_len + 1), while effective length is still seq_len
        data_list, _ = self.truncate(data_list, self.seq_len + 1, random_crop=self.random_crop)
        batched_data = np.stack(self.pad(data_list, self.seq_len + 1), axis=0)
        input_ids = torch.from_numpy(batched_data[:, :-1]).long()
        label_ids = torch.from_numpy(batched_data[:, 1:]).long()
        lengths = [len(data) - 1 for data in data_list]

        return DataBatch(input_ids, label_ids, attention_kind="causal", lengths=lengths, filenames=filenames)


class DataCollatorForMaskedLanguageModeling(DataCollator):
    def __init__(self, masking: Masking, seq_len: int, random_crop: bool = False):
        super().__init__(seq_len, random_crop)
        self.masking = masking

    def setup_tokenizer(self, tokenizer: MIDITokenizer):
        super().setup_tokenizer(tokenizer)
        self.masking.setup_tokenizer(tokenizer)

    def __call__(self, batch: List[DatasetItem]) -> DataBatch:
        data_list = [item.data for item in batch]
        extra_data_list = [item.extra_data for item in batch]
        note_maps = [item.note_map for item in batch]

        # truncate
        data_list, offsets = self.truncate(data_list, self.seq_len, random_crop=self.random_crop)
        lengths = [len(data) for data in data_list]

        if self.masking.need_to_mask_per_data:
            # mask each data separately, and then pad
            inputs, labels = [], []
            for data, note_map, extra_data, offset in zip(data_list, note_maps, extra_data_list, offsets):
                masked_data, label = self.masking.mask(data, note_map, offset=offset, **extra_data)
                inputs.append(masked_data)
                labels.append(label)
            input_ids = torch.from_numpy(np.stack(self.pad(inputs, self.seq_len), axis=0)).long()
            label_ids = torch.from_numpy(np.stack(self.pad(labels, self.seq_len), axis=0)).long()
        else:
            # pad, and then mask all data together
            batch_data = torch.from_numpy(np.stack(self.pad(data_list, self.seq_len), axis=0)).long()
            # get dict of extra data list from list of extra data dict
            extra_data = {k: [extra_data[k] for extra_data in extra_data_list] for k in extra_data_list[0]}
            input_ids, label_ids = self.masking.mask_batch(
                batch_data, note_maps, lengths, offsets=offsets, **extra_data
            )

        filenames = [item.filename for item in batch]
        return DataBatch(input_ids, label_ids, attention_kind="full", lengths=lengths, filenames=filenames)


class DataCollatorForInfilling(DataCollator):
    def __init__(
        self,
        masking: InfillingMasking,
        seq_len: int,
        random_crop: bool = False,
        permutated_infilling: bool = False,
    ):
        super().__init__(seq_len, random_crop)
        self.masking = masking
        self.permutated_infilling = permutated_infilling

        self.whole_seq_length = masking.get_estimated_infilling_seq_length(seq_len)
        print("estimated whole sequence length:", self.whole_seq_length)

    def setup_tokenizer(self, tokenizer: MIDITokenizer):
        super().setup_tokenizer(tokenizer)
        self.masking.setup_tokenizer(tokenizer)

    def _get_source_and_target(
        self,
        nonnoise_spans: List[np.ndarray],
        noise_spans: List[np.ndarray],
        noise_span_indices: List[int],
        permutated: bool,
        is_long_mask: bool,
    ) -> Tuple[np.ndarray, np.ndarray, List[int], List[int], List[int]]:
        num_spans = len(nonnoise_spans) + len(noise_spans)
        noise_span_indices = sorted(noise_span_indices)
        nonnoise_span_indices = set(range(num_spans)) - set(noise_span_indices)
        nonnoise_span_indices = sorted(nonnoise_span_indices)
        assert len(noise_span_indices) == len(noise_spans) and len(nonnoise_span_indices) == len(nonnoise_spans)

        source_list, target_list = [], []
        # record position of corresponding [MASK] token in source sequence for each noise span
        noise_span_mask_positions = []
        current_nonnoise, current_noise = 0, 0
        current_source_position = 0
        for span_index in range(num_spans):
            if current_noise < len(noise_spans) and span_index == noise_span_indices[current_noise]:
                source_list.append(
                    [self.tokenizer.long_mask_token_ids if is_long_mask else self.tokenizer.mask_token_ids]
                )
                target_list.append(np.concatenate(([self.tokenizer.sep_token_ids], noise_spans[current_noise]), axis=0))
                noise_span_mask_positions.append(current_source_position)
                current_source_position += 1
                current_noise += 1
            else:
                source_list.append(nonnoise_spans[current_nonnoise])
                current_source_position += len(nonnoise_spans[current_nonnoise])
                current_nonnoise += 1
        assert current_nonnoise == len(nonnoise_spans) and current_noise == len(noise_spans)

        if permutated and len(noise_spans) > 1:
            permutation = np.random.permutation(len(noise_spans))
            target_list = [target_list[i] for i in permutation]
            noise_span_mask_positions = [noise_span_mask_positions[i] for i in permutation]

        source = np.concatenate(source_list, axis=0)
        target = np.concatenate(target_list, axis=0)
        noise_span_lengths = [len(target) for target in target_list]
        sep_positions = [0] + np.cumsum(noise_span_lengths)[:-1].tolist()

        # # reorder sep positions back to original, so that sep postitions are corresponding to mask positions
        # if permutated and len(noise_spans) > 1:
        #     sep_positions = [sep_positions[i] for i in np.argsort(permutation)]

        return source, target, sep_positions, noise_span_mask_positions, noise_span_lengths

    def _get_input_and_label(
        self,
        source: np.ndarray,
        target: np.ndarray,
        sep_positions: List[int],
        field_padding_indices: Optional[List[int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if field_padding_indices is not None:
            target_labels = target.copy()
            # pad target with field padding indices, only on non [SEP] positions
            sep_position_mask = np.ones(len(target), dtype=np.bool)
            sep_position_mask[sep_positions] = False
            target_labels[np.ix_(sep_position_mask, field_padding_indices)] = self.tokenizer.pad_token_ids[
                field_padding_indices
            ]
        else:
            target_labels = target

        input = np.concatenate([source, target], axis=0)
        label = np.concatenate(
            [
                np.full_like(source, self.tokenizer.pad_token_ids),
                target_labels[1:],
                [self.tokenizer.sep_token_ids],
            ],
            axis=0,
        )
        return input, label

    def get_positional_ids(
        self, source_length: int, noise_span_mask_positions: List[int], noise_span_lengths: List[int]
    ) -> np.ndarray:
        source_part = np.arange(source_length)
        target_parts = [
            np.ones(length, dtype=np.int64) * mask_position
            for mask_position, length in zip(noise_span_mask_positions, noise_span_lengths)
        ]
        # +1 since 0 is used for padding
        return np.concatenate([source_part] + target_parts, axis=0) + 1

    def pad_positional_ids(
        self, positional_ids_list: List[np.ndarray], max_length: Optional[int] = None
    ) -> List[np.ndarray]:
        result = []
        max_length = max_length or max(len(data) for data in positional_ids_list)
        for data in positional_ids_list:
            assert (
                len(data) <= max_length
            ), f"length of data should be less than max_length, but got {len(data)} > {max_length}"
            if len(data) < max_length:
                pad = np.ones(max_length - len(data), dtype=data.dtype) * positional_id_padding_index
                data = np.concatenate([data, pad], axis=0)
            result.append(data)
        return result

    def __call__(self, batch: List[DatasetItem]) -> DataBatch:
        data_list = [item.data for item in batch]
        extra_data_list = [item.extra_data for item in batch]
        note_maps = [item.note_map for item in batch]

        # truncate
        data_list, offsets = self.truncate(data_list, self.seq_len, random_crop=self.random_crop)

        # collect all the data, and genereate the inputs and labels
        inputs, labels = [], []
        positional_ids_list = []
        source_lengths, input_lengths = [], []
        for data, note_map, extra_data, offset in zip(data_list, note_maps, extra_data_list, offsets):
            infilling_data = self.masking.mask_for_infilling(data, note_map, offset=offset, **extra_data)

            source, target, sep_positions, noise_span_mask_positions, noise_span_lengths = self._get_source_and_target(
                infilling_data.nonnoise_spans,
                infilling_data.noise_spans,
                noise_span_indices=infilling_data.noise_span_indices,
                permutated=self.permutated_infilling,
                is_long_mask=infilling_data.is_long_mask,
            )
            input, label = self._get_input_and_label(
                source,
                target,
                sep_positions=sep_positions,
                field_padding_indices=infilling_data.field_padding_indices,
            )
            if self.permutated_infilling:
                positional_ids = self.get_positional_ids(len(source), noise_span_mask_positions, noise_span_lengths)
                assert len(positional_ids) == len(input)
                positional_ids_list.append(positional_ids)

            inputs.append(input)
            labels.append(label)
            source_lengths.append(len(source))
            input_lengths.append(len(input))

        # pad
        max_length = max(input_lengths)
        if max_length > self.whole_seq_length:
            print(f"adjusting whole_seq_length from {self.whole_seq_length} to {max_length}...")
            self.whole_seq_length = max_length
        input_ids = torch.from_numpy(np.stack(self.pad(inputs, self.whole_seq_length), axis=0)).long()
        label_ids = torch.from_numpy(np.stack(self.pad(labels, self.whole_seq_length), axis=0)).long()

        # positional ids for permutated infilling
        positional_ids = None
        if self.permutated_infilling:
            positional_ids = torch.from_numpy(
                np.stack(self.pad_positional_ids(positional_ids_list, self.whole_seq_length), axis=0)
            ).long()

        filenames = [item.filename for item in batch]
        return DataBatch(
            input_ids,
            label_ids,
            attention_kind="prefix",
            lengths=input_lengths,
            source_lengths=source_lengths,
            filenames=filenames,
            positional_ids=positional_ids,
        )


class DataCollatorForRecovery(DataCollator):
    def __init__(
        self,
        masking: InfillingMasking,
        seq_len: int,
        random_crop: bool = False,
        random_mask_ratio: float = 0,
        random_replace_ratio: float = 0,
    ):
        super().__init__(seq_len, random_crop)
        self.masking = masking

        assert 0 <= random_mask_ratio <= 1 and 0 <= random_replace_ratio <= 1
        self.random_mask_ratio = random_mask_ratio
        self.random_replace_ratio = random_replace_ratio
        self.random_delete_ratio = 1 - random_mask_ratio - random_replace_ratio

        self.whole_seq_length = round(
            (self.random_delete_ratio + self.random_mask_ratio) * masking.get_estimated_recovery_seq_length(seq_len)
            + self.random_mask_ratio * masking.get_estimated_num_noise_spans(seq_len)
            + self.random_replace_ratio * (seq_len * 2 + 1)
        )
        print("estimated whole sequence length:", self.whole_seq_length)

    def setup_tokenizer(self, tokenizer: MIDITokenizer):
        super().setup_tokenizer(tokenizer)
        self.masking.setup_tokenizer(tokenizer)

    def _get_random_noise_method(self) -> str:
        return np.random.choice(
            ["mask", "replace", "delete"],
            p=[self.random_mask_ratio, self.random_replace_ratio, self.random_delete_ratio],
        )

    def _get_source_and_target(
        self,
        nonnoise_spans: List[np.ndarray],
        noise_spans: List[np.ndarray],
        noise_span_indices: List[int],
        is_long_mask: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_spans = len(nonnoise_spans) + len(noise_spans)
        noise_span_indices = sorted(noise_span_indices)
        nonnoise_span_indices = set(range(num_spans)) - set(noise_span_indices)
        nonnoise_span_indices = sorted(nonnoise_span_indices)
        assert len(noise_span_indices) == len(noise_spans) and len(nonnoise_span_indices) == len(nonnoise_spans)

        # rearrange source and target spans into one array
        spans = [None for _ in range(num_spans)]
        for index, nonnoise_span in zip(nonnoise_span_indices, nonnoise_spans):
            spans[index] = nonnoise_span
        for index, noise_span in zip(noise_span_indices, noise_spans):
            spans[index] = noise_span

        # create source and target
        source_list, target_list = [], []
        for i, span in enumerate(spans):
            if i in nonnoise_span_indices:
                source_list.append(span)
                target_list.append(span)
            else:
                target_list.append(span)
                # choose noise method: delete, mask or replace
                noise_method = self._get_random_noise_method()
                if noise_method == "delete":
                    # delete noise span in source
                    pass
                elif noise_method == "mask":
                    source_list.append(
                        [self.tokenizer.long_mask_token_ids if is_long_mask else self.tokenizer.mask_token_ids]
                    )
                elif noise_method == "replace":
                    raise NotImplementedError

        if len(source_list) > 0:
            source = np.concatenate(source_list, axis=0)
        else:
            # allow empty source, task turns into pure CLM
            source = np.empty((0, len(self.tokenizer.field_names)))
        target = np.concatenate(target_list, axis=0)
        return source, target

    def _get_input_and_label(
        self, source: np.ndarray, target: np.ndarray, field_padding_indices: Optional[List[int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if field_padding_indices is not None:
            target_labels = target.copy()
            target_labels[:, field_padding_indices] = self.tokenizer.pad_token_ids[field_padding_indices]
        else:
            target_labels = target
        input = np.concatenate([source, [self.tokenizer.sep_token_ids], target], axis=0)
        label = np.concatenate(
            [
                np.full_like(source, self.tokenizer.pad_token_ids),
                target_labels,
                [self.tokenizer.sep_token_ids],
            ],
            axis=0,
        )
        return input, label

    def __call__(self, batch: List[DatasetItem]) -> DataBatch:
        data_list = [item.data for item in batch]
        note_maps = [item.note_map for item in batch]
        extra_data_list = [item.extra_data for item in batch]

        # truncate
        data_list, offsets = self.truncate(data_list, self.seq_len, random_crop=self.random_crop)

        inputs, labels = [], []
        source_lengths, input_lengths = [], []
        for data, note_map, extra_data, offset in zip(data_list, note_maps, extra_data_list, offsets):
            infilling_data = self.masking.mask_for_infilling(data, note_map, offset=offset, **extra_data)

            source, target = self._get_source_and_target(
                infilling_data.nonnoise_spans,
                infilling_data.noise_spans,
                infilling_data.noise_span_indices,
                is_long_mask=infilling_data.is_long_mask,
            )
            input, label = self._get_input_and_label(
                source, target, field_padding_indices=infilling_data.field_padding_indices
            )

            inputs.append(input)
            labels.append(label)
            source_lengths.append(len(source))
            input_lengths.append(len(input))

        # pad
        max_length = max(input_lengths)
        if max_length > self.whole_seq_length:
            print(f"adjusting whole_seq_length from {self.whole_seq_length} to {max_length}...")
            self.whole_seq_length = max_length
        input_ids = torch.from_numpy(np.stack(self.pad(inputs, self.whole_seq_length), axis=0)).long()
        label_ids = torch.from_numpy(np.stack(self.pad(labels, self.whole_seq_length), axis=0)).long()

        filenames = [item.filename for item in batch]
        return DataBatch(
            input_ids,
            label_ids,
            attention_kind="prefix",
            lengths=input_lengths,
            source_lengths=source_lengths,
            filenames=filenames,
        )


class MelodyDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        load_bar_data: bool = False,
        load_ngram_data: bool = False,
        pitch_augumentation: bool = False,
    ):
        self.tokenizer: MIDITokenizer
        self.files = glob(os.path.join(data_dir, "*.npz"))
        self.load_bar_data = load_bar_data
        self.load_ngram_data = load_ngram_data
        self.pitch_augumentation = pitch_augumentation

    def setup_tokenizer(self, tokenizer: MIDITokenizer):
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = np.load(self.files[idx])
        filename = os.path.splitext(os.path.basename(self.files[idx]))[0]
        data = file["data"]
        note_map = file["note_map"]
        if self.pitch_augumentation:
            self.tokenizer.pitch_shift_augument_(data)
        extra_data = {}
        if self.load_bar_data:
            extra_data["bar_spans"] = file["bar_spans"]
        if self.load_ngram_data:
            if "ngrams" in file:
                extra_data["ngrams"] = file["ngrams"]
            else:
                extra_data["pitch_ngrams"] = file["pitch_ngrams"]
                extra_data["rhythm_ngrams"] = file["rhythm_ngrams"]
        return DatasetItem(data, note_map, extra_data, filename)


class MelodyPretrainDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        batch_size: int,
        num_workers: int = 0,
        load_bar_data: bool = False,
        load_ngram_data: bool = False,
        pitch_augumentation: bool = True,
        times_to_predict: int = 1,
        debug: bool = False,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        tokenizer_config_path = os.path.join(dataset_dir, "tokenizer_config.json")
        if not os.path.exists(tokenizer_config_path):
            raise ValueError(f"Tokenizer config file not found: {tokenizer_config_path}")
        self.tokenizer = MIDITokenizer.from_config(tokenizer_config_path)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.load_bar_data = load_bar_data
        self.load_ngram_data = load_ngram_data
        self.pitch_augumentation = pitch_augumentation
        self.times_to_predict = times_to_predict
        self.debug = debug

        self.data_collators: Dict[str, DataCollator] = {}

    def _make_dataset(self, data_dir: str):
        dataset = MelodyDataset(
            data_dir,
            load_bar_data=self.load_bar_data,
            load_ngram_data=self.load_ngram_data,
            pitch_augumentation=self.pitch_augumentation,
        )
        dataset.setup_tokenizer(self.tokenizer)
        return dataset

    def _make_data_loader(self, split_name: str, collator: DataCollator):
        # batch_size=1 for prediction currently
        return DataLoader(
            self.datasets[split_name],
            batch_size=self.batch_size if split_name != "predict" else 1,
            shuffle=split_name == "train" and not self.debug,
            drop_last=split_name == "train",
            collate_fn=collator,
            num_workers=self.num_workers if not self.debug else 0,
            pin_memory=True and not self.debug,
        )

    def _make_data_loaders(self, split_name: str):
        if split_name not in self.datasets:
            return []
        if split_name != "train" and len(self.data_collators) > 1:
            raise ValueError("Only one task is supported for evaluation.")

        if len(self.data_collators) == 1:
            return self._make_data_loader(split_name, next(iter(self.data_collators.values())))
        else:
            return {
                task_name: self._make_data_loader(split_name, collator)
                for task_name, collator in self.data_collators.items()
            }

    def setup(self, stage: str):
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.valid_dir = os.path.join(self.dataset_dir, "valid")
        self.test_dir = os.path.join(self.dataset_dir, "test")
        self.datasets = {"train": self._make_dataset(self.train_dir)}
        if os.path.exists(self.valid_dir):
            self.datasets["valid"] = self._make_dataset(self.valid_dir)
        if os.path.exists(self.test_dir):
            self.datasets["test"] = self._make_dataset(self.test_dir)
            self.datasets["predict"] = self._make_dataset(self.test_dir)

    def register_task(self, task_name: str, collator: DataCollator):
        self.data_collators[task_name] = collator
        self.data_collators[task_name].setup_tokenizer(self.tokenizer)

    def train_dataloader(self):
        return self._make_data_loaders("train")

    def val_dataloader(self):
        return self._make_data_loaders("valid")

    def test_dataloader(self):
        return self._make_data_loaders("test")

    def predict_dataloader(self):
        if self.times_to_predict > 1:
            return [self._make_data_loaders("predict") for _ in range(self.times_to_predict)]
        return self._make_data_loaders("predict")
