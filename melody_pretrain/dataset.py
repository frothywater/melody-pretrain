import os
from glob import glob
from typing import Dict, List, NamedTuple, Optional, Tuple

import lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .tokenizer import MIDITokenizer


default_padding_index_for_classification = -100


class DatasetItem(NamedTuple):
    data: np.ndarray
    extra_data: Dict[str, np.ndarray]


class InfillingMaskedDataPair(NamedTuple):
    """Used for text infilling, the data is a pair of (masked data, target).
    Allow extra data to be attached to the data pair.
    """

    masked_data: np.ndarray
    target: np.ndarray
    extra_label_ids: Optional[np.ndarray] = None
    ngram_type: Optional[str] = None


class DatasetBatch(NamedTuple):
    """A batch of data for training. Allow extra data to be attached to the batch.
    Note that in PyTorch's padding mask and attention mask, True means to ignore."""

    input_ids: torch.Tensor
    label_ids: torch.Tensor
    padding_mask: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    extra_label_ids: Optional[torch.Tensor] = None
    ngram_types: Optional[List[str]] = None


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

    def _get_special_tokens_mask_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Get special tokens mask for data.
        Args:
            data: (batch, seq_len, num_features)
        Returns:
            special_tokens_mask: (batch, seq_len)
        """
        # (batch, seq_len, num_features, num_tokens)
        batch = batch.unsqueeze(-1)
        candidates = torch.from_numpy(self.tokenizer.special_token_id_matrix).long()
        candidates = candidates[None, None, :, :].expand_as(batch)
        # If any of the features matches any of the special tokens, then it is a special token
        special_tokens_mask = (batch == candidates).any(dim=-1)
        special_tokens_mask = special_tokens_mask.any(dim=-1)
        return special_tokens_mask

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

        mask_indices = torch.zeros(mask_shape, dtype=torch.bool, device=inputs.device)
        for i, spans in enumerate(noise_spans):
            for start, end in spans:
                mask_indices[i, start:end] = True

        labels[~mask_indices] = self.pad_token_tensor
        inputs[mask_indices] = self.mask_token_tensor
        return inputs, labels

    def mask(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def mask_batch(self, inputs: torch.Tensor, lengths: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class InfillingMasking(Masking):
    def mask_for_infilling(self, data: np.ndarray, **kwargs) -> InfillingMaskedDataPair:
        raise NotImplementedError


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

    def mask(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        index = np.random.choice(len(self.maskings), p=self.probabilities)
        masking = self.maskings[index]
        return masking.mask(data, **kwargs)

    def mask_for_infilling(self, data: np.ndarray, **kwargs) -> InfillingMaskedDataPair:
        index = np.random.choice(len(self.maskings), p=self.probabilities)
        masking = self.maskings[index]
        return masking.mask_for_infilling(data, **kwargs)


class RandomTokenMasking(Masking):
    def __init__(self, corruption_rate: float = 0.15):
        super().__init__()
        self.corruption_rate = corruption_rate

    def mask_batch(self, inputs: torch.Tensor, lengths: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return inputs, labels


class SingleSpanMasking(InfillingMasking):
    def __init__(self, corruption_rate: float = 0.5):
        super().__init__()
        self.corruption_rate = corruption_rate

    def _get_random_span(self, length: int):
        num_noise_tokens = int(round(length * self.corruption_rate))
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        start = np.random.randint(0, length - num_noise_tokens)
        end = start + num_noise_tokens
        return start, end

    def mask_batch(self, inputs: torch.Tensor, lengths: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask input batch with single span. (MASS-style)
        Args:
            inputs: (batch_size, seq_len, num_features)
            lengths: (batch_size)
        Returns:
            inputs: (batch_size, seq_len, num_features)
            labels: (batch_size, seq_len, num_features)
        """
        noise_spans = [[self._get_random_span(length)] for length in lengths]
        return self._mask_batch_with_noise_spans(inputs, noise_spans)

    def mask_for_infilling(self, data: np.ndarray, **kwargs) -> InfillingMaskedDataPair:
        """Mask data with single span for infilling. Put a single mask token for the span.
        Args:
            data: (seq_len, num_features)
        Returns:
            masked_data: (masked_seq_len, num_features), with form of {tokens, [MASK], tokens}
            target: (infilling_seq_len, num_features), with form of {<SEP>, tokens}
        """
        seq_len, _ = data.shape
        start, end = self._get_random_span(seq_len)
        masked_data = np.concatenate((data[:start], [self.tokenizer.mask_token_ids], data[end:]))
        target = np.concatenate(([self.tokenizer.sep_token_ids], data[start:end]))
        return InfillingMaskedDataPair(masked_data, target)


class RandomSpanMasking(InfillingMasking):
    def __init__(self, corruption_rate: float = 0.15, mean_span_length: int = 3):
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
        num_noise_spans = int(round(num_noise_tokens / self.mean_span_length))
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

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

    def mask_batch(self, inputs: torch.Tensor, lengths: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask input batch with random spans. (SpanBERT-style)
        Args:
            inputs: (batch_size, seq_len, num_features)
            lengths: (batch_size)
        Returns:
            inputs: (batch_size, seq_len, num_features)
            labels: (batch_size, seq_len, num_features)
        """
        # TODO: Use BERT-style masking.
        noise_spans = [self._get_random_spans(length)[1::2] for length in lengths]
        return self._mask_batch_with_noise_spans(inputs, noise_spans)

    def mask_for_infilling(self, data: np.ndarray, **kwargs) -> InfillingMaskedDataPair:
        """Mask data with random spans for infilling. Put a single mask token for each span. (T5-style)
        Args:
            data: (seq_len, num_features)
        Returns:
            masked_data: (masked_seq_len, num_features), with form of {tokens, [MASK], ...}
            target: (infilling_seq_len, num_features), with form of {<SEP>, tokens, ...}
        """
        seq_len, _ = data.shape
        spans = self._get_random_spans(seq_len)
        # Collect masked data and target data. Maybe there is a better way to do this.
        masked_data, target = [], []
        for i, (start, end) in enumerate(spans):
            if i % 2 == 0:
                # non-noise span
                masked_data += [data[start:end], [self.tokenizer.mask_token_ids]]
            else:
                # noise span
                target += [[self.tokenizer.sep_token_ids], data[start:end]]
        masked_data = np.concatenate(masked_data)
        target = np.concatenate(target)
        return InfillingMaskedDataPair(masked_data, target)


class RandomBarMasking(InfillingMasking):
    need_to_mask_per_data = True

    def __init__(self, corruption_rate: float = 0.15, extra_data_field_name: str = "bar_spans"):
        super().__init__()
        self.corruption_rate = corruption_rate
        self.extra_data_field_name = extra_data_field_name

    def setup_tokenizer(self, tokenizer: MIDITokenizer):
        super().setup_tokenizer(tokenizer)
        self.bar_field_index = tokenizer.field_names.index("bar")
        self.bar_vocab_size = tokenizer.vocab_sizes[self.bar_field_index]

    def _process_bar_spans(self, bar_spans: np.ndarray, start: int, end: int) -> np.ndarray:
        """Pick bar spans within given range and shift them to start from 0."""
        bar_spans = bar_spans[(bar_spans[:, 0] >= start) & (bar_spans[:, 1] <= end)]
        assert len(bar_spans) > 0, "No bar spans found."
        bar_spans = bar_spans - start
        return bar_spans

    def _get_random_noise_bars(self, bar_spans: np.ndarray, length: int) -> np.ndarray:
        """Get random noise bars.
        Args:
            bar_spans: (num_bars, 2) array, where each row is a bar span.
            length: an integer scalar, the length of the sequence.
        Returns:
            noise_bars: (num_bars) bool array, where True means noise bar.
        """
        num_bars = len(bar_spans)
        num_noise_tokens = int(round(length * self.corruption_rate))
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)

        # Randomly select bars until we have enough noise bars
        random_bar_indices = np.arange(num_bars)
        np.random.shuffle(random_bar_indices)
        noise_bars = np.zeros(num_bars, dtype=bool)
        current_noise_tokens = 0
        for index in random_bar_indices:
            if current_noise_tokens >= num_noise_tokens:
                break
            noise_bars[index] = True
            start, end = bar_spans[index]
            current_noise_tokens += end - start

        return noise_bars

    def mask(self, data: np.ndarray, offset: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Mask data with random bars. (MusicBERT-style)
        Args:
            data: (seq_len, num_features)
            bar_spans: (num_bars, 2) array of (start, end) indices
            offset: offset of data in the original sequence, in case random cropping is used
        Returns:
            masked_data: (seq_len, num_features)
            label: (seq_len, num_features)
        """
        # TODO: Use BERT-style masking.
        label = data.copy()
        seq_len, _ = data.shape

        bar_spans = kwargs[self.extra_data_field_name]
        bar_spans = self._process_bar_spans(bar_spans, offset, offset + seq_len)
        noise_bars = self._get_random_noise_bars(bar_spans, seq_len)
        noise_bar_spans = bar_spans[noise_bars]
        mask_indices = np.zeros(seq_len, dtype=bool)
        for start, end in noise_bar_spans:
            mask_indices[start:end] = True

        data[mask_indices] = self.tokenizer.mask_token_ids
        label[~mask_indices] = self.tokenizer.pad_token_ids
        return data, label

    def mask_for_infilling(self, data: np.ndarray, offset: int, **kwargs) -> InfillingMaskedDataPair:
        """Mask data with random bars for infilling. Put a single mask token for each bar.
        Args:
            data: (seq_len, num_features)
            bar_spans: (num_bars, 2) array of (start, end) indices
            offset: offset of data in the original sequence, in case random cropping is used
        Returns:
            masked_data: (masked_seq_len, num_features), with form of {tokens, [MASK], ...}
            target: (infilling_seq_len, num_features), with form of {<SEP>, tokens, ...}
        """
        seq_len, _ = data.shape
        bar_spans = kwargs[self.extra_data_field_name]
        bar_spans = self._process_bar_spans(bar_spans, offset, offset + seq_len)
        noise_bars = self._get_random_noise_bars(bar_spans, seq_len)

        masked_data, target = [], []
        for i, (start, end) in enumerate(bar_spans):
            if noise_bars[i]:
                # noise bar
                masked_data += [[self.tokenizer.mask_token_ids]]
                target += [[self.tokenizer.sep_token_ids], data[start:end]]
            else:
                # non-noise bar
                masked_data += [data[start:end]]
        masked_data = np.concatenate(masked_data)
        target = np.concatenate(target)
        return InfillingMaskedDataPair(masked_data, target)


class FixedBarMasking(RandomBarMasking):
    need_to_mask_per_data = True

    def __init__(
        self,
        num_past_bars: int,
        num_middle_bars: int,
        num_future_bars: int,
        extra_data_field_name: str = "bar_spans",
        random_crop: bool = False,
    ):
        super().__init__()
        self.num_past_bars = num_past_bars
        self.num_middle_bars = num_middle_bars
        self.num_future_bars = num_future_bars
        self.num_total_bars = num_past_bars + num_middle_bars + num_future_bars
        self.extra_data_field_name = extra_data_field_name
        self.random_crop = random_crop

    def _get_fixed_bar_spans(self, bar_spans: np.ndarray) -> Tuple[int, int, int, int]:
        """Get fixed noise bar spans based on given numbers of past, masking and future bars.
        Args:
            bar_spans: (num_bars, 2) array of (start, end) indices
        Returns:
            past_start, past_end, future_start, future_end: indices of past, masking and future bars
        """
        num_bars = len(bar_spans)
        assert num_bars >= self.num_total_bars, f"num_bars ({num_bars}) < num_total_bars ({self.num_total_bars})"
        start_bar_index = np.random.randint(num_bars - self.num_total_bars + 1) if self.random_crop else 0
        past_start = bar_spans[start_bar_index, 0]
        past_end = bar_spans[start_bar_index + self.num_past_bars - 1, 1]
        future_start = bar_spans[start_bar_index + self.num_past_bars + self.num_middle_bars, 0]
        future_end = bar_spans[start_bar_index + self.num_total_bars - 1, 1]
        return past_start, past_end, future_start, future_end

    def mask_for_infilling(self, data: np.ndarray, **kwargs) -> InfillingMaskedDataPair:
        """Mask data with fixed noise bars based on given numbers of past, masking and future bars.
        Random cropping should be handled here rather than in the data collator.
        Args:
            data: (seq_len, num_features)
            bar_spans: (num_bars, 2) array of (start, end) indices
        Returns:
            masked_data: (masked_seq_len, num_features), with form of {tokens, [MASK], tokens}
            target: (infilling_seq_len, num_features), with form of {<SEP>, tokens}
        """
        bar_spans = kwargs[self.extra_data_field_name]
        past_start, past_end, future_start, future_end = self._get_fixed_bar_spans(bar_spans)

        past = data[past_start:past_end]
        future = data[future_start:future_end]
        middle = data[past_end:future_start]
        masked_data = np.concatenate([past, [self.tokenizer.mask_token_ids], future])
        target = np.concatenate([[self.tokenizer.sep_token_ids], middle])
        return InfillingMaskedDataPair(masked_data, target)


class RandomNgramMasking(InfillingMasking):
    need_to_mask_per_data = True

    def __init__(
        self,
        corruption_rate: float = 0.15,
        extra_data_field_name: str = "ngrams",
        fallback_mean_span_length: int = 3,
        field_specific_masking: bool = False,
        return_ngram_ids: bool = False,
    ):
        """Args:
        corruption_rate: corruption rate of ngram masking.
        extra_data_field_name: name of the extra data field containing ngram spans, `pitch_ngrams` or `rhythm_ngrams`.
        fallback_mean_span_length: mean span length for random span method when ngrams are not available.
        field_specific_masking: whether to use field-specific masking.
            If ngrams are of pitch, only `bar` and `pitch` fields will be masked.
            If ngrams are of rhythm, only `bar`, `position` and `duration` fields will be masked.
        return_ngram_ids: whether to return ngram ids.
        """
        super().__init__()
        self.corruption_rate = corruption_rate
        self.extra_data_field_name = extra_data_field_name

        self.random_span_masking = RandomSpanMasking(corruption_rate, fallback_mean_span_length)

        self.field_specific_masking = field_specific_masking

        self.return_ngram_ids = return_ngram_ids
        self.ngram_type = "pitch" if self.extra_data_field_name == "pitch_ngrams" else "rhythm"

    def setup_tokenizer(self, tokenizer: MIDITokenizer):
        super().setup_tokenizer(tokenizer)
        self.random_span_masking.setup_tokenizer(tokenizer)
        self.bar_field_index = self.tokenizer.field_indices["bar"]
        self.position_field_index = self.tokenizer.field_indices["position"]
        self.duration_field_index = self.tokenizer.field_indices["duration"]
        self.pitch_field_index = self.tokenizer.field_indices["pitch"]

    def _process_ngram_spans(self, ngram_spans: np.ndarray, start: int, end: int) -> np.ndarray:
        """Pick ngram spans within given range and shift them to start from 0."""
        starts = ngram_spans[:, 0]
        ends = ngram_spans[:, 0] + ngram_spans[:, 1]
        ngram_spans = ngram_spans[(starts >= start) & (ends <= end)]

        result = ngram_spans.copy()
        result[:, 0] -= start
        return result

    def _get_random_noise_ngrams(self, num_tokens: int, ngrams: np.ndarray) -> np.ndarray:
        """Get random ngrams.
        Args:
            length: length of sequence
            ngrams: (num_ngrams, 3) array of (start, length, id)
        Returns:
            noise_ngrams: (num_ngrams) bool array, where True means noise ngram.
            has_enough_noise_ngrams: whether there are enough noise ngrams.
        """
        if len(ngrams) == 0:
            return None, False

        num_noise_tokens = int(round(num_tokens * self.corruption_rate))
        num_noise_tokens = min(max(num_noise_tokens, 1), num_tokens - 1)

        # Randomly select noise ngrams
        shuffled_ngrams = ngrams.copy()
        np.random.shuffle(shuffled_ngrams)

        current_noise_tokens = 0
        covered_indices = np.zeros(num_tokens, dtype=bool)
        noise_ngrams = np.zeros(len(ngrams), dtype=bool)
        for i, (start, length, _) in enumerate(shuffled_ngrams):
            if current_noise_tokens >= num_noise_tokens:
                break
            if covered_indices[start : start + length].any():
                continue
            noise_ngrams[i] = True
            covered_indices[start : start + length] = True
            current_noise_tokens += length
        
        has_enough_noise_ngrams = current_noise_tokens >= num_noise_tokens
        return noise_ngrams, has_enough_noise_ngrams

    def mask(self, data: np.ndarray, offset: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Mask data with random n-grams.
        Args:
            data: (seq_len, num_features)
            ngrams: (num_ngrams, 3) array of (start, length, id)
            offset: offset of data in the original sequence, in case random cropping is used
        Returns:
            masked_data: (seq_len, num_features)
            label: (seq_len, num_features)
        """
        ngrams = kwargs[self.extra_data_field_name]
        ngrams = self._process_ngram_spans(ngrams, offset, offset + seq_len)
        noise_ngrams, has_enough_noise_ngrams = self._get_random_noise_ngrams(seq_len, ngrams)

        # If there are not enough ngrams, fallback to random span masking instead.
        if not has_enough_noise_ngrams:
            # TODO: Yet to implement single random span masking.
            return self.random_span_masking.mask(data)
        
        # TODO: Use BERT-style masking.
        noise_ngram_spans = ngrams[noise_ngrams]
        label = data.copy()
        seq_len, _ = data.shape

        mask_indices = np.zeros(seq_len, dtype=bool)
        for start, length, _ in noise_ngram_spans:
            mask_indices[start : start + length] = True

        data[mask_indices] = self.tokenizer.mask_token_ids
        label[~mask_indices] = self.tokenizer.pad_token_ids
        return data, label

    def mask_for_infilling(self, data: np.ndarray, offset: int, **kwargs) -> InfillingMaskedDataPair:
        """Mask data with random n-grams. Put a single mask token for each n-gram.
        Args:
            data: (seq_len, num_features)
            ngrams: (num_ngrams, 3) array of (start, length, id)
            offset: offset of data in the original sequence, in case random cropping is used
        Returns:
            masked_data: (masked_seq_len, num_features), with form of {tokens, [MASK], ...}
            target: (infilling_seq_len, num_features), with form of {<SEP>, tokens, ...}
        """
        seq_len, _ = data.shape
        ngrams = kwargs[self.extra_data_field_name]
        ngrams = self._process_ngram_spans(ngrams, offset, offset + seq_len)
        noise_ngrams, has_enough_noise_ngrams = self._get_random_noise_ngrams(seq_len, ngrams)

        # If there are not enough ngrams, fallback to random span masking instead.
        if not has_enough_noise_ngrams:
            return self.random_span_masking.mask_for_infilling(data)
        
        noise_ngram_spans = ngrams[noise_ngrams]
        # Sort by start index
        noise_ngram_spans = noise_ngram_spans[np.argsort(noise_ngram_spans[:, 0])]

        # TODO: Maybe add specific field masking.

        # Build masked data and target
        masked_data, target = [], []
        for i, (start, length, _) in enumerate(noise_ngram_spans):
            previous_end = noise_ngram_spans[i - 1, 0] + noise_ngram_spans[i - 1, 1] if i > 0 else 0
            masked_data += [data[previous_end:start], [self.tokenizer.mask_token_ids]]
            target += [[self.tokenizer.sep_token_ids], data[start : start + length]]
        # add the last part
        masked_data += [data[previous_end:]]
        masked_data = np.concatenate(masked_data)
        target = np.concatenate(target)

        # Build ngram ids if needed
        if self.return_ngram_ids:
            ngram_ids = np.ones(len(masked_data), dtype=np.int64) * default_padding_index_for_classification
            current_position = 0
            for i, (start, _, ngram_id) in enumerate(noise_ngram_spans):
                previous_end = noise_ngram_spans[i - 1, 0] + noise_ngram_spans[i - 1, 1] if i > 0 else 0
                current_position += start - previous_end
                ngram_ids[current_position] = ngram_id
                current_position += 1

        return (
            InfillingMaskedDataPair(masked_data, target, ngram_ids, self.ngram_type)
            if self.return_ngram_ids
            else InfillingMaskedDataPair(masked_data, target)
        )


class DataCollator:
    def __init__(self, seq_len: Optional[int] = None, random_crop: bool = False):
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

    def __init__(self, seq_len: Optional[int] = None):
        super().__init__(seq_len)

    def __call__(self, batch: List[DatasetItem]) -> DatasetBatch:
        # TODO: Consider right side padding for batch inference?
        data_list = [item.data for item in batch]
        data_list, _ = self.truncate(data_list, self.seq_len)
        data_list = self.pad(data_list, self.seq_len)
        input_ids = np.stack(data_list, axis=0)
        _, seq_len, _ = input_ids.shape
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        padding_mask = torch.tensor(
            [[0] * len(data) + [1] * (seq_len - len(data)) for data in data_list], dtype=torch.bool
        )
        label_ids = None
        attention_mask = None

        return DatasetBatch(input_ids, label_ids, padding_mask, attention_mask)


class DataCollatorForInfilling(DataCollator):
    """Data collator for infilling task, intended for prefix-style inference.
    Support fixed bar masking for now."""

    def __init__(self, masking: FixedBarMasking):
        super().__init__()
        self.masking = masking

    def setup_tokenizer(self, tokenizer: MIDITokenizer):
        super().setup_tokenizer(tokenizer)
        self.masking.setup_tokenizer(tokenizer)

    def __call__(self, batch: List[DatasetItem]) -> DatasetBatch:
        data_list = [item.data for item in batch]
        extra_data_list = [item.extra_data for item in batch]
        # Only collect masked data
        data_list = [
            self.masking.mask_for_infilling(data, **extra_data).masked_data
            for data, extra_data in zip(data_list, extra_data_list)
        ]
        data_list = self.pad(data_list)
        input_ids = np.stack(data_list, axis=0)
        _, seq_len, _ = input_ids.shape
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        padding_mask = torch.tensor(
            [[0] * len(data) + [1] * (seq_len - len(data)) for data in data_list], dtype=torch.bool
        )

        return DatasetBatch(input_ids, label_ids=None, padding_mask=padding_mask)


class DataCollatorForCausalLanguageModeling(DataCollator):
    def __init__(self, seq_len: Optional[int] = None, random_crop: bool = False):
        super().__init__(seq_len, random_crop)

    def __call__(self, batch: List[DatasetItem]) -> DatasetBatch:
        data_list = [item.data for item in batch]

        if self.seq_len is not None:
            data_list, _ = self.truncate(data_list, self.seq_len + 1, random_crop=self.random_crop)

        lengths = [len(data) for data in data_list]
        data_list = self.pad(data_list)
        batched_data = np.stack(data_list, axis=0)
        input_ids = torch.from_numpy(batched_data[:, :-1]).long()
        label_ids = torch.from_numpy(batched_data[:, 1:]).long()

        # causal attention mask
        batch_size, seq_len, _ = input_ids.shape
        attention_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
        padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        for i, length in enumerate(lengths):
            padding_mask[i, length:] = True
        return DatasetBatch(input_ids, label_ids, padding_mask, attention_mask)


class DataCollatorForMaskedLanguageModeling(DataCollator):
    def __init__(self, masking: Masking, seq_len: Optional[int] = None, random_crop: bool = False):
        super().__init__(seq_len, random_crop)
        self.masking = masking

    def setup_tokenizer(self, tokenizer: MIDITokenizer):
        super().setup_tokenizer(tokenizer)
        self.masking.setup_tokenizer(tokenizer)

    def __call__(self, batch: List[DatasetItem]) -> DatasetBatch:
        data_list = [item.data for item in batch]
        extra_data_list = [item.extra_data for item in batch]

        # truncate
        data_list, offsets = self.truncate(data_list, self.seq_len, random_crop=self.random_crop)
        lengths = [len(data) for data in data_list]
        if self.masking.need_to_mask_per_data:
            # mask each data separately, and then pad
            inputs, labels = [], []
            for data, extra_data, offset in zip(data_list, extra_data_list, offsets):
                masked_data, label = self.masking.mask(data, offset=offset, **extra_data)
                inputs.append(masked_data)
                labels.append(label)
            input_ids = torch.from_numpy(np.stack(self.pad(inputs), axis=0)).long()
            label_ids = torch.from_numpy(np.stack(self.pad(labels), axis=0)).long()
        else:
            # pad, and then mask all data together
            batch = torch.from_numpy(np.stack(self.pad(data_list), axis=0)).long()
            input_ids, label_ids = self.masking.mask_batch(batch, lengths)

        # bidirectional attention mask
        batch_size, seq_len, _ = input_ids.shape
        padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        for i, length in enumerate(lengths):
            padding_mask[i, length:] = True
        return DatasetBatch(input_ids, label_ids, padding_mask)


class DataCollatorForPrefixMaskedLanguageModeling(DataCollator):
    def __init__(self, masking: InfillingMasking, seq_len: Optional[int] = None, random_crop: bool = False):
        super().__init__(seq_len, random_crop)
        self.masking = masking

    def setup_tokenizer(self, tokenizer: MIDITokenizer):
        super().setup_tokenizer(tokenizer)
        self.masking.setup_tokenizer(tokenizer)

    def _get_input_and_label(self, masked_data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get input and label for infilling.
        Args:
            masked_data: (masked_seq_len, num_features)
            target: (infilling_seq_len, num_features)
        Returns:
            input: (masked_seq_len + infilling_seq_len, num_features)
            label: (masked_seq_len + infilling_seq_len, num_features)
        """
        masked_seq_len, _ = masked_data.shape
        input = np.concatenate([masked_data, target], axis=0)
        label = np.concatenate(
            [
                # pad tokens for masked data
                np.full((masked_seq_len, len(self.tokenizer.field_names)), self.tokenizer.pad_token_ids),
                # causal modeling for infilling data
                target[1:, :],
                # pad tokens for the last token
                [self.tokenizer.sep_token_ids],
            ],
            axis=0,
        )
        return input, label

    def _get_prefix_attention_mask(self, prefix_lengths: List[int], seq_len: int) -> torch.Tensor:
        attention_masks = []
        for prefix_length in prefix_lengths:
            # bidirectional attention mask for prefix sequence
            left_prefix_part = torch.zeros((seq_len, prefix_length), dtype=torch.bool)

            target_length = seq_len - prefix_length
            top_right_target_part = torch.ones((prefix_length, target_length), dtype=torch.bool)
            # causal attention mask for infilling sequence
            bottom_right_target_part = torch.triu(
                torch.ones((target_length, target_length), dtype=torch.bool), diagonal=1
            )

            right_target_part = torch.cat([top_right_target_part, bottom_right_target_part], dim=0)
            mask = torch.cat([left_prefix_part, right_target_part], dim=1)
            attention_masks.append(mask)
        return torch.stack(attention_masks, dim=0)

    def __call__(self, batch: List[DatasetItem]) -> DatasetBatch:
        data_list = [item.data for item in batch]
        extra_data_list = [item.extra_data for item in batch]

        # truncate
        data_list, offsets = self.truncate(data_list, self.seq_len, random_crop=self.random_crop)
        inputs, labels = [], []
        prefix_lengths, full_lengths = [], []
        extra_labels, ngram_types = [], []
        for data, extra_data, offset in zip(data_list, extra_data_list, offsets):
            # mask each data separately
            data_pair = self.masking.mask_for_infilling(data, offset=offset, **extra_data)
            # construct prefix sequence
            input, label = self._get_input_and_label(data_pair.masked_data, data_pair.target)
            inputs.append(input)
            labels.append(label)
            prefix_lengths.append(len(data_pair.masked_data))
            full_lengths.append(len(input))
            # save extra labels
            extra_labels.append(data_pair.extra_label_ids)
            ngram_types.append(data_pair.ngram_type)
        # pad
        input_ids = torch.from_numpy(np.stack(self.pad(inputs), axis=0)).long()
        label_ids = torch.from_numpy(np.stack(self.pad(labels), axis=0)).long()

        batch_size, seq_len, _ = input_ids.shape
        padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        for i, full_length in enumerate(full_lengths):
            padding_mask[i, full_length:] = True

        # prefix attention mask (batch_size, seq_len, seq_len)
        attention_mask = self._get_prefix_attention_mask(prefix_lengths, seq_len)

        # build extra label ids
        any_extra_label = any(extra_label is not None for extra_label in extra_labels)
        if any_extra_label:
            for i, extra_label in enumerate(extra_labels):
                if extra_label is not None:
                    pad = (
                        np.ones(seq_len - len(extra_label), dtype=extra_label.dtype)
                        * default_padding_index_for_classification
                    )
                    extra_labels[i] = np.concatenate([extra_label, pad], axis=0)
                else:
                    extra_labels[i] = (
                        np.ones(seq_len, dtype=np.int64) * default_padding_index_for_classification
                    )
            extra_label_ids = torch.from_numpy(np.stack(extra_labels, axis=0)).long()
            assert extra_label_ids.shape == (batch_size, seq_len)
            assert len(ngram_types) == batch_size

        return (
            DatasetBatch(input_ids, label_ids, padding_mask, attention_mask, extra_label_ids, ngram_types)
            if any_extra_label
            else DatasetBatch(input_ids, label_ids, padding_mask, attention_mask)
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
        data = file["data"]
        if self.pitch_augumentation:
            self.tokenizer.pitch_shift_augument_(data)
        extra_data = {}
        if self.load_bar_data:
            extra_data["bar_spans"] = file["bar_spans"]
        if self.load_ngram_data:
            extra_data["pitch_ngrams"] = file["pitch_ngrams"]
            extra_data["rhythm_ngrams"] = file["rhythm_ngrams"]
        return DatasetItem(data, extra_data)


class MelodyPretrainDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        data_collator: DataCollator,
        batch_size: int,
        num_workers: int = 0,
        load_bar_data: bool = False,
        load_ngram_data: bool = False,
        pitch_augumentation: bool = False,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        tokenizer_config_path = os.path.join(dataset_dir, "tokenizer_config.json")
        if not os.path.exists(tokenizer_config_path):
            raise ValueError(f"Tokenizer config file not found: {tokenizer_config_path}")
        self.tokenizer = MIDITokenizer.from_config(tokenizer_config_path)

        self.data_collator = data_collator
        self.data_collator.setup_tokenizer(self.tokenizer)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.load_bar_data = load_bar_data
        self.load_ngram_data = load_ngram_data
        self.pitch_augumentation = pitch_augumentation

    def setup(self, stage: str):
        train_dir = os.path.join(self.dataset_dir, "train")
        valid_dir = os.path.join(self.dataset_dir, "valid")
        test_dir = os.path.join(self.dataset_dir, "test")
        self.train_dataset = MelodyDataset(
            train_dir,
            load_bar_data=self.load_bar_data,
            load_ngram_data=self.load_ngram_data,
            pitch_augumentation=self.pitch_augumentation,
        )
        self.train_dataset.setup_tokenizer(self.tokenizer)
        if os.path.exists(valid_dir):
            self.valid_dataset = MelodyDataset(
                valid_dir,
                load_bar_data=self.load_bar_data,
                load_ngram_data=self.load_ngram_data,
                pitch_augumentation=self.pitch_augumentation,
            )
            self.valid_dataset.setup_tokenizer(self.tokenizer)
        if os.path.exists(test_dir):
            self.test_dataset = MelodyDataset(
                test_dir,
                load_bar_data=self.load_bar_data,
                load_ngram_data=self.load_ngram_data,
                pitch_augumentation=self.pitch_augumentation,
            )
            self.test_dataset.setup_tokenizer(self.tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.data_collator,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        # batch_size=1 for prediction currently
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            collate_fn=self.data_collator,
            num_workers=self.num_workers,
            pin_memory=True,
        )
