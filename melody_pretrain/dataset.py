import os
from glob import glob
from typing import Dict, List, NamedTuple, Optional, Tuple

import lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .tokenizer import MIDITokenizer


ngram_ids_ignore_index = -100
span_indices_padding_index = 0


class DatasetItem(NamedTuple):
    data: np.ndarray
    extra_data: Dict[str, np.ndarray]


class InfillingData(NamedTuple):
    """Used for text infilling, the data is a pair of (masked data, target).
    Allow extra data to be attached to the data pair.
    """

    sources: List[np.ndarray]
    targets: List[np.ndarray]
    target_span_indices: List[int]

    ngram_type: Optional[str] = None
    ngram_ids: Optional[List[int]] = None

    field_padding_indices: Optional[List[int]] = None


class DatasetBatch(NamedTuple):
    """A batch of data for training. Allow extra data to be attached to the batch.
    Note that in PyTorch's padding mask and attention mask, True means to ignore."""

    input_ids: torch.Tensor
    label_ids: torch.Tensor
    padding_mask: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None

    ngram_types: Optional[List[str]] = None
    ngram_ids: Optional[torch.Tensor] = None

    span_indices: Optional[torch.Tensor] = None


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
    def mask_for_infilling(self, data: np.ndarray, **kwargs) -> InfillingData:
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

    def mask_for_infilling(self, data: np.ndarray, **kwargs) -> InfillingData:
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

    def mask_for_infilling(self, data: np.ndarray, **kwargs) -> InfillingData:
        """Mask data with single span for infilling. Put a single mask token for the span.
        Args:
            data: (seq_len, num_features)
        """
        seq_len, _ = data.shape
        start, end = self._get_random_span(seq_len)
        sources = [data[:start], data[end:]]
        targets = [data[start:end]]
        # target span index is always 1, since the sequence is {source, target, source}
        target_span_indices = [1]
        return InfillingData(sources, targets, target_span_indices)


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

    def mask_for_infilling(self, data: np.ndarray, **kwargs) -> InfillingData:
        """Mask data with random spans for infilling. Put a single mask token for each span. (T5-style)
        Args:
            data: (seq_len, num_features)
        """
        seq_len, _ = data.shape
        spans = self._get_random_spans(seq_len)
        # Collect masked data and target data. Maybe there is a better way to do this.
        sources, targets = [], []
        target_span_indices = []
        for i, (start, end) in enumerate(spans):
            if i % 2 == 0:
                # non-noise span
                sources.append(data[start:end])
            else:
                # noise span
                targets.append(data[start:end])
                target_span_indices.append(i)
        return InfillingData(sources, targets, target_span_indices)


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

    def mask_for_infilling(self, data: np.ndarray, offset: int, **kwargs) -> InfillingData:
        """Mask data with random bars for infilling. Put a single mask token for each bar.
        Args:
            data: (seq_len, num_features)
            bar_spans: (num_bars, 2) array of (start, end) indices
            offset: offset of data in the original sequence, in case random cropping is used
        """
        seq_len, _ = data.shape
        bar_spans = kwargs[self.extra_data_field_name]
        bar_spans = self._process_bar_spans(bar_spans, offset, offset + seq_len)
        noise_bars = self._get_random_noise_bars(bar_spans, seq_len)

        sources, targets = [], []
        target_span_indices = []
        for i, (start, end) in enumerate(bar_spans):
            if noise_bars[i]:
                # noise bar
                targets.append(data[start:end])
                target_span_indices.append(i)
            else:
                # non-noise bar
                sources.append(data[start:end])
        return InfillingData(sources, targets, target_span_indices)


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

    def mask_for_infilling(self, data: np.ndarray, **kwargs) -> InfillingData:
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
        sources = [past, future]
        targets = [middle]
        # target span index is always 1, since the sequence is {past, middle, future}
        target_span_indices = [1]
        return InfillingData(sources, targets, target_span_indices)


class RandomNgramMasking(InfillingMasking):
    need_to_mask_per_data = True

    def __init__(
        self,
        corruption_rate: float = 0.15,
        extra_data_field_name: str = "ngrams",
        fallback_mean_span_length: int = 3,
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

        self.ngram_type = "pitch" if self.extra_data_field_name == "pitch_ngrams" else "rhythm"

    def setup_tokenizer(self, tokenizer: MIDITokenizer):
        super().setup_tokenizer(tokenizer)
        self.random_span_masking.setup_tokenizer(tokenizer)
        position_field_index = self.tokenizer.field_indices["position"]
        duration_field_index = self.tokenizer.field_indices["duration"]
        pitch_field_index = self.tokenizer.field_indices["pitch"]

        # bar, pitch
        self.pitch_ngram_padding_indices = [position_field_index, duration_field_index]
        # bar, position, duration
        self.rhythm_ngram_padding_indices = [pitch_field_index]

    def _process_ngram_spans(self, ngram_spans: np.ndarray, start: int, end: int) -> np.ndarray:
        """Pick ngram spans within given range and shift them to start from 0."""
        starts = ngram_spans[:, 0]
        ends = ngram_spans[:, 0] + ngram_spans[:, 1]
        ngram_spans = ngram_spans[(starts >= start) & (ends <= end)]
        ngram_spans[:, 0] -= start
        return ngram_spans

    def _get_random_noise_ngrams(self, num_tokens: int, ngrams: np.ndarray) -> np.ndarray:
        """Get random ngrams.
        Args:
            length: length of sequence
            ngrams: (num_ngrams, 3) array of (start, length, id)
        Returns:
            noise_ngram_indices: list of int, indices of noise ngrams
            has_enough_noise_ngrams: whether there are enough noise ngrams.
        """
        if len(ngrams) == 0:
            return None, False

        num_noise_tokens = int(round(num_tokens * self.corruption_rate))
        num_noise_tokens = min(max(num_noise_tokens, 1), num_tokens - 1)

        # Randomly select noise ngrams
        permutation = np.random.permutation(len(ngrams))

        current_noise_tokens = 0
        covered_indices = np.zeros(num_tokens, dtype=bool)
        noise_ngram_indices = []
        for index in permutation:
            start, length, _ = ngrams[index]
            if current_noise_tokens >= num_noise_tokens:
                break
            if np.any(covered_indices[start : start + length]):
                continue
            noise_ngram_indices.append(index)
            covered_indices[start : start + length] = True
            current_noise_tokens += length

        has_enough_noise_ngrams = current_noise_tokens >= num_noise_tokens
        return noise_ngram_indices, has_enough_noise_ngrams

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
        # TODO: Use BERT-style masking.
        seq_len, _ = data.shape
        label = data.copy()

        ngrams = kwargs[self.extra_data_field_name]
        ngrams = self._process_ngram_spans(ngrams, offset, offset + seq_len)
        noise_ngram_indices, has_enough_noise_ngrams = self._get_random_noise_ngrams(seq_len, ngrams)

        # If there are not enough ngrams, fallback to random span masking instead.
        if not has_enough_noise_ngrams:
            # TODO: Yet to implement single random span masking.
            return self.random_span_masking.mask(data)
        noise_ngram_spans = ngrams[noise_ngram_indices]

        mask_indices = np.zeros(seq_len, dtype=bool)
        for start, length, _ in noise_ngram_spans:
            mask_indices[start : start + length] = True

        data[mask_indices] = self.tokenizer.mask_token_ids
        label[~mask_indices] = self.tokenizer.pad_token_ids
        return data, label

    def mask_for_infilling(self, data: np.ndarray, offset: int, **kwargs) -> InfillingData:
        """Mask data with random n-grams. Put a single mask token for each n-gram.
        Args:
            data: (seq_len, num_features)
            ngrams: (num_ngrams, 3) array of (start, length, id)
            offset: offset of data in the original sequence, in case random cropping is used
        """
        seq_len, _ = data.shape
        ngrams = kwargs[self.extra_data_field_name]
        ngrams = self._process_ngram_spans(ngrams, offset, offset + seq_len)
        noise_ngram_indices, has_enough_noise_ngrams = self._get_random_noise_ngrams(seq_len, ngrams)

        # If there are not enough ngrams, fallback to random span masking instead.
        if not has_enough_noise_ngrams:
            return self.random_span_masking.mask_for_infilling(data)

        # Sort by start index
        noise_ngram_spans = ngrams[noise_ngram_indices]
        noise_ngram_spans = noise_ngram_spans[np.argsort(noise_ngram_spans[:, 0])]

        # Build masked data and target
        sources, targets = [], []
        target_span_indices = []
        ngram_ids = []
        num_span = 0
        for i, (start, length, ngram_id) in enumerate(noise_ngram_spans):
            previous_end = noise_ngram_spans[i - 1, 0] + noise_ngram_spans[i - 1, 1] if i > 0 else 0
            if start > previous_end:
                sources.append(data[previous_end:start])
                targets.append(data[start : start + length])
                target_span_indices.append(num_span + 1)
                ngram_ids.append(ngram_id)
                num_span += 2
            elif start == previous_end:
                # this ngram is adjacent to the previous one
                targets.append(data[start : start + length])
                target_span_indices.append(num_span)
                ngram_ids.append(ngram_id)
                num_span += 1
            else:
                raise RuntimeError("ngrams are overlapping")

        # field padding indices
        field_padding_indices = (
            self.pitch_ngram_padding_indices if self.ngram_type == "pitch" else self.rhythm_ngram_padding_indices
        )

        return InfillingData(
            sources,
            targets,
            target_span_indices,
            ngram_type=self.ngram_type,
            ngram_ids=ngram_ids,
            field_padding_indices=field_padding_indices,
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
        input_ids = np.stack(self.pad(data_list, self.seq_len), axis=0)
        input_ids = torch.from_numpy(input_ids).long()
        padding_mask = torch.tensor(
            [[0] * len(data) + [1] * (self.seq_len - len(data)) for data in data_list], dtype=torch.bool
        )

        return DatasetBatch(input_ids, label_ids=None, padding_mask=padding_mask)


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
        masked_data_list = []
        for data, extra_data in zip(data_list, extra_data_list):
            sources = self.masking.mask_for_infilling(data, **extra_data).sources
            assert len(sources) == 2, "Fixed bar masking should generate past and future spans."
            masked_data = np.concatenate([sources[0], [self.tokenizer.mask_token_ids], sources[1]], axis=0)
            masked_data_list.append(masked_data)
        input_ids = np.stack(self.pad(masked_data_list), axis=0)
        input_ids = torch.from_numpy(input_ids).long()
        _, seq_len, _ = input_ids.shape
        padding_mask = torch.tensor(
            [[0] * len(data) + [1] * (seq_len - len(masked_data)) for masked_data in masked_data_list], dtype=torch.bool
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
    def __init__(
        self,
        masking: InfillingMasking,
        seq_len: Optional[int] = None,
        random_crop: bool = False,
        permutated_infilling: bool = False,
        span_independent_infilling: bool = False,
        ngram_classification: bool = False,
        ngram_field_specific_masking: bool = False,
    ):
        super().__init__(seq_len, random_crop)
        self.masking = masking
        self.permutated_infilling = permutated_infilling
        self.span_independent_infilling = span_independent_infilling
        self.ngram_classification = ngram_classification
        self.ngram_field_specific_masking = ngram_field_specific_masking

    def setup_tokenizer(self, tokenizer: MIDITokenizer):
        super().setup_tokenizer(tokenizer)
        self.masking.setup_tokenizer(tokenizer)

    def _get_source_and_target(
        self,
        sources: List[np.ndarray],
        targets: List[np.ndarray],
        target_span_indices: List[int],
        permutated: bool,
    ) -> Tuple[np.ndarray, np.ndarray, List[int], List[int], List[int]]:
        assert len(target_span_indices) == len(targets)
        assert target_span_indices == sorted(target_span_indices)

        source_list, target_list = [], []
        current_source, current_target = 0, 0
        mask_positions, sep_positions, target_span_lengths = [], [], []
        current_source_position = 0
        for span_index in range(len(sources) + len(targets)):
            if current_target < len(targets) and span_index == target_span_indices[current_target]:
                source_list.append([self.tokenizer.mask_token_ids])
                target_list.append(np.concatenate(([self.tokenizer.sep_token_ids], targets[current_target]), axis=0))
                mask_positions.append(current_source_position)
                current_source_position += 1
                current_target += 1
            else:
                source_list.append(sources[current_source])
                current_source_position += len(sources[current_source])
                current_source += 1
        assert current_source == len(sources) and current_target == len(targets)

        if permutated and len(targets) > 1:
            permutation = np.random.permutation(len(targets))
            target_list = [target_list[i] for i in permutation]

        source = np.concatenate(source_list, axis=0)
        target = np.concatenate(target_list, axis=0)
        target_span_lengths = [len(target) for target in target_list]
        sep_positions = [0] + np.cumsum(target_span_lengths)[:-1].tolist()

        if permutated and len(targets) > 1:
            # reorder sep positions back to original, so that sep postitions are corresponding to mask positions
            sep_positions = [sep_positions[i] for i in np.argsort(permutation)]

        return source, target, mask_positions, sep_positions, target_span_lengths

    def _get_input_and_label(
        self,
        source: np.ndarray,
        target: np.ndarray,
        sep_positions: List[int],
        field_padding_indices: Optional[List[int]],
    ) -> Tuple[np.ndarray, np.ndarray]:

        if self.ngram_field_specific_masking and field_padding_indices is not None:
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

    def _get_attention_mask(
        self, source_length: int, seq_len: int, target_span_lengths: Optional[List[int]] = None
    ) -> torch.Tensor:
        # bidirectional attention mask for prefix sequence
        left_prefix_part = torch.zeros((seq_len, source_length), dtype=torch.bool)

        target_length = seq_len - source_length
        top_right_target_part = torch.ones((source_length, target_length), dtype=torch.bool)

        if target_span_lengths is not None:
            # independent causal attention mask for each infilling sequence
            assert (
                sum(target_span_lengths) <= target_length
            ), "sum of target span lengths must be less than target length"
            bottom_right_target_part = torch.ones((target_length, target_length), dtype=torch.bool)
            span_start = 0
            for span_length in target_span_lengths:
                span_end = span_start + span_length
                bottom_right_target_part[span_start:span_end, span_start:span_end] = torch.triu(
                    torch.ones((span_length, span_length), dtype=torch.bool), diagonal=1
                )
                span_start = span_end
        else:
            # causal attention mask for infilling sequence
            bottom_right_target_part = torch.triu(
                torch.ones((target_length, target_length), dtype=torch.bool), diagonal=1
            )

        right_target_part = torch.cat([top_right_target_part, bottom_right_target_part], dim=0)
        return torch.cat([left_prefix_part, right_target_part], dim=1)

    def _get_ngram_ids(self, mask_positions: List[int], ngram_id: Optional[List[int]], seq_len: int) -> torch.Tensor:
        ngram_ids = np.ones(seq_len, dtype=np.int64) * ngram_ids_ignore_index
        if ngram_id is not None:
            assert len(mask_positions) == len(ngram_id), "mask_positions and ngram_id should have the same length"
            ngram_ids[mask_positions] = ngram_id
        return torch.from_numpy(ngram_ids).long()

    def _get_span_indices(
        self, mask_positions: List[int], sep_positions: List[int], source_length: int, seq_len: int
    ) -> torch.Tensor:
        span_indices = np.ones(seq_len, dtype=np.int64) * span_indices_padding_index
        assert len(mask_positions) == len(sep_positions), "mask_positions and sep_positions should have the same length"
        for i, (mask_position, sep_position) in enumerate(zip(mask_positions, sep_positions)):
            span_indices[mask_position] = i + 1
            span_indices[sep_position + source_length] = i + 1
        return torch.from_numpy(span_indices).long()

    def __call__(self, batch: List[DatasetItem]) -> DatasetBatch:
        data_list = [item.data for item in batch]
        extra_data_list = [item.extra_data for item in batch]

        # truncate
        data_list, offsets = self.truncate(data_list, self.seq_len, random_crop=self.random_crop)

        # collect all the data, and genereate the inputs and labels
        inputs, labels = [], []
        source_lengths, input_lengths = [], []
        ngram_ids_list, ngram_types = [], []
        mask_positions_list, sep_positions_list, target_span_lengths_list = [], [], []
        for data, extra_data, offset in zip(data_list, extra_data_list, offsets):
            infilling_data = self.masking.mask_for_infilling(data, offset=offset, **extra_data)

            source, target, mask_positions, sep_positions, target_span_lengths = self._get_source_and_target(
                infilling_data.sources,
                infilling_data.targets,
                target_span_indices=infilling_data.target_span_indices,
                permutated=self.permutated_infilling,
            )
            input, label = self._get_input_and_label(
                source,
                target,
                sep_positions=sep_positions,
                field_padding_indices=infilling_data.field_padding_indices,
            )

            inputs.append(input)
            labels.append(label)
            source_lengths.append(len(source))
            input_lengths.append(len(input))
            mask_positions_list.append(mask_positions)
            sep_positions_list.append(sep_positions)
            target_span_lengths_list.append(target_span_lengths)

            ngram_ids_list.append(infilling_data.ngram_ids)
            ngram_types.append(infilling_data.ngram_type)

        # pad
        input_ids = torch.from_numpy(np.stack(self.pad(inputs), axis=0)).long()
        label_ids = torch.from_numpy(np.stack(self.pad(labels), axis=0)).long()

        # padding mask (batch_size, seq_len)
        batch_size, seq_len, _ = input_ids.shape
        padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        for i, input_length in enumerate(input_lengths):
            padding_mask[i, input_length:] = True

        # attention mask (batch_size, seq_len, seq_len)
        attention_mask = torch.stack(
            [
                self._get_attention_mask(
                    source_length,
                    seq_len,
                    target_span_lengths=target_span_lengths if self.span_independent_infilling else None,
                )
                for source_length, target_span_lengths in zip(source_lengths, target_span_lengths_list)
            ],
            dim=0,
        )

        # ngram ids and types
        if self.ngram_classification:
            ngram_ids = torch.stack(
                [
                    self._get_ngram_ids(mask_positions, ngram_id_, seq_len)
                    for mask_positions, ngram_id_ in zip(mask_positions_list, ngram_ids_list)
                ],
                dim=0,
            )
        else:
            ngram_types = None
            ngram_ids = None

        # span indices for advanced infilling
        if self.span_independent_infilling or self.permutated_infilling:
            span_indices = torch.stack(
                [
                    self._get_span_indices(mask_positions, sep_positions, source_length, seq_len)
                    for mask_positions, sep_positions, source_length in zip(
                        mask_positions_list, sep_positions_list, source_lengths
                    )
                ],
                dim=0,
            )
        else:
            span_indices = None

        return DatasetBatch(
            input_ids,
            label_ids,
            padding_mask,
            attention_mask,
            ngram_types=ngram_types,
            ngram_ids=ngram_ids,
            span_indices=span_indices,
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
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.valid_dir = os.path.join(self.dataset_dir, "valid")
        self.test_dir = os.path.join(self.dataset_dir, "test")
        self.train_dataset = MelodyDataset(
            self.train_dir,
            load_bar_data=self.load_bar_data,
            load_ngram_data=self.load_ngram_data,
            pitch_augumentation=self.pitch_augumentation,
        )
        self.train_dataset.setup_tokenizer(self.tokenizer)
        if os.path.exists(self.valid_dir):
            self.valid_dataset = MelodyDataset(
                self.valid_dir,
                load_bar_data=self.load_bar_data,
                load_ngram_data=self.load_ngram_data,
                pitch_augumentation=self.pitch_augumentation,
            )
            self.valid_dataset.setup_tokenizer(self.tokenizer)
        if os.path.exists(self.test_dir):
            self.test_dataset = MelodyDataset(
                self.test_dir,
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
        if not os.path.exists(self.valid_dir):
            return None
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        if not os.path.exists(self.test_dir):
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        if not os.path.exists(self.test_dir):
            return None
        # batch_size=1 for prediction currently
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            collate_fn=self.data_collator,
            num_workers=self.num_workers,
            pin_memory=True,
        )
