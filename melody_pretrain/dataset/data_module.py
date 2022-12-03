import os
from glob import glob
from typing import Dict, List, Tuple, NamedTuple, Optional

import numpy as np

# import pytorch_lightning as pl
from torch.utils.data import Dataset
import torch

from .tokenizer import MIDITokenizer


class Masking:
    def __init__(self, tokenizer: MIDITokenizer, need_to_mask_per_data: bool = False):
        self.tokenizer = tokenizer
        self.need_to_mask_per_data = need_to_mask_per_data

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
        candidates = torch.from_numpy(self.tokenizer.special_token_id_matrix)
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

        labels[~mask_indices] = self.tokenizer.pad_token_ids
        inputs[mask_indices] = self.tokenizer.mask_token_ids
        return inputs, labels

    def mask(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def mask_batch(self, inputs: torch.Tensor, lengths: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class InfillingMasking(Masking):
    def __init__(self, tokenizer: MIDITokenizer):
        super().__init__(tokenizer, need_to_mask_per_data=True)

    def mask_for_infilling(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class RandomTokenMasking(Masking):
    def __init__(self, tokenizer: MIDITokenizer, mlm_probability: float = 0.15):
        super().__init__(tokenizer)
        self.mlm_probability = mlm_probability

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

        problability_matrix = torch.full(mask_shape, self.mlm_probability)
        # Only mask the tokens that are not padding
        problability_matrix.masked_fill_(self._get_length_mask(lengths, seq_len), value=0.0)
        mask_indices = torch.bernoulli(torch.full(mask_shape, self.mlm_probability)).bool()
        labels[~mask_indices] = self.tokenizer.pad_token_ids

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(mask_shape, 0.8)).bool() & mask_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_ids

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(mask_shape, 0.5)).bool() & mask_indices & ~indices_replaced
        random_words = torch.stack(
            [torch.randint(size, mask_shape, dtype=inputs.dtype) for size in self.tokenizer.vocab_size]
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class SingleSpanMasking(InfillingMasking):
    def __init__(self, tokenizer: MIDITokenizer, corruption_rate: float = 0.5):
        super().__init__(tokenizer)
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

    def mask_for_infilling(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
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
        return masked_data, target


class RandomSpanMasking(InfillingMasking):
    def __init__(self, tokenizer: MIDITokenizer, corruption_rate: float = 0.15, mean_span_length: int = 3):
        super().__init__(tokenizer)
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
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, length - num_noise_spans)

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

    def mask_for_infilling(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
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
        return masked_data, target


class RandomBarMasking(InfillingMasking):
    def __init__(self, tokenizer: MIDITokenizer, corruption_rate: float = 0.15):
        super().__init__(tokenizer, need_to_mask_per_data=True)
        self.corruption_rate = corruption_rate
        self.bar_field_index = tokenizer.field_names.index("bar")
        self.bar_vocab_size = tokenizer.vocab_sizes[self.bar_field_index]

    def _process_bar_spans(self, bar_spans: np.ndarray, start: int, end: int) -> np.ndarray:
        """Pick bar spans within given range and shift them to start from 0."""
        bar_spans = bar_spans[(bar_spans[:, 0] >= start) & (bar_spans[:, 1] <= end)]
        bar_spans = bar_spans - start
        return bar_spans

    def _get_random_noise_bars(self, num_bars: int) -> np.ndarray:
        """Get random noise bars.
        Args:
            num_bars: number of bars
        Returns:
            noise_bars: (num_bars) bool array, where True means noise bar.
        """
        # Assuming the distribution of notes within bars is uniform,
        # so that the number of noise bars is proportional to the number of noise notes.
        num_noise_bars = int(round(num_bars * self.corruption_rate))
        num_noise_bars = min(max(num_noise_bars, 1), num_bars - 1)
        noise_bars = np.zeros(num_bars, dtype=bool)
        noise_bars[np.random.choice(num_bars, num_noise_bars, replace=False)] = True
        return noise_bars

    def mask(self, data: np.ndarray, bar_spans: np.ndarray, offset: int) -> Tuple[np.ndarray, np.ndarray]:
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

        bar_spans = self._process_bar_spans(bar_spans, offset, offset + seq_len)
        noise_bars = self._get_random_noise_bars(bar_spans)
        noise_bar_spans = bar_spans[noise_bars]
        mask_indices = np.zeros(seq_len, dtype=bool)
        for start, end in noise_bar_spans:
            mask_indices[start:end] = True

        data[mask_indices] = self.tokenizer.mask_token_ids
        label[~mask_indices] = self.tokenizer.pad_token_ids
        return data, label

    def mask_for_infilling(self, data: np.ndarray, bar_spans: np.ndarray, offset: int) -> Tuple[np.ndarray, np.ndarray]:
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
        bar_spans = self._process_bar_spans(bar_spans, offset, offset + seq_len)
        noise_bars = self._get_random_noise_bars(bar_spans)

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
        return masked_data, target


class RandomNgramMasking(InfillingMasking):
    def __init__(self, tokenizer: MIDITokenizer, corruption_rate: float = 0.15):
        super().__init__(tokenizer, need_to_mask_per_data=True)
        self.corruption_rate = corruption_rate

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
        """
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

        if current_noise_tokens < num_noise_tokens:
            print(f"Warning: Not enough ngrams to corrupt, {current_noise_tokens} / {num_noise_tokens}")
        return noise_ngrams

    def mask(self, data: np.ndarray, ngrams: np.ndarray, offset: int) -> Tuple[np.ndarray, np.ndarray]:
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
        label = data.copy()
        seq_len, _ = data.shape

        ngrams = self._process_ngram_spans(ngrams, offset, offset + seq_len)
        noise_ngrams = self._get_random_noise_ngrams(seq_len, ngrams)
        noise_ngram_spans = ngrams[noise_ngrams]
        mask_indices = np.zeros(seq_len, dtype=bool)
        for start, length, _ in noise_ngram_spans:
            mask_indices[start : start + length] = True

        data[mask_indices] = self.tokenizer.mask_token_ids
        label[~mask_indices] = self.tokenizer.pad_token_ids
        return data, label

    def mask_for_infilling(self, data: np.ndarray, ngrams: np.ndarray, offset: int) -> Tuple[np.ndarray, np.ndarray]:
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
        ngrams = self._process_ngram_spans(ngrams, offset, offset + seq_len)
        noise_ngrams = self._get_random_noise_ngrams(seq_len, ngrams)
        noise_ngram_spans = ngrams[noise_ngrams]
        # sort by start index
        noise_ngram_spans = noise_ngram_spans[np.argsort(noise_ngram_spans[:, 0])]

        masked_data, target = [], []
        for i, (start, length, _) in enumerate(noise_ngram_spans):
            previous_end = noise_ngram_spans[i - 1, 0] + noise_ngram_spans[i - 1, 1] if i > 0 else 0
            masked_data += [data[previous_end:start], [self.tokenizer.mask_token_ids]]
            target += [[self.tokenizer.sep_token_ids], data[start : start + length]]
        # add the last part
        masked_data += [data[previous_end:]]

        masked_data = np.concatenate(masked_data)
        target = np.concatenate(target)
        return masked_data, target


class DataCollator:
    def __init__(self, tokenizer: MIDITokenizer, seq_len: int, random_crop: bool = False):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.random_crop = random_crop

    def truncate(
        self, batch: List[np.ndarray], max_length: int, random_crop: bool = False
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Truncate batch to given maximum length.
        Args:
            batch: list of (seq_len, num_features)
            max_length: maximum length of the batch
            random_crop: whether to crop the batch randomly
        Returns:
            batch: list of (max_length, num_features)
            offsets: list of offsets
        """
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


class DatasetItem(NamedTuple):
    data: np.ndarray
    extra_data: Dict[str, np.ndarray]


class DatasetBatch(NamedTuple):
    input_ids: torch.Tensor
    label_ids: torch.Tensor
    padding_mask: torch.Tensor
    attention_mask: Optional[torch.Tensor]

    """Note that in PyTorch's padding mask and attention mask, True means to ignore."""


class DataCollatorForCausalLanguageModeling(DataCollator):
    def __init__(self, tokenizer: MIDITokenizer, seq_len: int, random_crop: bool = False):
        super().__init__(tokenizer, seq_len, random_crop)

    def __call__(self, batch: List[DatasetItem]) -> Tuple[torch.Tensor, torch.Tensor]:
        data_list = [item.data for item in batch]

        data_list, lengths = self.truncate(data_list, self.seq_len + 1, random_crop=self.random_crop)
        data_list = self.pad(data_list)
        batched_data = np.stack(data_list, axis=0)
        input_ids = torch.from_numpy(batched_data[:, :-1])
        label_ids = torch.from_numpy(batched_data[:, 1:])

        # causal attention mask
        batch_size, seq_len = input_ids.shape
        attention_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
        padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        for i, length in enumerate(lengths):
            padding_mask[i, length:] = True
        return DatasetBatch(input_ids, label_ids, padding_mask, attention_mask)


class DataCollatorForMaskedLanguageModeling(DataCollator):
    def __init__(self, tokenizer: MIDITokenizer, masking: Masking, seq_len: int, random_crop: bool = False):
        super().__init__(tokenizer, seq_len, random_crop)
        self.masking = masking

    def __call__(self, batch: List[DatasetItem]) -> Tuple[torch.Tensor, torch.Tensor]:
        data_list = [item.data for item in batch]
        extra_data_list = [item.extra_data for item in batch]

        # truncate
        data_list, offsets = self.truncate(data_list, self.seq_len, random_crop=self.random_crop)
        if self.masking.need_to_mask_per_data:
            # mask each data separately, and then pad
            inputs, labels = [], []
            for data, extra_data, offset in zip(data_list, extra_data_list, offsets):
                masked_data, label = self.masking.mask(data, offset=offset, **extra_data)
                inputs.append(masked_data)
                labels.append(label)
            input_ids = torch.from_numpy(np.stack(self.pad(inputs), axis=0))
            label_ids = torch.from_numpy(np.stack(self.pad(labels), axis=0))
        else:
            # pad, and then mask all data together
            lengths = [len(data) for data in batch]
            batch = torch.from_numpy(np.stack(self.pad(data_list), axis=0))
            input_ids, label_ids = self.masking.mask_batch(batch, lengths)

        # bidirectional attention mask
        batch_size, seq_len = input_ids.shape
        attention_mask = None
        padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        for i, length in enumerate(lengths):
            padding_mask[i, length:] = True
        return DatasetBatch(input_ids, label_ids, padding_mask, attention_mask)


class DataCollatorForPrefixMaskedLanguageModeling(DataCollator):
    def __init__(self, tokenizer: MIDITokenizer, masking: InfillingMasking, seq_len: int, random_crop: bool = False):
        super().__init__(tokenizer, seq_len, random_crop)
        self.masking = masking

    def get_input_and_label(self, masked_data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
                np.full((masked_seq_len, self.tokenizer.num_features), self.tokenizer.pad_token_ids),
                # causal modeling for infilling data
                target[1:, :],
                # pad tokens for the last token
                [self.tokenizer.sep_token_ids],
            ],
            axis=0,
        )
        return input, label

    def __call__(self, batch: List[DatasetItem]) -> Tuple[torch.Tensor, torch.Tensor]:
        data_list = [item.data for item in batch]
        extra_data_list = [item.extra_data for item in batch]

        # truncate
        data_list, offsets = self.truncate(data_list, self.seq_len, random_crop=self.random_crop)
        inputs, labels = [], []
        prefix_lengths, full_lengths = [], []
        for data, extra_data, offset in zip(data_list, extra_data_list, offsets):
            # mask each data separately
            masked_data, target = self.masking.mask_for_infilling(data, offset=offset, **extra_data)
            # construct prefix sequence
            input, label = self.get_inputs_and_labels(masked_data, target)
            inputs.append(input)
            labels.append(label)
            prefix_lengths.append(len(masked_data))
            full_lengths.append(len(input))
        # pad
        input_ids = torch.from_numpy(np.stack(self.pad(inputs), axis=0))
        label_ids = torch.from_numpy(np.stack(self.pad(labels), axis=0))

        # prefix attention mask
        batch_size, seq_len = input_ids.shape
        attention_mask = []
        padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        for i, (prefix_length, full_length) in enumerate(zip(prefix_lengths, full_lengths)):
            # bidirectional attention mask for prefix sequence
            
            padding_mask[i, full_length:] = True

        return DatasetBatch(input_ids, label_ids, padding_mask, attention_mask)


class MelodyDataset(Dataset):
    def __init__(self, data_dir: str, seq_len: int = None):
        self.files = glob(os.path.join(data_dir, "*.npy"))
        self.data = [np.load(f) for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.data[idx]
