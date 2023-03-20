import os
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import (
    DataBatch,
    DataCollator,
    DataCollatorForCausalLanguageModeling,
    DataCollatorForMaskedLanguageModeling,
    DataCollatorForPaddingOnly,
    DataCollatorForInfilling,
    DataCollatorForRecovery,
    FixedBarMasking,
    MultiTargetInfillingMasking,
    RandomBarMasking,
    RandomNgramMasking,
    RandomSkeletonUnitMasking,
    RandomSpanMasking,
    SingleSpanMasking,
    ngram_ids_ignore_index,
)
from .module import CompoundTokenFuser
from .ngram import get_lexicon_size
from .utils import gumbel_sample


class TrainingTask:
    def __init__(self, task_name: str, weight: float = 1.0):
        self.task_name = task_name
        self.weight = weight

    def get_data_collator(self) -> DataCollator:
        raise NotImplementedError

    def register_extra_modules(self, model) -> None:
        pass

    def __call__(self, model, batch: DataBatch, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class LanguageModelingTask(TrainingTask):
    def __init__(
        self,
        task_name: str = "clm",
        weight: float = 1.0,
        seq_len: int = 256,
        random_crop: bool = True,
        padding_only: bool = False,
    ):
        super().__init__(task_name, weight)
        self.seq_len = seq_len
        self.random_crop = random_crop
        self.padding_only = padding_only

    def get_data_collator(self) -> DataCollator:
        if self.padding_only:
            return DataCollatorForPaddingOnly(seq_len=self.seq_len)
        return DataCollatorForCausalLanguageModeling(seq_len=self.seq_len, random_crop=self.random_crop)

    def __call__(self, model, batch: DataBatch, **kwargs) -> torch.Tensor:
        logits = model(batch)
        return model._get_loss(logits, batch.label_ids)


class InfillingTask(TrainingTask):
    def __init__(
        self,
        task_name: str = "infilling",
        kind: Union[str, List[str]] = "span",
        weight: float = 1.0,
        probabilities: Optional[List[float]] = None,
        corruption_rate: float = 0.15,
        mean_span_length: int = 4,
        seq_len: int = 256,
        random_crop: bool = True,
        permutated_infilling: bool = False,
        span_independent_infilling: bool = False,
        field_specific_masking: bool = False,
    ):
        super().__init__(f"{kind}_{task_name}", weight)
        self.kinds = kind if isinstance(kind, list) else [kind]
        self.probabilities = probabilities
        self.corruption_rate = corruption_rate
        self.mean_span_length = mean_span_length
        self.seq_len = seq_len
        self.random_crop = random_crop
        self.permutated_infilling = permutated_infilling
        self.span_independent_infilling = span_independent_infilling
        self.field_specific_masking = field_specific_masking

    def get_data_collator(self) -> DataCollator:
        masking = get_masking(
            kinds=self.kinds,
            corruption_rate=self.corruption_rate,
            mean_span_length=self.mean_span_length,
            random_crop=self.random_crop,
            probabilities=self.probabilities,
            field_specific_masking=self.field_specific_masking,
        )
        return DataCollatorForInfilling(
            masking=masking,
            seq_len=self.seq_len,
            random_crop=self.random_crop,
            permutated_infilling=self.permutated_infilling,
            span_independent_infilling=self.span_independent_infilling,
        )

    def __call__(self, model, batch: DataBatch, **kwargs) -> torch.Tensor:
        logits = model(batch)
        return model._get_loss(logits, batch.label_ids)


class NgramClassificationTask(TrainingTask):
    def __init__(self, task_name: str = "ngram_classification", weight: float = 1.0):
        super().__init__(task_name, weight)

    def register_extra_modules(self, model) -> None:
        lexicon_path = os.path.join(model.dataset_dir, "ngram_data", "lexicon.pkl")
        pitch_size, rhythm_size = get_lexicon_size(lexicon_path)
        model.pitch_ngram_head = nn.Linear(model.model_dim, pitch_size)
        model.rhythm_ngram_head = nn.Linear(model.model_dim, rhythm_size)

    def __call__(self, model, batch: DataBatch, model_outputs: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        batch_size, seq_len, _ = model_outputs.shape
        assert len(batch.ngram_types) == batch_size, "ngram_types must be a list of length batch_size"
        assert batch.ngram_ids.shape == (batch_size, seq_len), "label_ids must be of shape (batch_size, seq_len)"
        pitch_logits = model.pitch_ngram_head(model_outputs)
        rhythm_logits = model.rhythm_ngram_head(model_outputs)

        pitch_indices = [i for i, ngram_type in enumerate(batch.ngram_types) if ngram_type == "pitch"]
        rhythm_indices = [i for i, ngram_type in enumerate(batch.ngram_types) if ngram_type == "rhythm"]
        # extract corresponding ngram types
        pitch_logits = pitch_logits[pitch_indices]
        pitch_label_ids = batch.ngram_ids[pitch_indices]
        rhythm_logits = rhythm_logits[rhythm_indices]
        rhythm_label_ids = batch.ngram_ids[rhythm_indices]

        # (batch_size, seq_len, vocab_size) -> (batch_size, vocab_size, seq_len)
        # use reduction="sum" to avoid averaging over the batch
        pitch_loss = F.cross_entropy(
            pitch_logits.transpose(1, 2), pitch_label_ids, reduction="sum", ignore_index=ngram_ids_ignore_index
        )
        rhythm_loss = F.cross_entropy(
            rhythm_logits.transpose(1, 2), rhythm_label_ids, reduction="sum", ignore_index=ngram_ids_ignore_index
        )
        # average over the number of non-padding tokens
        count = torch.count_nonzero(pitch_label_ids != ngram_ids_ignore_index) + torch.count_nonzero(
            rhythm_label_ids != ngram_ids_ignore_index
        )
        return (pitch_loss + rhythm_loss) / count


class RewritingTask(TrainingTask):
    def __init__(
        self,
        task_name: str = "rewriting",
        kind: Union[str, List[str]] = "span",
        weight: float = 1.0,
        probabilities: Optional[List[float]] = None,
        corruption_rate: float = 0.15,
        mean_span_length: int = 4,
        seq_len: int = 256,
        random_crop: bool = True,
        generator_size_factor: int = 2,
        sampling_temperature: float = 1.0,
    ):
        super().__init__(f"{kind}_{task_name}", weight)
        self.kinds = kind if isinstance(kind, list) else [kind]
        self.probabilities = probabilities
        self.corruption_rate = corruption_rate
        self.mean_span_length = mean_span_length
        self.seq_len = seq_len
        self.whole_seq_len = 2 * seq_len
        self.random_crop = random_crop
        self.generator_size_factor = generator_size_factor
        self.sampling_temperature = sampling_temperature

    def get_data_collator(self) -> DataCollator:
        masking = get_masking(
            kinds=self.kinds,
            corruption_rate=self.corruption_rate,
            mean_span_length=self.mean_span_length,
            random_crop=self.random_crop,
            probabilities=self.probabilities,
        )
        return DataCollatorForMaskedLanguageModeling(
            masking=masking,
            seq_len=self.seq_len,
            random_crop=self.random_crop,
        )

    def register_extra_modules(self, model) -> None:
        fake_model_dim = model.model_dim // self.generator_size_factor

        model.fake_fuser = CompoundTokenFuser(model.tokenizer, model.embedding_dim, fake_model_dim)
        model.fake_transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=fake_model_dim,
                nhead=model.num_heads,
                dim_feedforward=model.feedforward_dim // self.generator_size_factor,
                dropout=model.dropout,
                activation=F.gelu,
                batch_first=True,
            ),
            num_layers=model.num_layers,
        )
        # tie embeddings
        model.fake_fuser.embeddings = model.fuser.embeddings

        self.mask_token_tensor = torch.tensor(model.tokenizer.mask_token_ids, dtype=torch.long)
        self.sep_token_tensor = torch.tensor(model.tokenizer.sep_token_ids, dtype=torch.long)
        self.pad_token_tensor = torch.tensor(model.tokenizer.pad_token_ids, dtype=torch.long)
        self.sep_token_tensor.unsqueeze_(0)
        self.pad_token_tensor.unsqueeze_(0)

    def _move_tensors(self, device: torch.device):
        if self.mask_token_tensor.device != device:
            self.mask_token_tensor = self.mask_token_tensor.to(device)
            self.sep_token_tensor = self.sep_token_tensor.to(device)
            self.pad_token_tensor = self.pad_token_tensor.to(device)

    def _get_attention_mask(self, source_length: int, seq_len: int, device: torch.device) -> torch.Tensor:
        target_length = seq_len - source_length
        left_prefix_part = torch.zeros((seq_len, source_length), dtype=torch.bool, device=device)
        top_right_target_part = torch.ones((source_length, target_length), dtype=torch.bool, device=device)
        bottom_right_target_part = torch.triu(
            torch.ones((target_length, target_length), dtype=torch.bool, device=device), diagonal=1
        )
        right_target_part = torch.cat([top_right_target_part, bottom_right_target_part], dim=0)
        return torch.cat([left_prefix_part, right_target_part], dim=1)

    def __call__(self, model, batch: DataBatch, **kwargs) -> torch.Tensor:
        batch_size, _, num_features = batch.input_ids.shape
        self._move_tensors(model.device)

        # 1. Feed masked data to the small model for MLM task
        x = model.fake_fuser(batch.input_ids)
        x = model.fake_transformer_encoder(x, src_key_padding_mask=batch.padding_mask)
        x = model.fake_fuser.decode(x)
        fake_model_loss = model._get_loss(x, batch.label_ids)

        # 2. Sample fake tokens
        fake_input_ids = batch.input_ids.clone()
        replaced_token_mask = (fake_input_ids == self.mask_token_tensor).all(dim=-1)

        for i in range(num_features):
            # (batch_size, seq_len, vocab_size)
            logits = x[i]
            sample_logits = logits[replaced_token_mask]
            sampled = gumbel_sample(sample_logits, self.sampling_temperature).detach()
            fake_input_ids.index_put_((replaced_token_mask, torch.tensor(i)), sampled)

        # 3. Build rewriting data by concatenating the fake input and the original input
        inputs, labels = [], []
        source_lengths, input_lengths = [], []
        for i in range(batch_size):
            length = batch.lengths[i]
            real = batch.input_ids[i, :length]
            fake = fake_input_ids[i, :length]
            input = torch.cat([fake, self.sep_token_tensor, real[:-1]])
            label = torch.cat([self.pad_token_tensor.repeat((len(fake), 1)), real[1:], self.sep_token_tensor])
            inputs.append(input)
            labels.append(label)
            source_lengths.append(len(fake))
            input_lengths.append(len(input))

        # pad
        input_ids = torch.stack(
            [torch.cat([input, self.pad_token_tensor.repeat(self.whole_seq_len - len(input), 1)]) for input in inputs]
        )
        label_ids = torch.stack(
            [torch.cat([label, self.pad_token_tensor.repeat(self.whole_seq_len - len(label), 1)]) for label in labels]
        )
        padding_mask = (
            torch.arange(self.whole_seq_len, device=model.device)[None, :]
            >= torch.tensor(input_lengths, device=model.device)[:, None]
        )
        # attention mask
        attention_mask = torch.stack(
            [self._get_attention_mask(length, self.whole_seq_len, model.device) for length in source_lengths]
        )
        new_batch = DataBatch(input_ids, label_ids, padding_mask, attention_mask)

        # 4. Feed prefixed data (fake + real) to the original model for seq2seq task
        x = model(new_batch)
        rewriting_loss = model._get_loss(x, new_batch.label_ids)

        if model.training:
            model.log(f"train_loss:fake_mlm", fake_model_loss, sync_dist=True, batch_size=batch_size)
            model.log(f"train_loss:rewriting", rewriting_loss, sync_dist=True, batch_size=batch_size)
        return fake_model_loss + rewriting_loss


class RecoveryTask(TrainingTask):
    def __init__(
        self,
        task_name: str = "recovery",
        kind: Union[str, List[str]] = "span",
        weight: float = 1.0,
        probabilities: Optional[List[float]] = None,
        corruption_rate: float = 0.15,
        mean_span_length: int = 4,
        seq_len: int = 256,
        random_crop: bool = True,
        field_specific_masking: bool = False,
    ):
        super().__init__(f"{kind}_{task_name}", weight)
        self.kinds = kind if isinstance(kind, list) else [kind]
        self.probabilities = probabilities
        self.corruption_rate = corruption_rate
        self.mean_span_length = mean_span_length
        self.seq_len = seq_len
        self.random_crop = random_crop
        self.field_specific_masking = field_specific_masking

    def get_data_collator(self) -> DataCollator:
        masking = get_masking(
            kinds=self.kinds,
            corruption_rate=self.corruption_rate,
            mean_span_length=self.mean_span_length,
            random_crop=self.random_crop,
            probabilities=self.probabilities,
            field_specific_masking=self.field_specific_masking,
        )
        return DataCollatorForRecovery(
            masking=masking,
            seq_len=self.seq_len,
            random_crop=self.random_crop,
        )

    def __call__(self, model, batch: DataBatch, **kwargs) -> torch.Tensor:
        logits = model(batch)
        return model._get_loss(logits, batch.label_ids)


def get_masking(
    kinds: Union[str, List[str]],
    corruption_rate: float,
    mean_span_length: int,
    random_crop: bool,
    probabilities: Optional[List[float]] = None,
    field_specific_masking: bool = False,
):
    def _get_masking(kind: str):
        if kind == "span":
            return RandomSpanMasking(corruption_rate=corruption_rate, mean_span_length=mean_span_length)
        elif kind == "bar":
            return RandomBarMasking(corruption_rate=corruption_rate)
        elif kind == "pitch_ngram":
            return RandomNgramMasking(
                corruption_rate=corruption_rate,
                fallback_mean_span_length=mean_span_length,
                extra_data_field_name="pitch_ngrams",
                field_specific_masking=field_specific_masking,
            )
        elif kind == "rhythm_ngram":
            return RandomNgramMasking(
                corruption_rate=corruption_rate,
                fallback_mean_span_length=mean_span_length,
                extra_data_field_name="rhythm_ngrams",
                field_specific_masking=field_specific_masking,
            )
        elif kind == "skeleton":
            return RandomSkeletonUnitMasking(
                corruption_rate=corruption_rate, rhythm_specific_masking=field_specific_masking
            )
        elif kind == "single":
            return SingleSpanMasking(corruption_rate=corruption_rate)
        elif kind == "fixed_bar":
            return FixedBarMasking(num_past_bars=6, num_middle_bars=4, num_future_bars=6, random_crop=random_crop)
        else:
            raise ValueError(f"Unknown infilling kind: {kind}")

    if len(kinds) == 1:
        return _get_masking(kinds[0])
    else:
        assert probabilities is None or len(probabilities) == len(kinds)
        if probabilities is None:
            probabilities = [1 / len(kinds)] * len(kinds)
        return MultiTargetInfillingMasking([_get_masking(kind) for kind in kinds], probabilities=probabilities)
