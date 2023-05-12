import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import BasePredictionWriter

from .dataset import DataBatch
from .module import CompoundTokenFuser, CustomPositionalEncoding, PositionalEncoding
from .task import TrainingTask
from .tokenizer import MIDITokenizer
from .utils import top_k_top_p_sample

DataBatchDict = Dict[str, DataBatch]


class MelodyModel(pl.LightningModule):
    """Base model for core step logic."""

    def __init__(
        self,
        # Instantiate the tokenizer using config file in the dataset directory
        dataset_dir: str,
        # Model hyperparameters
        embedding_dim: Dict[str, int],
        model_dim: int,
        feedforward_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        use_positional_encoding: bool = False,
        use_custom_positional_encoding: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        tokenizer_config_path = os.path.join(dataset_dir, "tokenizer_config.json")
        if not os.path.exists(tokenizer_config_path):
            raise ValueError(f"Tokenizer config file not found: {tokenizer_config_path}")
        self.tokenizer = MIDITokenizer.from_config(tokenizer_config_path)
        print(self.tokenizer)

        self.num_features = len(self.tokenizer.field_names)

        self.embedding_dim = embedding_dim
        self.model_dim = model_dim
        self.feedforward_dim = feedforward_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.default_seq_len = 512
        self.prefix_source_length = self.default_seq_len

        self.fuser = CompoundTokenFuser(self.tokenizer, embedding_dim, model_dim)

        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(model_dim)

        self.use_custom_positional_encoding = use_custom_positional_encoding
        if use_custom_positional_encoding:
            self.custom_positional_encoding = CustomPositionalEncoding(model_dim, max_seq_len=self.default_seq_len)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation=F.gelu,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

    def _create_causal_attention_mask(self, length: int) -> torch.Tensor:
        return torch.triu(torch.ones((length, length), dtype=torch.bool, device=self.device), diagonal=1)

    def _get_causal_attention_mask(self, length: int) -> torch.Tensor:
        if not hasattr(self, "causal_attention_mask") or self.causal_attention_mask.shape[0] < length:
            self.causal_attention_mask = self._create_causal_attention_mask(max(length, self.default_seq_len))
        return self.causal_attention_mask[:length, :length]

    def _create_prefix_attention_mask(self, source_length: int, length: int) -> torch.Tensor:
        # bidirectional attention mask for prefix sequence
        left_prefix_part = torch.zeros((length, source_length), dtype=torch.bool, device=self.device)

        target_length = length - source_length
        top_right_target_part = torch.ones((source_length, target_length), dtype=torch.bool, device=self.device)

        # causal attention mask for infilling sequence
        bottom_right_target_part = torch.triu(
            torch.ones((target_length, target_length), dtype=torch.bool, device=self.device), diagonal=1
        )

        right_target_part = torch.cat([top_right_target_part, bottom_right_target_part], dim=0)
        return torch.cat([left_prefix_part, right_target_part], dim=1)

    def _get_prefix_attention_mask(self, source_length: int, length: int) -> torch.Tensor:
        if (
            not hasattr(self, "prefix_attention_mask")
            or self.prefix_source_length < source_length
            or self.prefix_attention_mask.shape[0] < length
        ):
            new_source_length = max(source_length, self.prefix_source_length)
            new_length = max(length, 3 * self.default_seq_len)
            self.prefix_attention_mask = self._create_prefix_attention_mask(new_source_length, new_length)
            self.prefix_source_length = new_source_length
        start = self.prefix_source_length - source_length
        end = start + length
        return self.prefix_attention_mask[start:end, start:end]

    def _get_padding_mask(self, lengths: List[int], seq_len: int) -> torch.Tensor:
        padding_mask = torch.zeros((len(lengths), seq_len), dtype=torch.bool, device=self.device)
        for i, length in enumerate(lengths):
            padding_mask[i, length:] = True
        return padding_mask

    def forward(self, batch: DataBatch, return_outputs: bool = False) -> List[torch.Tensor]:
        """Args:
            batch: DatasetBatch
            return_outputs: bool, whether to return transformer encoder outputs
        Returns:
            decoded: list of num_features * (batch_size, seq_len, vocab_size of the feature)
            outputs: (batch_size, seq_len, model_dim), transformer encoder outputs
        """
        batch_size, seq_len, _ = batch.input_ids.shape

        padding_mask = self._get_padding_mask(batch.lengths, seq_len)
        if batch.attention_kind == "full":
            attention_mask = None
        elif batch.attention_kind == "causal":
            attention_mask = self._get_causal_attention_mask(seq_len)
        elif batch.attention_kind == "prefix":
            attention_mask = (
                torch.stack(
                    [self._get_prefix_attention_mask(source_length, seq_len) for source_length in batch.source_lengths]
                )
                .view(batch_size, 1, seq_len, seq_len)
                .expand(-1, self.num_heads, -1, -1)
                .reshape(batch_size * self.num_heads, seq_len, seq_len)
            )

        x = self.fuser(batch.input_ids)
        if self.use_positional_encoding:
            x = self.positional_encoding(x)
        if self.use_custom_positional_encoding and batch.positional_ids is not None:
            x = self.custom_positional_encoding(x, batch.positional_ids)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask, mask=attention_mask)
        decoded = self.fuser.decode(x)
        if return_outputs:
            return decoded, x
        else:
            return decoded

    def _get_loss(
        self, logits: List[torch.Tensor], label_ids: torch.Tensor, return_parts: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        losses = []
        for i, logit in enumerate(logits):
            # (batch_size, seq_len, vocab_size) -> (batch_size, vocab_size, seq_len)
            loss = F.cross_entropy(
                logit.transpose(1, 2), label_ids[:, :, i], ignore_index=self.tokenizer.pad_token_ids[i]
            )
            losses.append(loss)
        loss = torch.stack(losses).mean()
        if return_parts:
            return loss, losses
        return loss


class MelodyPretrainModel(MelodyModel):
    """Use this subclass for pretraining or finetuning the model."""

    def __init__(
        self,
        # Instantiate the tokenizer using config file in the dataset directory
        dataset_dir: str,
        # Model hyperparameters
        embedding_dim: Dict[str, int],
        model_dim: int,
        feedforward_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        # Optimizer hyperparameters
        lr: float,
        betas: Tuple[float, float],
        epsilon: float,
        weight_decay: float,
        warmup_percent: float,
        fixed_lr: bool = False,
        # Training configuration
        use_positional_encoding: bool = False,
        use_custom_positional_encoding: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset_dir=dataset_dir,
            embedding_dim=embedding_dim,
            model_dim=model_dim,
            feedforward_dim=feedforward_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
            use_custom_positional_encoding=use_custom_positional_encoding,
        )

        self.lr = lr
        self.betas = betas
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.warmup_percent = warmup_percent
        self.fixed_lr = fixed_lr

        self.tasks: Dict[str, TrainingTask] = {}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, betas=self.betas, eps=self.epsilon, weight_decay=self.weight_decay
        )
        if self.fixed_lr:
            return optimizer
        else:
            total_steps = (
                self.trainer.max_steps if self.trainer.max_steps > 0 else self.trainer.estimated_stepping_batches
            )
            print(f"Total steps: {total_steps}")
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=total_steps,
                anneal_strategy="cos",
                pct_start=self.warmup_percent,
            )
            scheduler = {"scheduler": lr_scheduler, "interval": "step"}
            return [optimizer], [scheduler]

    def register_task(self, task: TrainingTask):
        self.tasks[task.task_name] = task
        task.register_extra_modules(self)

    def _get_batch_size(self, batch: DataBatchDict) -> int:
        if len(self.tasks) == 1:
            return batch.input_ids.shape[0]
        return next(iter(batch.values())).input_ids.shape[0]

    def _shared_step(self, batch: DataBatchDict) -> torch.Tensor:
        if len(self.tasks) == 1:
            task = next(iter(self.tasks.values()))
            return task(self, batch), {}
        losses = {}
        for task_name, task in self.tasks.items():
            losses[task_name] = task(self, batch[task_name])
        weighted_losses = [loss * self.tasks[task_name].weight for task_name, loss in losses.items()]
        loss = torch.stack(weighted_losses).sum()
        return loss, losses

    def training_step(self, batch: DataBatchDict, batch_idx: int) -> Tuple[torch.Tensor, Dict[str, any]]:
        batch_size = self._get_batch_size(batch)
        loss, losses = self._shared_step(batch)
        if len(self.tasks) > 1:
            for task_name in self.tasks:
                self.log(f"train_loss:{task_name}", losses[task_name], sync_dist=True, batch_size=batch_size)
        self.log("train_loss", loss, sync_dist=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch: DataBatchDict, batch_idx: int) -> Tuple[torch.Tensor, Dict[str, any]]:
        batch_size = self._get_batch_size(batch)
        loss, _ = self._shared_step(batch)
        self.log("val_loss", loss, sync_dist=True, batch_size=batch_size)
        return loss


class MelodyTestingModel(MelodyModel):
    def __init__(
        self,
        # Instantiate the tokenizer using config file in the dataset directory
        dataset_dir: str,
        # Model hyperparameters
        embedding_dim: Dict[str, int],
        model_dim: int,
        feedforward_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        # Testing hyperparameters
        max_seq_len: Optional[int] = None,
        perplexity_stride: Optional[int] = None,
        use_positional_encoding: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset_dir=dataset_dir,
            embedding_dim=embedding_dim,
            model_dim=model_dim,
            feedforward_dim=feedforward_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
        )

        self.max_seq_len = max_seq_len
        self.perplexity_stride = perplexity_stride

        self.pad_token_tensor = torch.from_numpy(self.tokenizer.pad_token_ids).long()

    def get_ppl_directly(self, batch: DataBatch) -> torch.Tensor:
        logits = self(batch)
        loss = self._get_loss(logits, batch.label_ids)
        return torch.exp(loss)

    def get_ppl_strided(self, batch: DataBatch) -> torch.Tensor:
        seq_len = max(batch.lengths)
        if self.pad_token_tensor.device != self.device:
            self.pad_token_tensor = self.pad_token_tensor.to(self.device)

        nlls = []
        valid_counts = []
        previous_end_index = 0
        for start_index in range(0, seq_len, self.perplexity_stride):
            end_index = min(start_index + self.max_seq_len, seq_len)
            target_length = end_index - previous_end_index
            if target_length <= 0:
                break

            # Mask out context tokens for labels
            new_label_ids = batch.label_ids[:, start_index:end_index, :].clone()
            new_label_ids[:, :-target_length] = self.pad_token_tensor
            new_lengths = [max(length - start_index, 0) for length in batch.lengths]
            target_lengths = [max(min(target_length, length - previous_end_index), 0) for length in batch.lengths]
            new_batch = DataBatch(
                input_ids=batch.input_ids[:, start_index:end_index, :],
                label_ids=new_label_ids,
                attention_kind="causal",
                lengths=new_lengths,
            )
            logits = self(new_batch)
            loss = self._get_loss(logits, new_batch.label_ids)

            valid_count = sum(target_lengths)
            negative_log_likelihood = loss * valid_count
            nlls.append(negative_log_likelihood)
            valid_counts.append(valid_count)
            previous_end_index = end_index

        assert sum(valid_counts) == sum(batch.lengths)
        return torch.exp(torch.stack(nlls).sum() / sum(valid_counts))

    def test_step(self, batch: DataBatch, batch_idx: int) -> torch.Tensor:
        """Calculate strided fixed-length perplexity.
        Reference: https://huggingface.co/docs/transformers/perplexity
        """
        if batch.attention_kind == "prefix":
            ppl = self.get_ppl_directly(batch)
        elif batch.attention_kind == "causal":
            ppl = self.get_ppl_strided(batch)
        else:
            raise ValueError(f"Unsupported attention kind: {batch.attention_kind}")

        self.log("perplexity", ppl, batch_size=batch.input_ids.shape[0], sync_dist=True)
        return ppl


class MelodyCompletionModel(MelodyModel):
    """Use this subclass for the melody completion downstream task."""

    def __init__(
        self,
        # Instantiate the tokenizer using config file in the dataset directory
        dataset_dir: str,
        # Model hyperparameters
        embedding_dim: Dict[str, int],
        model_dim: int,
        feedforward_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        # Inference hyperparameters
        conditional_bar_length: int,
        prediction_bar_length: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        max_length: int = 512,
        times_to_predict: int = 1,
        use_positional_encoding: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset_dir=dataset_dir,
            embedding_dim=embedding_dim,
            model_dim=model_dim,
            feedforward_dim=feedforward_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
        )

        self.conditional_bar_length = conditional_bar_length
        self.prediction_bar_length = prediction_bar_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_length = max_length
        self.times_to_predict = times_to_predict

        self.attention_mask = torch.triu(torch.ones((max_length, max_length), dtype=torch.bool), diagonal=1)
        self.bos_token_tensor = torch.tensor(self.tokenizer.bos_token_ids, dtype=torch.long)
        self.eos_token_tensor = torch.tensor(self.tokenizer.eos_token_ids, dtype=torch.long)

        self.bar_field_index = self.tokenizer.field_indices["bar"]
        self.bar_vocab_size = self.tokenizer.vocab_sizes[self.bar_field_index]

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> List[torch.Tensor]:
        x = self.fuser(input_ids)
        if self.use_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer_encoder(x, mask=attn_mask)
        return self.fuser.decode(x)

    def predict_step(self, batch: DataBatch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        if self.attention_mask.device != self.device:
            self.attention_mask = self.attention_mask.to(self.device)
            self.bos_token_tensor = self.bos_token_tensor.to(self.device)
            self.eos_token_tensor = self.eos_token_tensor.to(self.device)

        input_ids = self.bos_token_tensor.expand(self.times_to_predict, 1, -1)
        reached_eos = np.zeros(self.times_to_predict, dtype=np.bool_)
        while input_ids.shape[1] < self.max_length:
            seq_len = input_ids.shape[1]
            attn_mask = self.attention_mask[:seq_len, :seq_len]
            logits = self(input_ids, attn_mask=attn_mask)

            sampled_tokens = []
            for logit in logits:
                # Decode according to the sampling strategy
                sampled_token = top_k_top_p_sample(
                    logit[:, -1, :], top_k=self.top_k, top_p=self.top_p, temperature=self.temperature
                )
                sampled_tokens.append(sampled_token)
            # (batch_size, num_features)
            sampled_tokens = torch.cat(sampled_tokens, dim=-1)

            # until <EOS> token is generated or the predicted bar length is reached
            for batch_index in range(self.times_to_predict):
                token = sampled_tokens[batch_index]
                if (
                    torch.all(token == self.eos_token_tensor)
                    or self.prediction_bar_length <= token[self.bar_field_index] < self.bar_vocab_size
                ):
                    reached_eos[batch_index] = True
            if np.all(reached_eos):
                break

            # Append the sampled token to the input
            input_ids = torch.cat([input_ids, sampled_tokens[:, None, :]], dim=1)

        return input_ids


class MelodyInfillingModel(MelodyModel):
    """Use this subclass for the melody infilling downstream task."""

    def __init__(
        self,
        # Instantiate the tokenizer using config file in the dataset directory
        dataset_dir: str,
        # Model hyperparameters
        embedding_dim: Dict[str, int],
        model_dim: int,
        feedforward_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        # Inference hyperparameters
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        max_length: int = 512,
        use_positional_encoding: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset_dir=dataset_dir,
            embedding_dim=embedding_dim,
            model_dim=model_dim,
            feedforward_dim=feedforward_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
        )

        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_length = max_length

        self.sep_token_tensor = torch.tensor(self.tokenizer.sep_token_ids, dtype=torch.long)
        self.long_mask_token_tensor = torch.tensor(self.tokenizer.long_mask_token_ids, dtype=torch.long)

        self.bar_field_index = self.tokenizer.field_indices["bar"]
        self.bar_vocab_size = self.tokenizer.vocab_sizes[self.bar_field_index]

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> List[torch.Tensor]:
        x = self.fuser(input_ids)
        if self.use_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer_encoder(x, mask=attn_mask)
        return self.fuser.decode(x)

    def predict_step(self, batch: DataBatch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        input_ids = batch.input_ids
        assert batch.label_ids is None
        batch_size, original_len, _ = input_ids.shape
        assert batch_size == 1, "Only support batch size of 1 for prediction for now"
        if self.sep_token_tensor.device != self.device:
            self.sep_token_tensor = self.sep_token_tensor.to(self.device)
            self.long_mask_token_tensor = self.long_mask_token_tensor.to(self.device)

        mask_token_position = torch.all(input_ids[0, :] == self.long_mask_token_tensor, dim=1).nonzero()[0].item()
        future_start_bar = input_ids[0, mask_token_position + 1, self.bar_field_index]

        # Add one seperator token to the end
        input_ids = torch.cat([input_ids, self.sep_token_tensor.unsqueeze(0).unsqueeze(0)], dim=1)

        while input_ids.shape[1] < self.max_length:
            seq_len = input_ids.shape[1]
            attention_mask = self._get_prefix_attention_mask(original_len, seq_len)
            logits = self(input_ids, attention_mask)

            sampled_tokens = []
            for logit in logits:
                # Decode according to the sampling strategy
                sampled_token = top_k_top_p_sample(
                    logit[:, -1, :], top_k=self.top_k, top_p=self.top_p, temperature=self.temperature
                )[0]
                sampled_tokens.append(sampled_token)
            sampled_tokens = torch.cat(sampled_tokens, dim=-1)

            # token = self.tokenizer.convert_id_to_token(sampled_tokens.cpu().numpy())
            # print(token)

            # until <SEP> token is predicted
            if torch.all(sampled_tokens == self.sep_token_tensor):
                break
            # or until the first bar of future part is reached
            if future_start_bar <= sampled_tokens[self.bar_field_index] <= self.bar_vocab_size:
                break

            # Append the sampled token to the input
            input_ids = torch.cat([input_ids, sampled_tokens.unsqueeze(0).unsqueeze(0)], dim=1)

        # Rearrange the sequence to the original order
        input_ids = torch.cat(
            [
                input_ids[:, :mask_token_position, :],
                input_ids[:, original_len + 1 :, :],
                input_ids[:, mask_token_position + 1 : original_len + 1, :],
            ],
            dim=1,
        )
        return input_ids


class CustomWriter(BasePredictionWriter):
    """Write the prediction to a MIDI file."""

    def __init__(self, output_dir: str):
        super().__init__(write_interval="batch")
        self.output_dir = output_dir

    def decode(
        self, token_ids: np.ndarray, tokenizer: MIDITokenizer, prediction_bar_length: Optional[int] = None, **kwargs
    ):
        # (seq_len, num_features)
        bar_field_index = tokenizer.field_indices["bar"]
        bar_vocab_size = tokenizer.vocab_sizes[bar_field_index]
        # Crop the sequence at the first <EOS> token or where the bar length is reached
        for index in range(token_ids.shape[0]):
            token = token_ids[index]
            if (
                np.all(token == tokenizer.eos_token_ids)
                or prediction_bar_length is not None
                and prediction_bar_length <= token[bar_field_index] < bar_vocab_size
            ):
                token_ids = token_ids[:index]
                break
        midi = tokenizer.decode(token_ids)
        midi = tokenizer.filter_overlapping_notes(midi)
        return midi

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: MelodyCompletionModel,
        prediction: torch.Tensor,
        batch_indices: Optional[Sequence[int]],
        batch: DataBatch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if prediction is None:
            return
        filename = batch.filenames[0] if batch.filenames is not None else str(batch_idx)

        # (batch_size, seq_len, num_features)
        prediction = prediction.cpu().numpy()
        for index in range(prediction.shape[0]):
            if prediction.shape[0] > 1:
                dest_path = os.path.join(self.output_dir, f"{filename}+{index}.mid")
            else:
                dest_path = os.path.join(self.output_dir, f"{filename}+{dataloader_idx}.mid")
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            midi = self.decode(
                prediction[index],
                tokenizer=pl_module.tokenizer,
                prediction_bar_length=pl_module.prediction_bar_length
                if hasattr(pl_module, "prediction_bar_length")
                else None,
            )
            midi.dump(dest_path)
