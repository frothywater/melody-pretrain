import os
from typing import List, Optional, Sequence, Tuple, Union

import lightning as pl
from lightning.pytorch.callbacks import BasePredictionWriter
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import DatasetBatch
from .tokenizer import MIDITokenizer


class CompoundTokenFuser(nn.Module):
    """Fuses multiple token embeddings into a single embedding."""

    def __init__(self, tokenizer: MIDITokenizer, embedding_dim: Union[int, Tuple[int, ...]], model_dim: int) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.num_features = len(tokenizer.field_names)
        self.field_sizes = tokenizer.field_sizes
        self.total_field_size = sum(self.field_sizes)

        self.model_dim = model_dim
        if isinstance(embedding_dim, int):
            self.embedding_dims = [embedding_dim for _ in range(self.num_features)]
        else:
            assert len(embedding_dim) == self.num_features, "embedding_dim must be int or list of length num_features"
            self.embedding_dims = embedding_dim
        self.total_embedding_dim = sum(self.embedding_dims)

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=field_size, embedding_dim=embedding_dim_, padding_idx=pad_token_id)
                for field_size, embedding_dim_, pad_token_id in zip(
                    self.field_sizes, self.embedding_dims, tokenizer.pad_token_ids
                )
            ]
        )
        self.encoder = nn.Linear(self.total_embedding_dim, model_dim)
        self.decoder = nn.Linear(model_dim, self.total_field_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            input_ids: (batch_size, seq_len, num_features)
        Returns:
            fused: (batch_size, seq_len, model_dim)
        """
        _, _, num_features = x.shape
        assert num_features == self.num_features, f"num_features must be {self.num_features}"

        # embeddings: (batch_size, seq_len, total_embedding_dim)
        x = torch.concat([embedding(x[:, :, i]) for i, embedding in enumerate(self.embeddings)], dim=2)
        # fused: (batch_size, seq_len, model_dim)
        x = self.encoder(x)
        return x

    def decode(self, fused: torch.Tensor) -> List[torch.Tensor]:
        """Args:
            fused: (batch_size, seq_len, model_dim)
        Returns:
            logits: List[torch.Tensor] of length num_features,
            each of shape (batch_size, seq_len, vocab_size of the feature)
        """
        # embeddings: (batch_size, seq_len, total_field_size)
        embeddings = self.decoder(fused)
        return torch.split(embeddings, self.field_sizes, dim=2)


class MelodyModel(pl.LightningModule):
    """Base model for core step logic."""

    def __init__(
        self,
        # Instantiate the tokenizer using config file in the dataset directory
        dataset_dir: str,
        # Model hyperparameters
        embedding_dim: Union[int, Tuple[int, ...]],  # embedding_dim: (bar, position, duration, pitch)
        model_dim: int,
        feedforward_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        **kwargs,
    ) -> None:
        super().__init__()

        tokenizer_config_path = os.path.join(dataset_dir, "tokenizer_config.json")
        if not os.path.exists(tokenizer_config_path):
            raise ValueError(f"Tokenizer config file not found: {tokenizer_config_path}")
        self.tokenizer = MIDITokenizer.from_config(tokenizer_config_path)

        self.num_features = len(self.tokenizer.field_names)

        self.num_heads = num_heads

        self.fuser = CompoundTokenFuser(self.tokenizer, embedding_dim, model_dim)
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

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Args:
            x: (batch_size, seq_len, num_features)
            padding_mask: (batch_size, seq_len), optional
            attn_mask: (batch_size, seq_len, seq_len), optional
        Returns:
            x: list of num_features * (batch_size, seq_len, vocab_size of the feature)
        """
        x = self.fuser(x)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask, mask=attn_mask)
        x = self.fuser.decode(x)
        return x

    def _get_loss(
        self, logits: List[torch.Tensor], label_ids: torch.Tensor, return_parts: bool = False
    ) -> torch.Tensor:
        losses = []
        for i, logit in enumerate(logits):
            # (batch_size, seq_len, vocab_size) -> (batch_size, vocab_size, seq_len)
            loss = F.cross_entropy(
                logit.transpose(1, 2), label_ids[:, :, i], ignore_index=self.tokenizer.pad_token_ids[i]
            )
            losses.append(loss)
        loss = torch.mean(torch.stack(losses))
        if return_parts:
            return loss, losses
        return loss

    def _shared_step(self, batch: DatasetBatch) -> List[torch.Tensor]:
        batch_size, seq_len, _ = batch.input_ids.shape
        input_ids, _, padding_mask, attention_mask = batch
        if attention_mask is not None and len(attention_mask.shape) == 3:
            # attention_mask: (batch_size * num_heads, seq_len, seq_len)
            attention_mask = attention_mask.expand(self.num_heads, -1, -1, -1).reshape(
                self.num_heads * batch_size, seq_len, seq_len
            )
        logits = self(input_ids, padding_mask, attention_mask)
        return logits

    def training_step(self, batch: DatasetBatch, batch_idx: int) -> torch.Tensor:
        logits = self._shared_step(batch)
        loss, losses = self._get_loss(logits, batch.label_ids, return_parts=True)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log_dict(
            {f"train_loss:{self.tokenizer.field_names[i]}": loss for i, loss in enumerate(losses)}, sync_dist=True
        )
        return loss

    def validation_step(self, batch: DatasetBatch, batch_idx: int) -> torch.Tensor:
        logits = self._shared_step(batch)
        loss = self._get_loss(logits, batch.label_ids)
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def test_step(self, batch: DatasetBatch, batch_idx: int) -> torch.Tensor:
        logits = self._shared_step(batch)
        loss = self._get_loss(logits, batch.label_ids)
        self.log("test_loss", loss, sync_dist=True)
        return loss


class MelodyPretrainModel(MelodyModel):
    """Use this subclass for pretraining or finetuning the model."""

    def __init__(
        self,
        # Instantiate the tokenizer using config file in the dataset directory
        dataset_dir: str,
        # Model hyperparameters
        embedding_dim: Union[int, Tuple[int, ...]],  # embedding_dim: (bar, position, duration, pitch)
        model_dim: int,
        feedforward_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        # Optimizer hyperparameters
        lr: float,
        betas: Tuple[float, float],
        weight_decay: float,
        warmup_percent: float,
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
        )

        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_percent = warmup_percent

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.max_steps,
            anneal_strategy="cos",
            pct_start=self.warmup_percent,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step"}
        return [optimizer], [scheduler]


class MelodyCompletionModel(MelodyModel):
    """Use this subclass for the melody completion downstream task."""
    def __init__(
        self,
        # Instantiate the tokenizer using config file in the dataset directory
        dataset_dir: str,
        # Model hyperparameters
        embedding_dim: Union[int, Tuple[int, ...]],  # embedding_dim: (bar, position, duration, pitch)
        model_dim: int,
        feedforward_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        # Inference hyperparameters
        conditional_bar_length: int,
        prediction_bar_length: int,
        temperature: float,
        top_k: int,
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
        )

        self.conditional_bar_length = conditional_bar_length
        self.prediction_bar_length = prediction_bar_length
        self.temperature = temperature
        self.top_k = top_k

    def predict_step(self, batch: DatasetBatch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        input_ids, _, _, _ = batch
        batch_size, _, _ = input_ids.shape
        assert batch_size == 1, "Only support batch size of 1 for prediction for now"

        # Crop the input to the conditional bar length
        bar_field_index = self.tokenizer.field_indices["bar"]
        conditional_bar_token_id = self.tokenizer.encoder["bar"][self.conditional_bar_length]
        prediction_bar_token_id = self.tokenizer.encoder["bar"][self.prediction_bar_length]

        conditional_bar_test = (input_ids[0, :, bar_field_index] == conditional_bar_token_id).nonzero()
        assert conditional_bar_test.shape[1] >= 1, "No conditional bar token found in the input"
        conditional_bar_index = conditional_bar_test[0, 0].item()
        input_ids = input_ids[:, :conditional_bar_index, :]

        # Inference on a single sequence
        while True:
            seq_len = input_ids.shape[1]
            attention_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1).to(
                input_ids.device
            )
            logits = self(input_ids, attn_mask=attention_mask)
            sampled_tokens = []
            for logit in logits:
                # Decode according to the sampling strategy
                sampled_token = top_k_sample(logit[0, -1, :], k=self.top_k, t=self.temperature)
                sampled_tokens.append(sampled_token)
            sampled_tokens = torch.cat(sampled_tokens, dim=-1)

            token = self.tokenizer.convert_id_to_token(sampled_tokens.cpu().numpy())
            # print(token)

            # until the desired bar length is reached
            if sampled_tokens[bar_field_index] == prediction_bar_token_id:
                break

            # Append the sampled token to the input
            input_ids = torch.cat([input_ids, sampled_tokens.unsqueeze(0).unsqueeze(0)], dim=1)

        return input_ids


class MelodyInfillingModel(MelodyModel):
    """Use this subclass for the melody infilling downstream task."""
    def __init__(
        self,
        # Instantiate the tokenizer using config file in the dataset directory
        dataset_dir: str,
        # Model hyperparameters
        embedding_dim: Union[int, Tuple[int, ...]],  # embedding_dim: (bar, position, duration, pitch)
        model_dim: int,
        feedforward_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        # Inference hyperparameters
        num_middle_bars: int,
        temperature: float,
        top_k: int,
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
        )

        self.num_middle_bars = num_middle_bars
        self.temperature = temperature
        self.top_k = top_k

        self.sep_token = torch.from_numpy(self.tokenizer.sep_token_ids)

    def predict_step(self, batch: DatasetBatch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        input_ids, _, _, _ = batch
        batch_size, original_len, _ = input_ids.shape
        assert batch_size == 1, "Only support batch size of 1 for prediction for now"

        bar_field_index = self.tokenizer.field_indices["bar"]
        bar_mask_token_id = self.tokenizer.mask_token_ids[bar_field_index]
        bar_sep_token_id = self.tokenizer.sep_token_ids[bar_field_index]
        mask_token_position = (input_ids[0, :, bar_field_index] == bar_mask_token_id).nonzero()[0].item()
        future_bar_token_id = input_ids[0, mask_token_position + 1, bar_field_index]

        # Add one seperator token to the end
        self.sep_token = self.sep_token.to(input_ids.device)
        input_ids = torch.cat([input_ids, self.sep_token.unsqueeze(0).unsqueeze(0)], dim=1)
        # Attention mask for the prefix part
        attention_mask = torch.zeros((original_len, original_len), dtype=torch.bool).to(input_ids.device)

        while True:
            seq_len = input_ids.shape[1]

            # Construct causal part of the attention mask for the next token
            previous_len = attention_mask.shape[0]
            mask_column = torch.ones((previous_len, 1), dtype=torch.bool).to(attention_mask.device)
            attn_row = torch.zeros((1, seq_len), dtype=torch.bool).to(attention_mask.device)
            attention_mask = torch.cat([attention_mask, mask_column], dim=1)
            attention_mask = torch.cat([attention_mask, attn_row], dim=0)
            # (previous_len, previous_len) -> (seq_len, seq_len)

            # print("input_ids:", input_ids)
            # print("attention_mask:", attention_mask)

            logits = self(input_ids, attn_mask=attention_mask)
            sampled_tokens = []
            for logit in logits:
                # Decode according to the sampling strategy
                sampled_token = top_k_sample(logit[0, -1, :], k=self.top_k, t=self.temperature)
                sampled_tokens.append(sampled_token)
            sampled_tokens = torch.cat(sampled_tokens, dim=-1)

            token = self.tokenizer.convert_id_to_token(sampled_tokens.cpu().numpy())
            print(token)

            # until the model predicts a seperator token, or the desired bar length is reached
            if (
                sampled_tokens[bar_field_index] == bar_sep_token_id
                or sampled_tokens[bar_field_index] == future_bar_token_id
            ):
                break

            # Append the sampled token to the input
            input_ids = torch.cat([input_ids, sampled_tokens.unsqueeze(0).unsqueeze(0)], dim=1)

        # Rearrange the sequence to the original order
        input_ids = torch.cat(
            [
                input_ids[:, :mask_token_position, :],
                input_ids[:, original_len:, :],
                input_ids[:, mask_token_position + 1 : original_len, :],
            ],
            dim=1,
        )
        return input_ids


def top_k_sample(logits: torch.Tensor, k: int, t: float = 1.0) -> torch.Tensor:
    """Sample from the top k logits with temperature t"""
    assert k > 0, "k must be greater than 0"
    assert t > 0, "t must be greater than 0"
    logits = logits / t
    top_k_logits, top_k_indices = torch.topk(logits, k)
    top_k_probs = F.softmax(top_k_logits, dim=-1)
    sampled_index = torch.multinomial(top_k_probs, 1)
    sampled_token = top_k_indices.gather(0, sampled_index)
    return sampled_token


class CustomWriter(BasePredictionWriter):
    """Write the prediction to a MIDI file."""

    # TODO: Write correct file name.

    def __init__(self, output_dir: str):
        super().__init__(write_interval="batch")
        self.output_dir = output_dir

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: MelodyPretrainModel,
        prediction: torch.Tensor,
        batch_indices: Optional[Sequence[int]],
        batch: DatasetBatch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        dest_path = os.path.join(self.output_dir, f"{batch_idx}.mid")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        prediction = prediction[0].cpu().numpy()
        midi = pl_module.tokenizer.decode(prediction)
        midi.dump(dest_path)
