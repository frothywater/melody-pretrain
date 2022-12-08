import os
from typing import List, Optional, Tuple, Union

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import DatasetBatch
from .tokenizer import MIDITokenizer


class CompoundTokenFuser(nn.Module):
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


class MelodyPretrainModel(pl.LightningModule):
    def __init__(
        self,
        # Instantiate the tokenizer using config file in the dataset directory
        dataset_dir: str,
        # embedding_dim: (bar, position, duration, pitch)
        embedding_dim: Union[int, Tuple[int, ...]],
        model_dim: int,
        feedforward_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        lr: float,
        betas: Tuple[float, float],
        weight_decay: float,
        warmup_percent: float,
        **kwargs,
    ) -> None:
        super().__init__()

        tokenizer_config_path = os.path.join(dataset_dir, "tokenizer_config.json")
        if not os.path.exists(tokenizer_config_path):
            raise ValueError(f"Tokenizer config file not found: {tokenizer_config_path}")
        self.tokenizer = MIDITokenizer.from_config(tokenizer_config_path)

        self.num_features = len(self.tokenizer.field_names)

        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_percent = warmup_percent
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

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Args:
            x: (batch_size, seq_len, num_features)
            padding_mask: (batch_size, seq_len)
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
