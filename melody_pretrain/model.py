import argparse
from typing import Iterable, List, Union, Optional

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dataset.tokenizer import MIDITokenizer
from .dataset.data_module import DatasetBatch


class CompoundTokenFuser(nn.Module):
    def __init__(self, tokenizer: MIDITokenizer, embedding_dim: Union[int, Iterable[int]], model_dim: int) -> None:
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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Args:
            input_ids: (batch_size, seq_len, num_features)
        Returns:
            fused: (batch_size, seq_len, model_dim)
        """
        _, _, num_features = input_ids.shape
        assert num_features == self.num_features, f"num_features must be {self.num_features}"

        # embeddings: (batch_size, seq_len, total_embedding_dim)
        embeddings = torch.concat([embedding(input_ids[:, :, i]) for i, embedding in enumerate(self.embeddings)], dim=2)
        # fused: (batch_size, seq_len, model_dim)
        fused = self.encoder(embeddings)
        return fused

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
        tokenizer: MIDITokenizer,
        embedding_dim: Union[int, Iterable[int]],
        model_dim: int,
        feedforward_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        lr: float,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore="tokenizer")

        self.tokenizer = tokenizer
        self.num_features = len(tokenizer.field_names)

        self.lr = lr
        self.num_heads = num_heads

        self.fuser = CompoundTokenFuser(tokenizer, embedding_dim, model_dim)
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

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("MelodyPretrainModel")
        parser.add_argument("--embedding_dim", type=int, nargs=4, default=(32, 256, 256, 512))
        parser.add_argument("--model_dim", type=int, default=512)
        parser.add_argument("--feedforward_dim", type=int, default=2048)
        parser.add_argument("--num_layers", type=int, default=6)
        parser.add_argument("--num_heads", type=int, default=8)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--lr", type=float, default=4e-4)
        return parent_parser

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

    def get_loss(self, logits: List[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        losses = []
        for i, logit in enumerate(logits):
            # (batch_size, seq_len, vocab_size) -> (batch_size, vocab_size, seq_len)
            logit = logit.transpose(1, 2)
            loss = F.cross_entropy(logit, labels[:, :, i], ignore_index=self.tokenizer.pad_token_ids[i])
            losses.append(loss)
        loss = torch.mean(torch.stack(losses))
        return loss

    def step(self, batch: DatasetBatch) -> torch.Tensor:
        # attention_mask: (batch_size * num_heads, seq_len, seq_len)
        batch_size, seq_len, _ = batch.input_ids.shape
        attn_mask = batch.attention_mask.expand(self.num_heads, -1, -1, -1).reshape(
            self.num_heads * batch_size, seq_len, seq_len
        )
        logits = self(batch.input_ids, batch.padding_mask, attn_mask)
        loss = self.get_loss(logits, batch.label_ids)
        return loss

    def training_step(self, batch: DatasetBatch, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: DatasetBatch, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch)
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
        return [optimizer], [scheduler]
