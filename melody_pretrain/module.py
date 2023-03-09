from typing import Dict, List

import torch
import torch.nn as nn

from .dataset import span_indices_padding_index
from .tokenizer import MIDITokenizer


class CompoundTokenFuser(nn.Module):
    """Fuses multiple token embeddings into a single embedding."""

    def __init__(self, tokenizer: MIDITokenizer, embedding_dim: Dict[str, int], model_dim: int) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.num_features = len(tokenizer.field_names)
        self.field_sizes = tokenizer.field_sizes
        self.total_field_size = sum(self.field_sizes)

        self.model_dim = model_dim
        self.total_embedding_dim = sum(embedding_dim.values())

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=field_size, embedding_dim=embedding_dim[field_name], padding_idx=pad_token_id
                )
                for field_name, field_size, pad_token_id in zip(
                    tokenizer.field_names, tokenizer.field_sizes, tokenizer.pad_token_ids
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


class SpanPositionalEncoding(nn.Module):
    """Learnable positional encoding added on [MASK] and [SEP] tokens for advanced span prediction."""

    def __init__(self, model_dim: int, max_length: int = 128) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.max_length = max_length

        self.positional_encoding = nn.Embedding(max_length, model_dim, padding_idx=span_indices_padding_index)

    def forward(self, span_indices: torch.Tensor) -> torch.Tensor:
        """Args:
            span_indices: (batch_size, seq_len)
        Returns:
            span_encoding: (batch_size, seq_len, model_dim)
        """
        return self.positional_encoding(span_indices)
