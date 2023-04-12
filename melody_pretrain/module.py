import math
from typing import Dict, List

import torch
import torch.nn as nn

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
        self.total_embedding_dim = sum(embedding_dim[field_name] for field_name in tokenizer.field_names)

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


class PositionalEncoding(nn.Module):
    """Positional encoding for the transformer."""
    
    def __init__(self, model_dim: int, max_seq_len: int = 2048) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.register_buffer("positional_encoding", self._get_positional_encoding())

    def _get_positional_encoding(self) -> torch.Tensor:
        positional_encoding = torch.zeros(self.max_seq_len, self.model_dim)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.model_dim, 2).float() * (-math.log(10000.0) / self.model_dim))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        return positional_encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (batch_size, seq_len, model_dim)
        Returns:
            x: (batch_size, seq_len, model_dim)
        """
        _, seq_len, _ = x.shape
        x = x + self.positional_encoding[:seq_len].unsqueeze(0)
        return x
