"""Small auditable BPR-MF and SASRec models for the locked MovieLens study."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class BPRMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int) -> None:
        super().__init__()
        if min(n_users, n_items, embedding_dim) <= 0:
            raise ValueError("BPR dimensions must be positive")
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def pairwise_loss(
        self, users: torch.Tensor, positive_items: torch.Tensor, negative_items: torch.Tensor
    ) -> torch.Tensor:
        user = self.user_embedding(users)
        positive = self.item_embedding(positive_items)
        negative = self.item_embedding(negative_items)
        margin = (user * positive).sum(dim=-1) - (user * negative).sum(dim=-1)
        return -F.logsigmoid(margin).mean()

    def score_candidates(self, users: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        if candidates.ndim != 2 or users.ndim != 1 or candidates.shape[0] != users.shape[0]:
            raise ValueError("BPR candidate scoring shape mismatch")
        user = self.user_embedding(users).unsqueeze(1)
        items = self.item_embedding(candidates)
        return (user * items).sum(dim=-1)


class SASRec(nn.Module):
    """Causal self-attention sequential recommender with zero reserved for padding."""

    def __init__(
        self,
        n_items: int,
        embedding_dim: int,
        maximum_sequence_length: int,
        transformer_blocks: int,
        attention_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if n_items <= 0 or embedding_dim <= 0 or maximum_sequence_length <= 0:
            raise ValueError("SASRec dimensions must be positive")
        if embedding_dim % attention_heads:
            raise ValueError("SASRec embedding dimension must divide attention heads")
        self.maximum_sequence_length = maximum_sequence_length
        self.embedding_dim = embedding_dim
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(maximum_sequence_length, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=attention_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=transformer_blocks)
        self.output_norm = nn.LayerNorm(embedding_dim)
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def encode(self, sequences: torch.Tensor) -> torch.Tensor:
        if sequences.ndim != 2 or sequences.shape[1] > self.maximum_sequence_length:
            raise ValueError("SASRec sequence shape mismatch")
        batch, length = sequences.shape
        positions = torch.arange(length, device=sequences.device).unsqueeze(0).expand(batch, -1)
        padding = sequences.eq(0)
        hidden = self.item_embedding(sequences) * (self.embedding_dim**0.5)
        hidden = self.dropout(hidden + self.position_embedding(positions))
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=sequences.device), diagonal=1
        )
        hidden = self.encoder(hidden, mask=causal, src_key_padding_mask=padding)
        hidden = self.output_norm(hidden)
        return hidden.masked_fill(padding.unsqueeze(-1), 0.0)

    def pointwise_loss(
        self,
        input_sequences: torch.Tensor,
        positive_targets: torch.Tensor,
        negative_targets: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encode(input_sequences)
        positive = self.item_embedding(positive_targets)
        negative = self.item_embedding(negative_targets)
        mask = positive_targets.ne(0)
        if not torch.any(mask):
            raise ValueError("SASRec batch has no training targets")
        positive_logits = (hidden * positive).sum(dim=-1)[mask]
        negative_logits = (hidden * negative).sum(dim=-1)[mask]
        return (
            F.binary_cross_entropy_with_logits(positive_logits, torch.ones_like(positive_logits))
            + F.binary_cross_entropy_with_logits(negative_logits, torch.zeros_like(negative_logits))
        )

    def score_candidates(
        self,
        sequences: torch.Tensor,
        lengths: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        if candidates.ndim != 2 or sequences.shape[0] != candidates.shape[0]:
            raise ValueError("SASRec candidate scoring shape mismatch")
        if torch.any(lengths <= 0) or torch.any(lengths > sequences.shape[1]):
            raise ValueError("SASRec lengths outside sequence bounds")
        hidden = self.encode(sequences)
        rows = torch.arange(sequences.shape[0], device=sequences.device)
        final = hidden[rows, lengths - 1].unsqueeze(1)
        items = self.item_embedding(candidates)
        return (final * items).sum(dim=-1)


def right_padded_sequences(
    sequences: Sequence[Sequence[int]], maximum_sequence_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode zero-based item indices as one-based, right-padded SASRec inputs."""
    if maximum_sequence_length <= 0 or not sequences:
        raise ValueError("invalid sequence batch")
    clipped = [list(values[-maximum_sequence_length:]) for values in sequences]
    if any(not values for values in clipped):
        raise ValueError("SASRec inference requires non-empty histories")
    output = torch.zeros((len(clipped), maximum_sequence_length), dtype=torch.long)
    lengths = torch.tensor([len(values) for values in clipped], dtype=torch.long)
    for row, values in enumerate(clipped):
        output[row, : len(values)] = torch.tensor([value + 1 for value in values], dtype=torch.long)
    return output, lengths
