"""Transformer Encoder model for SDC test case scoring.

The model processes a variable-length sequence of road segment features through
a Transformer Encoder, pools the encoded representations, fuses them with
global road statistics via an MLP, and produces a scalar priority score.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_pe(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Sinusoidal positional encoding computed on-the-fly (Vaswani et al., 2017)."""
    position = torch.arange(seq_len, device=device).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TransformerEncoder(nn.Module):
    """Pointwise Transformer Encoder for SDC test case prioritization.

    Args:
        cfg: Configuration dict with keys under "model":
            d_seq, d_global, d_model, n_heads, n_layers, dim_feedforward, dropout,
            pooling, d_global_hidden, d_head_hidden
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        d_seq = int(m["d_seq"])
        d_global = int(m["d_global"])
        d_model = int(m["d_model"])
        n_heads = int(m["n_heads"])
        n_layers = int(m["n_layers"])
        dim_feedforward = int(m["dim_feedforward"])
        dropout = float(m["dropout"])
        self.pooling = str(m.get("pooling", "mean"))
        d_global_hidden = int(m.get("d_global_hidden", 32))
        d_head_hidden = int(m.get("d_head_hidden", 64))

        self.d_model = d_model
        self.seq_proj = nn.Linear(d_seq, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # MLP for global road-level features
        self.global_mlp = nn.Sequential(
            nn.Linear(d_global, d_global_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_global_hidden, d_global_hidden),
            nn.GELU(),
        )

        # Scoring head: fuses encoder output with global features
        self.head = nn.Sequential(
            nn.Linear(d_model + d_global_hidden, d_head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_head_hidden, 1),
        )

    def pool(self, enc: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Pool encoded sequence into a single vector, ignoring padding tokens."""
        valid = valid_mask.unsqueeze(-1).to(enc.dtype)
        if self.pooling == "max":
            enc = enc.masked_fill(~valid_mask.unsqueeze(-1), float("-inf"))
            return enc.max(dim=1).values
        # Default: masked mean pooling
        return (enc * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    def forward(self, seq: torch.Tensor, valid_mask: torch.Tensor, glob: torch.Tensor) -> torch.Tensor:
        """Compute priority score for each test case in the batch.

        Args:
            seq: [B, T, d_seq] segment feature tokens
            valid_mask: [B, T] boolean mask (True = real token, False = padding)
            glob: [B, d_global] global road statistics

        Returns:
            scores: [B] scalar priority scores (higher = more likely to fail)
        """
        b, t, _ = seq.shape
        x = self.seq_proj(seq)
        x = x + sinusoidal_pe(t, self.d_model, seq.device).unsqueeze(0)

        pad_mask = ~valid_mask
        enc = self.encoder(x, src_key_padding_mask=pad_mask)

        pooled = self.pool(enc, valid_mask)
        h_global = self.global_mlp(glob)
        fused = torch.cat([pooled, h_global], dim=1)
        score = self.head(fused).squeeze(-1)
        return score
