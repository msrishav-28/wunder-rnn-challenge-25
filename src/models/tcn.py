"""Causal Temporal Convolutional Network for next-state prediction.

A different inductive bias from the RNNs (dilated causal convolutions instead of
a recurrence), which decorrelates its errors from the GRU/LSTM members and so
adds more in an ensemble than another RNN seed.

Exposes the same interface as CausalGRUForecaster so it is a drop-in for the
trainer, the stateful inference wrapper, and the ensembler:
  forward(x, h0=None) -> (preds, None)   where preds[:, t] predicts x[:, t+1]
  step(state, h)      -> (pred, buffer)  buffer holds the last receptive-field states
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class _ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dim of a (B, C, T) tensor, independent per
    timestep — causal, unlike GroupNorm/BatchNorm which mix statistics across T."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):  # (B, C, T)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class _CausalConvBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation, padding=self.pad)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation, padding=self.pad)
        self.norm1 = _ChannelLayerNorm(d_model)
        self.norm2 = _ChannelLayerNorm(d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # x: (B, d, T)
        T = x.shape[-1]
        h = self.conv1(x)[..., :T]      # trim right (future) padding -> causal
        h = self.drop(self.act(self.norm1(h)))
        h = self.conv2(h)[..., :T]
        h = self.drop(self.act(self.norm2(h)))
        return x + h


class CausalTCN(nn.Module):
    def __init__(
        self,
        n_features: int = 32,
        d_model: int = 192,
        n_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.1,
        head_hidden: Optional[int] = None,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.kernel_size = int(kernel_size)
        head_hidden = int(head_hidden) if head_hidden else d_model

        self.input_proj = nn.Linear(n_features, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([
            _CausalConvBlock(d_model, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_features),
        )
        # receptive field for buffered inference
        self.receptive_field = 1 + 2 * sum((kernel_size - 1) * (2 ** i) for i in range(n_layers))

    def forward(self, x: torch.Tensor, h0=None) -> Tuple[torch.Tensor, None]:
        z = self.input_norm(self.input_proj(x))  # (B, T, d)
        z = z.transpose(1, 2)                     # (B, d, T)
        for block in self.blocks:
            z = block(z)
        z = z.transpose(1, 2)                     # (B, T, d)
        return self.head(z), None

    @torch.no_grad()
    def step(self, state: torch.Tensor, h: Optional[torch.Tensor] = None):
        """h is the running buffer of recent states (1, L, F)."""
        if state.dim() == 1:
            state = state.view(1, 1, -1)
        elif state.dim() == 2:
            state = state.unsqueeze(1)
        buf = state if h is None else torch.cat([h, state], dim=1)
        buf = buf[:, -self.receptive_field:]
        preds, _ = self.forward(buf)
        return preds[0, -1], buf


def build_tcn(config: dict) -> CausalTCN:
    return CausalTCN(
        n_features=config.get("n_features", 32),
        d_model=config.get("d_model", 192),
        n_layers=config.get("n_layers", 6),
        kernel_size=config.get("kernel_size", 3),
        dropout=config.get("dropout", 0.1),
        head_hidden=config.get("head_hidden"),
    )
