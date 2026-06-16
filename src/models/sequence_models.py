"""Causal sequence models for the Wunder stepwise next-state task.

Design constraints derived from the data analysis (see memory / report):
  - Features are pre-whitened (~N(0,1), no heavy tails) so no internal
    normalization is required; the model predicts the next *level* directly.
  - Inference is online, one row at a time, on a single CPU core within a
    60-minute budget. A unidirectional GRU carries hidden state across steps,
    giving O(1) work per prediction.
  - Training uses full 1000-step sequences with back-prop-through-time and a
    loss masked to the scored steps (100..998). Because a GRU is a pure
    recurrence, the full-sequence forward and the stepwise stateful forward
    produce identical outputs — so what we train is exactly what we score.

`forward(x, h0)` returns per-step predictions of the *next* state:
    preds[:, t] predicts x[:, t + 1].
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class CausalGRUForecaster(nn.Module):
    """Unidirectional GRU that predicts the next state at every step.

    Args:
        n_features: input/output width (32).
        d_model: GRU hidden size and embedding width.
        n_layers: stacked GRU layers.
        dropout: dropout between GRU layers and inside the head.
        head_hidden: width of the MLP head hidden layer (defaults to d_model).
    """

    def __init__(
        self,
        n_features: int = 32,
        d_model: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        head_hidden: Optional[int] = None,
        rnn_type: str = "gru",
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.rnn_type = str(rnn_type).lower()
        head_hidden = int(head_hidden) if head_hidden else self.d_model

        # Per-step input embedding (applied identically in batch and stepwise modes).
        self.input_proj = nn.Linear(n_features, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.gru = rnn_cls(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_features),
        )

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
            x: (batch, seq_len, n_features)
            h0: optional initial hidden state (n_layers, batch, d_model)
        Returns:
            preds: (batch, seq_len, n_features) where preds[:, t] predicts x[:, t+1]
            h_n:   final hidden state (n_layers, batch, d_model)
        """
        z = self.input_norm(self.input_proj(x))
        out, h_n = self.gru(z, h0)
        preds = self.head(out)
        return preds, h_n

    @torch.no_grad()
    def step(
        self,
        state: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single online step.

        Args:
            state: (n_features,) or (1, n_features) current state.
            h: previous hidden state (n_layers, 1, d_model) or None.
        Returns:
            pred: (n_features,) prediction of the next state.
            h_new: updated hidden state.
        """
        if state.dim() == 1:
            state = state.view(1, 1, -1)
        elif state.dim() == 2:
            state = state.unsqueeze(1)
        preds, h_new = self.forward(state, h)
        return preds[0, -1], h_new


def build_gru(config: dict) -> CausalGRUForecaster:
    """Construct a CausalGRUForecaster from a plain dict config."""
    return CausalGRUForecaster(
        n_features=config.get("n_features", 32),
        d_model=config.get("d_model", 256),
        n_layers=config.get("n_layers", 2),
        dropout=config.get("dropout", 0.1),
        head_hidden=config.get("head_hidden"),
        rnn_type=config.get("rnn_type", "gru"),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
