"""Online stateful inference wrapper for the causal GRU.

Carries the GRU hidden state across steps so each prediction is O(1) work,
which is what makes the model fit the 1-core / 60-minute submission budget.
Used by both local stepwise evaluation and the packaged solution.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.models.sequence_models import CausalGRUForecaster


class GRUStatefulPredictionModel:
    """Wrap a CausalGRUForecaster as a competition PredictionModel."""

    def __init__(self, model: CausalGRUForecaster, n_features: int = 32):
        self.model = model.eval()
        self.n_features = n_features
        self.current_seq: Optional[int] = None
        self.h: Optional[torch.Tensor] = None

    def reset(self):
        self.current_seq = None
        self.h = None

    @torch.no_grad()
    def predict(self, data_point):
        if self.current_seq != data_point.seq_ix:
            self.current_seq = data_point.seq_ix
            self.h = None
        state = torch.from_numpy(np.asarray(data_point.state, dtype=np.float32))
        pred, self.h = self.model.step(state, self.h)
        if not data_point.need_prediction:
            return None
        return pred.detach().numpy().astype(np.float32)


def build_model_from_config(params: dict, model_type: str = "CausalGRUForecaster"):
    if model_type == "CausalTCN":
        from src.models.tcn import CausalTCN
        return CausalTCN(
            n_features=params.get("n_features", 32),
            d_model=params.get("d_model", 192),
            n_layers=params.get("n_layers", 6),
            kernel_size=params.get("kernel_size", 3),
            dropout=params.get("dropout", 0.1),
            head_hidden=params.get("head_hidden"),
        )
    return CausalGRUForecaster(
        n_features=params.get("n_features", 32),
        d_model=params.get("d_model", 256),
        n_layers=params.get("n_layers", 2),
        dropout=params.get("dropout", 0.1),
        head_hidden=params.get("head_hidden"),
        rnn_type=params.get("rnn_type", "gru"),
    )


def load_gru_checkpoint(path: str | Path, map_location: str = "cpu"):
    """Reconstruct a forecaster (GRU/LSTM/TCN) from a saved checkpoint."""
    ckpt = torch.load(str(path), map_location=map_location, weights_only=False)
    model_cfg = ckpt.get("config", {}).get("model", {})
    params = model_cfg.get("params", {})
    model = build_model_from_config(params, model_cfg.get("type", "CausalGRUForecaster"))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model
