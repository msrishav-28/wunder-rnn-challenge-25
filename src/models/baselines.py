"""Causal baseline predictors for the Wunder stepwise API."""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.data.causal_features import build_causal_tabular_features


class PersistencePredictionModel:
    """Predict the current state as the next state."""

    def __init__(self, n_features: int = 32):
        self.n_features = n_features
        self.current_seq: Optional[int] = None
        self.last_state: Optional[np.ndarray] = None

    def predict(self, data_point):
        if self.current_seq != data_point.seq_ix:
            self.current_seq = data_point.seq_ix
            self.last_state = None
        self.last_state = data_point.state.astype(np.float32, copy=True)
        if not data_point.need_prediction:
            return None
        return self.last_state.astype(np.float32, copy=True)


class MomentumPredictionModel:
    """Predict next state with a first-difference momentum term."""

    def __init__(self, alpha: float = 1.0, n_features: int = 32):
        self.alpha = float(alpha)
        self.n_features = n_features
        self.current_seq: Optional[int] = None
        self.previous_state: Optional[np.ndarray] = None
        self.current_state: Optional[np.ndarray] = None

    def predict(self, data_point):
        if self.current_seq != data_point.seq_ix:
            self.current_seq = data_point.seq_ix
            self.previous_state = None
            self.current_state = None

        self.previous_state = self.current_state
        self.current_state = data_point.state.astype(np.float32, copy=True)
        if not data_point.need_prediction:
            return None
        if self.previous_state is None:
            return self.current_state.astype(np.float32, copy=True)
        pred = self.current_state + self.alpha * (self.current_state - self.previous_state)
        return pred.astype(np.float32)


class EWMAPredictionModel:
    """Predict with an exponentially weighted moving average state."""

    def __init__(self, alpha: float = 0.2, n_features: int = 32):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.n_features = n_features
        self.current_seq: Optional[int] = None
        self.ewma_state: Optional[np.ndarray] = None

    def predict(self, data_point):
        if self.current_seq != data_point.seq_ix:
            self.current_seq = data_point.seq_ix
            self.ewma_state = None
        state = data_point.state.astype(np.float32, copy=False)
        if self.ewma_state is None:
            self.ewma_state = state.copy()
        else:
            self.ewma_state = self.alpha * state + (1.0 - self.alpha) * self.ewma_state
        if not data_point.need_prediction:
            return None
        return self.ewma_state.astype(np.float32, copy=True)


class TabularStatefulPredictionModel:
    """Stateful wrapper for sklearn-style multi-output tabular regressors."""

    def __init__(self, estimator, n_features: int = 32, feature_schema: str = "compact_v1"):
        self.estimator = estimator
        self.n_features = n_features
        self.feature_schema = feature_schema
        self.current_seq: Optional[int] = None
        self.history: list[np.ndarray] = []

    def predict(self, data_point):
        if self.current_seq != data_point.seq_ix:
            self.current_seq = data_point.seq_ix
            self.history = []
        self.history.append(data_point.state.astype(np.float32, copy=True))
        if not data_point.need_prediction:
            return None
        features = build_causal_tabular_features(
            np.asarray(self.history, dtype=np.float32),
            schema=self.feature_schema,
        )
        pred = self.estimator.predict(features.reshape(1, -1))[0]
        return np.asarray(pred, dtype=np.float32)
