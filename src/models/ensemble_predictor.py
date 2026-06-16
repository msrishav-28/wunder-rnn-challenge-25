"""Uniform ensemble PredictionModel for the Wunder stepwise task.

Both GRUStatefulPredictionModel and TabularStatefulPredictionModel expose the
same contract: ``predict(data_point)`` advances per-sequence internal state and
returns None during warm-up or an (n_features,) array when a prediction is
needed. An ensemble is therefore just a list of member predictors plus an
optional per-feature weight matrix. The same class powers local holdout
evaluation and the packaged solution.py.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class EnsemblePredictionModel:
    def __init__(self, members: list, weights: Optional[np.ndarray] = None, n_features: int = 32):
        """Args:
            members: objects exposing predict(data_point) -> Optional[np.ndarray].
            weights: optional (n_features, n_members) blend weights. If None, the
                     members are averaged equally.
        """
        if not members:
            raise ValueError("EnsemblePredictionModel needs at least one member")
        self.members = members
        self.n_features = n_features
        self.weights = None
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float32)
            if weights.shape != (n_features, len(members)):
                raise ValueError(
                    f"weights shape {weights.shape} != {(n_features, len(members))}"
                )
            self.weights = weights

    def predict(self, data_point) -> Optional[np.ndarray]:
        # Always advance every member's state, even during warm-up.
        member_preds = [m.predict(data_point) for m in self.members]
        if not data_point.need_prediction:
            return None
        stacked = np.stack([np.asarray(p, dtype=np.float32) for p in member_preds], axis=0)  # (M, F)
        if self.weights is not None:
            # out[f] = sum_m stacked[m, f] * weights[f, m]
            out = np.einsum("mf,fm->f", stacked, self.weights)
        else:
            out = stacked.mean(axis=0)
        return out.astype(np.float32)
