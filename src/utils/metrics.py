"""R2 metrics for the Wunder Fund RNN Challenge.

The authoritative local metric mirrors ``competition_package.utils``:
``sklearn.metrics.r2_score`` is computed independently per feature and then
averaged.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score


def _validate_shapes(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    if y_true.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got y_true.ndim={y_true.ndim}")


def compute_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean per-feature R2 using sklearn, matching the starter pack."""
    return float(np.mean(list(compute_r2_per_feature(y_true, y_pred).values())))


def compute_r2_per_feature(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute sklearn R2 independently for each output feature."""
    _validate_shapes(y_true, y_pred)
    n_features = y_true.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    if len(feature_names) != n_features:
        raise ValueError("feature_names length must match prediction width")

    return {
        feature_names[i]: float(r2_score(y_true[:, i], y_pred[:, i]))
        for i in range(n_features)
    }


def compute_manual_r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Legacy epsilon-stabilized R2 helper for diagnostics only."""
    _validate_shapes(y_true, y_pred)
    scores = []
    for i in range(y_true.shape[1]):
        ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
        ss_tot = np.sum((y_true[:, i] - y_true[:, i].mean()) ** 2)
        scores.append(1 - (ss_res / (ss_tot + epsilon)))
    return float(np.mean(scores))
