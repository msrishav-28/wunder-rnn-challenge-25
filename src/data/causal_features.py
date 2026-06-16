"""Causal online feature builders shared by Phase 2 tabular baselines."""

from __future__ import annotations

import hashlib
import json

import numpy as np


COMPACT_V1_WINDOWS = (3, 5, 10, 20, 50, 100)
EXPANDED_V2_LAGS = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610)
EXPANDED_V2_EWMA_ALPHAS = (0.02, 0.05, 0.10, 0.20, 0.40)
EXPANDED_V2_WINDOWS = (3, 5, 10, 20, 50, 100, 200, 500)


def _validate_history(history: np.ndarray) -> np.ndarray:
    history = np.asarray(history, dtype=np.float32)
    if history.ndim != 2 or history.shape[1] != 32:
        raise ValueError(f"Expected history shape (t, 32), got {history.shape}")
    if len(history) == 0:
        raise ValueError("history must contain at least one state")
    return history


def _ewma(history: np.ndarray, alpha: float) -> np.ndarray:
    state = history[0].astype(np.float32, copy=True)
    for row in history[1:]:
        state = alpha * row + (1.0 - alpha) * state
    return state.astype(np.float32)


def build_compact_v1_features(history: np.ndarray) -> np.ndarray:
    """Build the original compact raw-32 causal feature vector."""
    history = _validate_history(history)
    current = history[-1]
    prev = history[-2] if len(history) >= 2 else np.zeros_like(current)
    delta1 = current - prev

    features = [current, delta1]
    for window in COMPACT_V1_WINDOWS:
        chunk = history[-window:]
        mean = chunk.mean(axis=0)
        features.extend([mean, chunk.std(axis=0), current - mean])
    return np.concatenate(features).astype(np.float32)


def build_expanded_v2_features(history: np.ndarray) -> np.ndarray:
    """Build the Phase 2 max-safe expanded causal feature vector."""
    history = _validate_history(history)
    current = history[-1]
    features = [current]

    for lag in EXPANDED_V2_LAGS:
        lagged = history[-1 - lag] if len(history) > lag else history[0]
        features.extend([lagged, current - lagged])

    for alpha in EXPANDED_V2_EWMA_ALPHAS:
        smoothed = _ewma(history, alpha)
        features.extend([smoothed, current - smoothed])

    for window in EXPANDED_V2_WINDOWS:
        chunk = history[-window:]
        mean = chunk.mean(axis=0)
        features.extend(
            [
                mean,
                chunk.std(axis=0),
                chunk.min(axis=0),
                chunk.max(axis=0),
                current - mean,
            ]
        )
    return np.concatenate(features).astype(np.float32)


def build_causal_tabular_features(history: np.ndarray, schema: str = "compact_v1") -> np.ndarray:
    if schema == "compact_v1":
        return build_compact_v1_features(history)
    if schema == "expanded_v2":
        return build_expanded_v2_features(history)
    raise ValueError(f"Unknown feature schema: {schema}")


def feature_width(schema: str) -> int:
    return int(build_causal_tabular_features(np.zeros((1, 32), dtype=np.float32), schema).shape[0])


def infer_feature_schema_from_width(width: int) -> str:
    width = int(width)
    for schema in ("compact_v1", "expanded_v2"):
        if feature_width(schema) == width:
            return schema
    raise ValueError(f"No known causal feature schema has width {width}")


def feature_schema_payload(schema: str) -> dict:
    if schema == "compact_v1":
        return {
            "schema": schema,
            "windows": list(COMPACT_V1_WINDOWS),
            "width": feature_width(schema),
        }
    if schema == "expanded_v2":
        return {
            "schema": schema,
            "lags": list(EXPANDED_V2_LAGS),
            "ewma_alphas": list(EXPANDED_V2_EWMA_ALPHAS),
            "windows": list(EXPANDED_V2_WINDOWS),
            "width": feature_width(schema),
        }
    raise ValueError(f"Unknown feature schema: {schema}")


def feature_schema_hash(schema: str) -> str:
    payload = json.dumps(feature_schema_payload(schema), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
