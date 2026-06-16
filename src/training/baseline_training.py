"""Training utilities for Phase 2 causal tabular baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.protocol import get_feature_columns, load_wunder_dataframe
from src.models.baselines import build_causal_tabular_features


def build_tabular_supervised_matrix(
    parquet_path: str,
    seq_ids: list[int],
    *,
    feature_schema: str = "compact_v1",
) -> tuple[np.ndarray, np.ndarray]:
    """Build causal tabular features and next-state targets for selected sequences."""
    df = load_wunder_dataframe(parquet_path, seq_ids=seq_ids)
    feature_cols = get_feature_columns(df)
    rows = []
    targets = []
    for _, seq_df in df.groupby("seq_ix", sort=True):
        states = seq_df[feature_cols].to_numpy(dtype=np.float32)
        need_prediction = seq_df["need_prediction"].to_numpy(dtype=bool)
        for pos in range(len(states) - 1):
            if not need_prediction[pos]:
                continue
            rows.append(build_causal_tabular_features(states[: pos + 1], schema=feature_schema))
            targets.append(states[pos + 1])
    if not rows:
        raise ValueError("No supervised rows were created")
    return np.vstack(rows).astype(np.float32), np.vstack(targets).astype(np.float32)


def create_tabular_estimator(model_type: str, seed: int = 42, n_jobs: int = 1):
    """Create a deterministic sklearn-compatible multi-output regressor."""
    model_type = model_type.lower()
    if model_type == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=1.0, random_state=seed)
    if model_type == "ridge_cv":
        from sklearn.linear_model import RidgeCV
        from sklearn.multioutput import MultiOutputRegressor

        base = RidgeCV(alphas=np.logspace(-4, 4, 17))
        return MultiOutputRegressor(base, n_jobs=max(1, int(n_jobs)))
    if model_type == "elasticnet":
        from sklearn.linear_model import MultiTaskElasticNet

        return MultiTaskElasticNet(alpha=0.0005, l1_ratio=0.05, random_state=seed, max_iter=5000)
    if model_type == "lightgbm":
        import lightgbm as lgb
        from sklearn.multioutput import MultiOutputRegressor

        base = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
        return MultiOutputRegressor(base, n_jobs=max(1, int(n_jobs)))
    raise ValueError(f"Unknown tabular model type: {model_type}")
