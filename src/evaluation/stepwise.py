"""Official-style row-by-row evaluation without sequence-boundary leakage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from competition_package.utils import DataPoint
from src.data.protocol import get_feature_columns, validate_wunder_dataframe
from src.utils.metrics import compute_r2_per_feature, compute_r2_score


@dataclass(frozen=True)
class StepwiseScoreResult:
    mean_r2: float
    r2_per_feature: dict[str, float]
    n_predictions_scored: int
    n_predictions_requested: int
    n_predictions_dropped_final_step: int
    feature_columns: list[str]


class StepwiseScorer:
    """Replay held-out sequences through the competition `predict` API."""

    def __init__(self, dataframe: pd.DataFrame):
        self.dataset = dataframe.sort_values(["seq_ix", "step_in_seq"]).reset_index(drop=True)
        self.feature_columns = validate_wunder_dataframe(self.dataset)
        self.dim = len(self.feature_columns)

    @classmethod
    def from_parquet(cls, path: str, seq_ids: Optional[list[int]] = None) -> "StepwiseScorer":
        df = pd.read_parquet(path)
        if seq_ids is not None:
            df = df[df["seq_ix"].isin(set(seq_ids))].copy()
        return cls(df)

    def score(self, model) -> StepwiseScoreResult:
        predictions = []
        targets = []
        pending_prediction = None
        pending_seq_ix = None
        requested = 0
        dropped_final = 0

        for _, seq_df in self.dataset.groupby("seq_ix", sort=True):
            pending_prediction = None
            pending_seq_ix = None

            seq_ix_values = seq_df["seq_ix"].to_numpy()
            step_values = seq_df["step_in_seq"].to_numpy()
            need_values = seq_df["need_prediction"].to_numpy()
            state_values = seq_df[self.feature_columns].to_numpy(dtype=np.float32)

            for pos in range(len(seq_df)):
                seq_ix = int(seq_ix_values[pos])
                step_in_seq = int(step_values[pos])
                need_prediction = bool(need_values[pos])
                state = state_values[pos]

                if pending_prediction is not None:
                    if pending_seq_ix != seq_ix:
                        raise AssertionError("Internal scorer leaked across sequences")
                    predictions.append(pending_prediction)
                    targets.append(state)

                data_point = DataPoint(seq_ix, step_in_seq, need_prediction, state)
                next_prediction = model.predict(data_point)
                self._check_prediction(data_point, next_prediction)
                if need_prediction:
                    requested += 1
                pending_prediction = next_prediction
                pending_seq_ix = seq_ix

            if pending_prediction is not None:
                dropped_final += 1

        if not predictions:
            raise ValueError("No predictions were scored")

        y_pred = np.asarray(predictions, dtype=np.float64)
        y_true = np.asarray(targets, dtype=np.float64)
        return StepwiseScoreResult(
            mean_r2=compute_r2_score(y_true, y_pred),
            r2_per_feature=compute_r2_per_feature(y_true, y_pred, self.feature_columns),
            n_predictions_scored=len(predictions),
            n_predictions_requested=requested,
            n_predictions_dropped_final_step=dropped_final,
            feature_columns=self.feature_columns,
        )

    def _check_prediction(self, data_point: DataPoint, prediction: Optional[np.ndarray]) -> None:
        if not data_point.need_prediction:
            if prediction is not None:
                raise ValueError(f"Prediction is not needed for {data_point}")
            return
        if prediction is None:
            raise ValueError(f"Prediction is required for {data_point}")
        if not isinstance(prediction, np.ndarray):
            raise ValueError(f"Prediction must be np.ndarray, got {type(prediction)}")
        if prediction.shape != (self.dim,):
            raise ValueError(f"Prediction has wrong shape: {prediction.shape} != {(self.dim,)}")
        if not np.isfinite(prediction).all():
            raise ValueError("Prediction contains NaN or Inf")
