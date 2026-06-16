import numpy as np
import pandas as pd

from src.evaluation.stepwise import StepwiseScorer


class NextStateOracle:
    def __init__(self, dataframe):
        self.lookup = {}
        feature_cols = list(dataframe.columns[3:])
        for _, seq_df in dataframe.groupby("seq_ix"):
            seq_df = seq_df.sort_values("step_in_seq")
            states = seq_df[feature_cols].to_numpy(dtype=np.float32)
            steps = seq_df["step_in_seq"].to_numpy()
            seq_ix = int(seq_df["seq_ix"].iloc[0])
            for pos, step in enumerate(steps[:-1]):
                self.lookup[(seq_ix, int(step))] = states[pos + 1]

    def predict(self, data_point):
        if not data_point.need_prediction:
            return None
        return self.lookup.get(
            (int(data_point.seq_ix), int(data_point.step_in_seq)),
            data_point.state,
        ).astype(np.float32)


class AlwaysPredictCurrent:
    def predict(self, data_point):
        if not data_point.need_prediction:
            return None
        return data_point.state.astype(np.float32)


def test_stepwise_scorer_scores_next_state_without_cross_sequence_leak(sample_dataframe):
    scorer = StepwiseScorer(sample_dataframe)
    result = scorer.score(NextStateOracle(sample_dataframe))
    assert result.mean_r2 == 1.0
    assert result.n_predictions_scored == 5 * 899
    assert result.n_predictions_requested == 5 * 900
    assert result.n_predictions_dropped_final_step == 5


def test_stepwise_scorer_rejects_unneeded_predictions(sample_dataframe):
    class BadModel:
        def predict(self, data_point):
            return np.zeros(32, dtype=np.float32)

    scorer = StepwiseScorer(sample_dataframe)
    try:
        scorer.score(BadModel())
    except ValueError as exc:
        assert "not needed" in str(exc)
    else:
        raise AssertionError("Expected scorer to reject warm-up predictions")
