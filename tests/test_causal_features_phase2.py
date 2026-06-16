import numpy as np

from src.data.causal_features import (
    build_causal_tabular_features,
    feature_schema_hash,
    feature_width,
    infer_feature_schema_from_width,
)


def test_causal_feature_widths_and_hashes_are_stable():
    assert feature_width("compact_v1") == 640
    assert feature_width("expanded_v2") == 2528
    assert infer_feature_schema_from_width(640) == "compact_v1"
    assert infer_feature_schema_from_width(2528) == "expanded_v2"
    assert len(feature_schema_hash("compact_v1")) == 64
    assert len(feature_schema_hash("expanded_v2")) == 64


def test_tabular_inference_features_match_training_builder():
    """The tabular inference path must build the exact same causal features as
    training. Both now call src.data.causal_features.build_causal_tabular_features
    (single source of truth); this verifies the actual inference wrapper too."""
    from src.models.baselines import TabularStatefulPredictionModel

    rng = np.random.RandomState(42)
    history = rng.randn(137, 32).astype(np.float32)

    class _RecordingEstimator:
        def __init__(self):
            self.last_X = None

        def predict(self, X):
            self.last_X = np.asarray(X)
            return np.zeros((X.shape[0], 32), dtype=np.float32)

    for schema in ("compact_v1", "expanded_v2"):
        expected = build_causal_tabular_features(history, schema=schema)
        est = _RecordingEstimator()
        model = TabularStatefulPredictionModel(est, feature_schema=schema)
        for i, row in enumerate(history):
            dp = type("DP", (), {
                "seq_ix": 0, "step_in_seq": i,
                "need_prediction": (i == len(history) - 1), "state": row,
            })()
            model.predict(dp)
        actual = est.last_X[0].astype(np.float32)
        assert np.allclose(actual, expected)
