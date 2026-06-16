"""
Tests for competition submission format validation.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPredictionModelInterface:
    """Validate that solution.py PredictionModel meets competition API."""

    def test_predict_returns_none_when_not_needed(self, monkeypatch):
        """predict() should return None when need_prediction=False."""
        from competition_package.utils import DataPoint

        # Import lazily to avoid hard dependency on ONNX models
        # We test the interface contract, not the actual model loading
        from solution import PredictionModel

        monkeypatch.setenv("ALLOW_FALLBACK", "1")
        model = PredictionModel()

        dp = DataPoint(
            seq_ix=0,
            step_in_seq=0,
            need_prediction=False,
            state=np.random.randn(32).astype(np.float32),
        )
        result = model.predict(dp)
        assert result is None

    def test_predict_returns_array_when_needed(self, monkeypatch):
        """predict() should return ndarray(32,) when need_prediction=True."""
        from competition_package.utils import DataPoint
        from solution import PredictionModel

        monkeypatch.setenv("ALLOW_FALLBACK", "1")
        model = PredictionModel()

        # Feed warm-up steps
        for step in range(100):
            dp = DataPoint(
                seq_ix=0,
                step_in_seq=step,
                need_prediction=False,
                state=np.random.randn(32).astype(np.float32),
            )
            model.predict(dp)

        # Prediction step
        dp = DataPoint(
            seq_ix=0,
            step_in_seq=100,
            need_prediction=True,
            state=np.random.randn(32).astype(np.float32),
        )
        result = model.predict(dp)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (32,)
        assert result.dtype == np.float32

    def test_handles_new_sequence(self, monkeypatch):
        """Model should reset state for new sequence."""
        from competition_package.utils import DataPoint
        from solution import PredictionModel

        monkeypatch.setenv("ALLOW_FALLBACK", "1")
        model = PredictionModel()

        # Process some steps from sequence 0
        for step in range(10):
            dp = DataPoint(seq_ix=0, step_in_seq=step, need_prediction=False,
                          state=np.random.randn(32).astype(np.float32))
            model.predict(dp)

        # Switch to sequence 1
        dp = DataPoint(seq_ix=1, step_in_seq=0, need_prediction=False,
                      state=np.random.randn(32).astype(np.float32))
        result = model.predict(dp)
        assert result is None  # Not needed yet

    def test_prediction_values_finite(self, monkeypatch):
        """All prediction values should be finite (no NaN/Inf)."""
        from competition_package.utils import DataPoint
        from solution import PredictionModel

        monkeypatch.setenv("ALLOW_FALLBACK", "1")
        model = PredictionModel()

        for step in range(200):
            dp = DataPoint(
                seq_ix=0,
                step_in_seq=step,
                need_prediction=step >= 100,
                state=np.random.randn(32).astype(np.float32),
            )
            result = model.predict(dp)

            if result is not None:
                assert np.all(np.isfinite(result)), \
                    f"Non-finite values at step {step}: {result}"

    def test_missing_artifacts_fail_closed(self, monkeypatch):
        from solution import PredictionModel

        monkeypatch.delenv("ALLOW_FALLBACK", raising=False)
        monkeypatch.setattr(PredictionModel, "_try_load_gru", lambda self: None)
        monkeypatch.setattr(PredictionModel, "_try_load_onnx", lambda self: None)
        monkeypatch.setattr(PredictionModel, "_try_load_tabular", lambda self: None)
        monkeypatch.setattr(PredictionModel, "_try_load_pytorch", lambda self: None)
        try:
            PredictionModel()
        except RuntimeError as exc:
            assert "No trained model artifacts" in str(exc)
        else:
            raise AssertionError("PredictionModel should fail closed without artifacts")
