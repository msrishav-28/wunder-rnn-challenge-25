"""
Unit tests for R² metric computation.
CRITICAL: These must match the competition metric exactly.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.metrics import compute_r2_score, compute_r2_per_feature


class TestR2Score:
    """Test R² score computation."""

    def test_perfect_prediction(self):
        y_true = np.random.randn(100, 32)
        y_pred = y_true.copy()
        r2 = compute_r2_score(y_true, y_pred)
        assert abs(r2 - 1.0) < 1e-6, f"Expected R²=1.0, got {r2}"

    def test_mean_prediction(self):
        np.random.seed(42)
        y_true = np.random.randn(1000, 32)
        y_pred = np.tile(y_true.mean(axis=0), (1000, 1))
        r2 = compute_r2_score(y_true, y_pred)
        assert abs(r2) < 0.01, f"Expected R²≈0.0, got {r2}"

    def test_zero_prediction(self):
        y_true = np.random.randn(100, 32) + 5.0
        y_pred = np.zeros((100, 32))
        r2 = compute_r2_score(y_true, y_pred)
        assert r2 < 0, f"Expected negative R², got {r2}"

    def test_constant_target(self):
        y_true = np.ones((100, 32)) * 5.0
        y_pred = np.ones((100, 32)) * 3.0
        r2 = compute_r2_score(y_true, y_pred)
        assert np.isfinite(r2), "R² should be finite with constant target"

    def test_single_sample(self):
        y_true = np.random.randn(1, 32)
        y_pred = np.random.randn(1, 32)
        r2 = compute_r2_score(y_true, y_pred)
        assert np.isnan(r2)


class TestR2PerFeature:

    def test_returns_dict(self):
        y_true = np.random.randn(100, 32)
        y_pred = np.random.randn(100, 32)
        r2_dict = compute_r2_per_feature(y_true, y_pred)
        assert isinstance(r2_dict, dict)
        assert len(r2_dict) == 32
        assert all(f'feature_{i}' in r2_dict for i in range(32))

    def test_all_finite(self):
        y_true = np.random.randn(100, 32)
        y_pred = np.random.randn(100, 32)
        r2_dict = compute_r2_per_feature(y_true, y_pred)
        for feat, r2 in r2_dict.items():
            assert np.isfinite(r2), f"{feat} has non-finite R²"

    def test_matches_overall(self):
        np.random.seed(42)
        y_true = np.random.randn(100, 32)
        y_pred = y_true + np.random.randn(100, 32) * 0.1
        overall_r2 = compute_r2_score(y_true, y_pred)
        r2_dict = compute_r2_per_feature(y_true, y_pred)
        mean_r2 = np.mean(list(r2_dict.values()))
        assert abs(overall_r2 - mean_r2) < 1e-6
