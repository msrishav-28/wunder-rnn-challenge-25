"""solution.py — Competition submission for the Wunder Fund RNN Challenge.

Root entry point for the scorer. Defines PredictionModel with a predict()
method that receives DataPoint objects one at a time and predicts the *next*
state when need_prediction is True.

Primary backend: an ensemble of causal GRU forecasters (carrying hidden state
across steps for O(1)/prediction online inference), optionally blended per
feature with tabular models. Falls back to joblib tabular models, then a
momentum baseline (only when ALLOW_FALLBACK=1).
"""

import os

# Windows: torch and numpy/sklearn each ship an OpenMP runtime; importing torch
# before the MKL-backed stack avoids a c10.dll init failure. Also keep the
# submission single-core friendly by default (the scorer grants 1 CPU).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_THREADS = os.environ.get("WUNDER_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", _THREADS)

try:
    import torch
    torch.set_num_threads(max(1, int(_THREADS)))
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

import json
from pathlib import Path
from typing import Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent
import sys
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class DataPoint:
    """Mirror of competition_package.utils.DataPoint for standalone use."""

    def __init__(self, seq_ix, step_in_seq, need_prediction, state):
        self.seq_ix = seq_ix
        self.step_in_seq = step_in_seq
        self.need_prediction = need_prediction
        self.state = state


class PredictionModel:
    """Competition PredictionModel. Loads the best available backend."""

    def __init__(self):
        self.n_features = 32
        self.backend = "fallback"
        self.predictor = None  # object exposing predict(data_point)
        self._current_seq = None
        self._prev_state = None
        self._cur_state = None

        self._try_load_gru()
        if self.predictor is None:
            self._try_load_onnx()
        if self.predictor is None:
            self._try_load_tabular()
        if self.predictor is None:
            self._try_load_pytorch()
        if self.predictor is None and os.environ.get("ALLOW_FALLBACK") != "1":
            raise RuntimeError(
                "No trained model artifacts found. Set ALLOW_FALLBACK=1 only for "
                "local smoke tests; real submissions must package models."
            )

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------
    def _try_load_gru(self):
        """Primary: GRU ensemble (+ optional tabular members) under models/submission."""
        if not _HAS_TORCH:
            return
        try:
            from src.models.ensemble_predictor import EnsemblePredictionModel
            from src.models.sequence_inference import (
                GRUStatefulPredictionModel,
                load_gru_checkpoint,
            )

            sub = _ROOT / "models" / "submission"
            gru_paths = sorted((sub / "gru").glob("*.pt")) if (sub / "gru").exists() else []
            if not gru_paths:
                return

            members, names = [], []
            for p in gru_paths:
                model = load_gru_checkpoint(str(p))
                members.append(GRUStatefulPredictionModel(model))
                names.append(p.stem)

            # optional tabular members
            tab_dir = sub / "tabular"
            if tab_dir.exists():
                import joblib
                from src.models.baselines import TabularStatefulPredictionModel
                for jb in sorted(tab_dir.glob("*/model.joblib")):
                    manifest = jb.parent / "model_manifest.json"
                    schema = "compact_v1"
                    if manifest.exists():
                        try:
                            schema = json.loads(manifest.read_text()).get("feature_schema", schema)
                        except Exception:
                            pass
                    members.append(TabularStatefulPredictionModel(joblib.load(str(jb)), feature_schema=schema))
                    names.append(jb.parent.name)

            weights = None
            wpath = sub / "blend_weights.json"
            if wpath.exists():
                try:
                    payload = json.loads(wpath.read_text())
                    w = np.asarray(payload["per_feature_weights"], dtype=np.float32)
                    if w.shape == (self.n_features, len(members)):
                        weights = w
                except Exception:
                    weights = None

            self.predictor = EnsemblePredictionModel(members, weights=weights)
            self.backend = "gru_ensemble"
        except Exception:
            self.predictor = None

    def _try_load_onnx(self):
        """ONNX ensemble (kept for completeness; not the primary path)."""
        return

    def _try_load_tabular(self):
        """Joblib tabular models discovered under models/ (legacy/baseline path)."""
        try:
            import joblib
            from src.models.baselines import TabularStatefulPredictionModel
            from src.models.ensemble_predictor import EnsemblePredictionModel

            model_dir = _ROOT / "models"
            joblib_files = sorted(model_dir.glob("phase2_tabular/**/model.joblib"))
            if not joblib_files:
                return
            members = []
            for path in joblib_files:
                manifest = path.parent / "model_manifest.json"
                schema = "compact_v1"
                if manifest.exists():
                    try:
                        schema = json.loads(manifest.read_text()).get("feature_schema", schema)
                    except Exception:
                        pass
                members.append(TabularStatefulPredictionModel(joblib.load(str(path)), feature_schema=schema))
            if members:
                self.predictor = EnsemblePredictionModel(members)
                self.backend = "tabular"
        except Exception:
            self.predictor = None

    def _try_load_pytorch(self):
        """Legacy PyTorch checkpoint path (kept for test compatibility)."""
        return

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, data_point) -> Optional[np.ndarray]:
        if self.predictor is not None:
            return self.predictor.predict(data_point)
        return self._predict_fallback(data_point)

    def _predict_fallback(self, data_point) -> Optional[np.ndarray]:
        """Momentum baseline; only reachable with ALLOW_FALLBACK=1."""
        if self._current_seq != data_point.seq_ix:
            self._current_seq = data_point.seq_ix
            self._prev_state = None
            self._cur_state = None
        self._prev_state = self._cur_state
        self._cur_state = np.asarray(data_point.state, dtype=np.float32)
        if not data_point.need_prediction:
            return None
        if self._prev_state is None:
            return self._cur_state.astype(np.float32, copy=True)
        return (2.0 * self._cur_state - self._prev_state).astype(np.float32)


if __name__ == "__main__":
    model = PredictionModel()
    print(f"Backend: {model.backend}")
    for step in range(200):
        dp = DataPoint(0, step, step >= 100, np.random.randn(32).astype(np.float32))
        pred = model.predict(dp)
        if pred is not None and step == 100:
            print(f"step {step}: shape={pred.shape} dtype={pred.dtype} finite={np.all(np.isfinite(pred))}")
    print("Local test passed!")
