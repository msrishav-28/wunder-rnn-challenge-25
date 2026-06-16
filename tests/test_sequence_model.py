"""Correctness tests for the causal GRU and its stateful inference wrapper.

The critical invariant: stepping the model one row at a time while carrying the
hidden state reproduces the batched full-sequence forward bit-for-bit (within
float tolerance). This is what guarantees the online submission scores exactly
what was trained/validated.
"""

import numpy as np
import torch

from src.models.sequence_models import CausalGRUForecaster
from src.models.sequence_inference import GRUStatefulPredictionModel


def _make_model(seed=0):
    torch.manual_seed(seed)
    m = CausalGRUForecaster(n_features=32, d_model=48, n_layers=2, dropout=0.0)
    m.eval()
    return m


def test_forward_shapes():
    m = _make_model()
    x = torch.randn(4, 50, 32)
    preds, h = m(x)
    assert preds.shape == (4, 50, 32)
    assert h.shape == (2, 4, 48)


def test_step_matches_batched_forward():
    m = _make_model()
    x = torch.randn(1, 60, 32)
    with torch.no_grad():
        full_preds, _ = m(x)
    # step one row at a time, carrying hidden state
    h = None
    step_preds = []
    for t in range(x.shape[1]):
        p, h = m.step(x[0, t], h)
        step_preds.append(p)
    step_preds = torch.stack(step_preds)
    assert torch.allclose(full_preds[0], step_preds, atol=1e-5), \
        "stateful stepping must match batched forward"


class _DP:
    def __init__(self, seq_ix, step, need, state):
        self.seq_ix = seq_ix
        self.step_in_seq = step
        self.need_prediction = need
        self.state = state


def test_stateful_predictor_api():
    m = _make_model()
    pm = GRUStatefulPredictionModel(m)
    rng = np.random.RandomState(0)
    out_warm = pm.predict(_DP(0, 0, False, rng.randn(32).astype(np.float32)))
    assert out_warm is None
    out_pred = pm.predict(_DP(0, 1, True, rng.randn(32).astype(np.float32)))
    assert isinstance(out_pred, np.ndarray) and out_pred.shape == (32,)
    assert np.isfinite(out_pred).all()
    # new sequence resets hidden state
    pm.predict(_DP(1, 0, False, rng.randn(32).astype(np.float32)))
    assert pm.current_seq == 1


def test_reset_equivalence_across_sequences():
    """Predictions for a sequence must not depend on a prior sequence."""
    m = _make_model()
    rng = np.random.RandomState(1)
    seq = rng.randn(20, 32).astype(np.float32)

    pm1 = GRUStatefulPredictionModel(m)
    preds_clean = [pm1.predict(_DP(5, t, t > 0, seq[t])) for t in range(20)]

    pm2 = GRUStatefulPredictionModel(m)
    for t in range(10):  # process an unrelated sequence first
        pm2.predict(_DP(4, t, t > 0, rng.randn(32).astype(np.float32)))
    preds_after = [pm2.predict(_DP(5, t, t > 0, seq[t])) for t in range(20)]

    for a, b in zip(preds_clean, preds_after):
        if a is None:
            assert b is None
        else:
            assert np.allclose(a, b, atol=1e-6)
