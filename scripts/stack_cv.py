#!/usr/bin/env python3
"""Per-feature non-negative stacking with honest (nested) cross-validation.

For each held-out dev fold, fits per-feature non-negative least-squares weights
over the member OOF predictions of the *other* folds, then applies them to the
held-out fold. This avoids fitting weights on the same data they're scored on,
so the reported CV mean is an honest estimate of stacked-ensemble performance.

Also fits one final weight set on all folds (for inference) and writes it as a
(n_features, n_members) matrix compatible with EnsemblePredictionModel and
solution.py's blend_weights.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent.parent


def mean_r2(y_true, y_pred):
    return float(np.mean([r2_score(y_true[:, j], y_pred[:, j]) for j in range(y_true.shape[1])]))


def load(run, fold):
    d = ROOT / "outputs" / "oof" / run
    p, t = d / f"fold_{fold}_predictions.npy", d / f"fold_{fold}_targets.npy"
    if not p.exists():
        return None, None
    return np.load(p).astype(np.float64), np.load(t).astype(np.float64)


def fit_per_feature_weights(member_preds, targets):
    """member_preds: list of (N,F); targets: (N,F). Returns (F, M) normalized weights."""
    F = targets.shape[1]
    M = len(member_preds)
    W = np.zeros((F, M))
    for j in range(F):
        A = np.stack([p[:, j] for p in member_preds], axis=1)
        w, _ = nnls(A, targets[:, j])
        s = w.sum()
        W[j] = w / s if s > 0 else np.full(M, 1.0 / M)
    return W


def apply_weights(member_preds, W):
    out = np.zeros_like(member_preds[0])
    A = np.stack(member_preds, axis=0)  # (M, N, F)
    for j in range(W.shape[0]):
        out[:, j] = np.einsum("m,mn->n", W[j], A[:, :, j])  # W[j]: (M,), A[:,:,j]: (M,N)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--save-weights", default="")
    args = ap.parse_args()

    # load all members for all folds
    data = {f: {"preds": [], "targets": None} for f in args.folds}
    for f in args.folds:
        for run in args.runs:
            p, t = load(run, f)
            if p is None:
                raise SystemExit(f"missing OOF: run={run} fold={f}")
            if data[f]["targets"] is None:
                data[f]["targets"] = t
            elif not np.allclose(t, data[f]["targets"], atol=1e-4):
                raise SystemExit(f"targets misaligned: run={run} fold={f}")
            data[f]["preds"].append(p)

    # nested CV: fit weights on other folds, score held-out fold
    per_fold = {}
    for held in args.folds:
        train_folds = [f for f in args.folds if f != held]
        cat_preds = [np.concatenate([data[f]["preds"][m] for f in train_folds], axis=0)
                     for m in range(len(args.runs))]
        cat_tgt = np.concatenate([data[f]["targets"] for f in train_folds], axis=0)
        W = fit_per_feature_weights(cat_preds, cat_tgt)
        blended = apply_weights(data[held]["preds"], W)
        per_fold[held] = mean_r2(data[held]["targets"], blended)
        print(f"fold {held}: stacked R2 = {per_fold[held]:.6f}")

    cv = float(np.mean(list(per_fold.values())))
    print(f"\nstacked CV mean over folds {sorted(per_fold)} = {cv:.6f}")
    print(f"members: {args.runs}")

    if args.save_weights:
        # final weights fit on ALL folds, for inference
        all_preds = [np.concatenate([data[f]["preds"][m] for f in args.folds], axis=0)
                     for m in range(len(args.runs))]
        all_tgt = np.concatenate([data[f]["targets"] for f in args.folds], axis=0)
        W_final = fit_per_feature_weights(all_preds, all_tgt)
        Path(args.save_weights).write_text(json.dumps({
            "members": args.runs,
            "per_feature_weights": W_final.tolist(),
            "stacked_cv_mean_r2": cv,
        }, indent=2), encoding="utf-8")
        print(f"mean member weights: {dict(zip(args.runs, [round(float(x),3) for x in W_final.mean(0)]))}")
        print(f"saved weights -> {args.save_weights}")


if __name__ == "__main__":
    main()
