#!/usr/bin/env python3
"""Blend out-of-fold predictions from multiple models and report R2.

Loads aligned OOF arrays (same fold, same scored points/order) for one or more
runs under outputs/oof/<run>/fold_<k>_{predictions,targets}.npy, then reports:
  - each model's mean R2,
  - simple-average blend,
  - per-feature non-negative least-squares blend (stacking weights),
and saves the fitted per-feature weights for use at inference time.

Per-feature weights are fit on this fold's OOF, so the number here is mildly
optimistic; the authoritative check is re-fitting on full dev OOF and scoring
the untouched holdout once.
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


def load_oof(run: str, fold: int):
    d = ROOT / "outputs" / "oof" / run
    pred = np.load(d / f"fold_{fold}_predictions.npy").astype(np.float64)
    tgt = np.load(d / f"fold_{fold}_targets.npy").astype(np.float64)
    return pred, tgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="OOF run dir names under outputs/oof/")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    preds, targets = [], None
    for run in args.runs:
        p, t = load_oof(run, args.fold)
        if targets is None:
            targets = t
        else:
            if t.shape != targets.shape or not np.allclose(t, targets, atol=1e-4):
                raise SystemExit(f"OOF targets for '{run}' do not align with the first run")
        preds.append(p)
    n_feat = targets.shape[1]
    print(f"fold {args.fold}  points={targets.shape[0]}  features={n_feat}  models={len(args.runs)}")

    print("\nindividual mean R2:")
    for run, p in zip(args.runs, preds):
        print(f"  {run:28s} {mean_r2(targets, p):.6f}")

    avg = np.mean(preds, axis=0)
    print(f"\nsimple average               {mean_r2(targets, avg):.6f}")

    # per-feature non-negative stacking weights
    M = len(preds)
    weights = np.zeros((n_feat, M))
    blended = np.zeros_like(targets)
    for j in range(n_feat):
        A = np.stack([p[:, j] for p in preds], axis=1)  # (N, M)
        w, _ = nnls(A, targets[:, j])
        s = w.sum()
        if s > 0:
            w = w / s  # normalize so prediction stays in-scale
        weights[j] = w
        blended[:, j] = A @ w
    print(f"per-feature NNLS blend       {mean_r2(targets, blended):.6f}")
    print("\nmean per-feature weights:")
    for i, run in enumerate(args.runs):
        print(f"  {run:28s} {weights[:, i].mean():.3f}")

    out = args.out or str(ROOT / "outputs" / "oof" / f"blend_fold{args.fold}_weights.json")
    Path(out).write_text(json.dumps({
        "fold": args.fold,
        "runs": args.runs,
        "per_feature_weights": weights.tolist(),
        "simple_average_r2": mean_r2(targets, avg),
        "nnls_blend_r2": mean_r2(targets, blended),
        "individual_r2": {run: mean_r2(targets, p) for run, p in zip(args.runs, preds)},
    }, indent=2), encoding="utf-8")
    print(f"\nsaved weights -> {out}")


if __name__ == "__main__":
    main()
