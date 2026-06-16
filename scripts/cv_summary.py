#!/usr/bin/env python3
"""Summarize cross-validated R2 for a (possibly ensembled) set of OOF runs.

For each dev fold, averages the OOF predictions of the given runs (simple mean)
and computes mean per-feature R2, then reports the CV mean across folds. OOF
arrays must be aligned per fold (same scored points/order), which holds for any
runs produced by train_sequence.py / train_tabular_oof.py on the same folds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent.parent


def mean_r2(y_true, y_pred):
    return float(np.mean([r2_score(y_true[:, j], y_pred[:, j]) for j in range(y_true.shape[1])]))


def load(run, fold):
    d = ROOT / "outputs" / "oof" / run
    p = d / f"fold_{fold}_predictions.npy"
    t = d / f"fold_{fold}_targets.npy"
    if not p.exists() or not t.exists():
        return None, None
    return np.load(p).astype(np.float64), np.load(t).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    per_fold = {}
    for f in args.folds:
        preds, targets = [], None
        for run in args.runs:
            p, t = load(run, f)
            if p is None:
                print(f"  fold {f}: missing OOF for run '{run}' -> skipping fold")
                preds = None
                break
            if targets is None:
                targets = t
            elif not np.allclose(t, targets, atol=1e-4):
                raise SystemExit(f"fold {f}: targets for '{run}' misaligned")
            preds.append(p)
        if not preds:
            continue
        avg = np.mean(preds, axis=0)
        per_fold[f] = mean_r2(targets, avg)
        print(f"fold {f}: ensemble R2 = {per_fold[f]:.6f}  ({len(preds)} members)")

    if per_fold:
        cv = float(np.mean(list(per_fold.values())))
        print(f"\nCV mean over folds {sorted(per_fold)} = {cv:.6f}   runs={args.runs}")
        if args.out:
            Path(args.out).write_text(json.dumps(
                {"runs": args.runs, "per_fold_r2": per_fold, "cv_mean_r2": cv}, indent=2), encoding="utf-8")
            print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
