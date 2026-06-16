#!/usr/bin/env python3
"""Fast per-feature GBDT out-of-fold predictions for ensembling.

Builds causal tabular features for all scored points in one vectorized pass and
predicts the validation fold in batch (no row-by-row replay), so the OOF arrays
align exactly with the sequence-model OOF (same fold, same points/order). This
makes the GBDT a cheap, strongly-decorrelated stacking member.

HistGradientBoostingRegressor is trained per target (multi-threaded internally,
no per-process data duplication), so it is RAM-safe.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.protocol import load_sequence_folds, resolve_dataset_path
from src.training.baseline_training import build_tabular_supervised_matrix


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/train.parquet")
    ap.add_argument("--folds", default="config/folds.json")
    ap.add_argument("--feature-schema", default="compact_v1", choices=["compact_v1", "expanded_v2"])
    ap.add_argument("--run-name", default="gbdt_fast")
    ap.add_argument("--dev-folds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--max-iter", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--max-leaf-nodes", type=int, default=31)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    data_path = resolve_dataset_path(args.data)
    folds = load_sequence_folds(args.folds)
    oof_dir = ROOT / "outputs" / "oof" / args.run_name
    oof_dir.mkdir(parents=True, exist_ok=True)

    for fold in args.dev_folds:
        if fold == folds.final_holdout_fold:
            continue
        t0 = time.perf_counter()
        val_ids = sorted(int(x) for x in folds.folds[str(fold)])
        train_ids = sorted(s for s in folds.train_dev_seq_ids if s not in set(val_ids))
        X_tr, Y_tr = build_tabular_supervised_matrix(str(data_path), train_ids, feature_schema=args.feature_schema)
        X_va, Y_va = build_tabular_supervised_matrix(str(data_path), val_ids, feature_schema=args.feature_schema)

        preds = np.zeros_like(Y_va)
        for j in range(Y_tr.shape[1]):
            model = HistGradientBoostingRegressor(
                max_iter=args.max_iter, learning_rate=args.lr,
                max_leaf_nodes=args.max_leaf_nodes, l2_regularization=args.l2,
                random_state=args.seed,
            )
            model.fit(X_tr, Y_tr[:, j])
            preds[:, j] = model.predict(X_va)

        np.save(oof_dir / f"fold_{fold}_predictions.npy", preds.astype(np.float32))
        np.save(oof_dir / f"fold_{fold}_targets.npy", Y_va.astype(np.float32))
        from sklearn.metrics import r2_score
        r2 = float(np.mean([r2_score(Y_va[:, j], preds[:, j]) for j in range(Y_va.shape[1])]))
        print(f"fold {fold}: mean R2 = {r2:.6f}  ({time.perf_counter()-t0:.0f}s, rows tr={X_tr.shape[0]} feat={X_tr.shape[1]})", flush=True)

    print("GBDT FAST OOF DONE", flush=True)


if __name__ == "__main__":
    main()
