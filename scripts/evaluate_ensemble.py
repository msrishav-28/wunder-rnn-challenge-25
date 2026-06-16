#!/usr/bin/env python3
"""Build an ensemble from saved artifacts and score it on a split.

Members can be any mix of trained GRU checkpoints and joblib tabular models.
The split is either the untouched final holdout or a named dev fold. Uses the
leak-free StepwiseScorer (row-by-row replay, never crossing seq_ix).

Example:
  python scripts/evaluate_ensemble.py --split holdout \
      --gru models/phase2_seq/gru_d256/fold_0/model.pt \
            models/phase2_seq/gru_s123/fold_0/model.pt
"""

from __future__ import annotations

import os
# Windows: numpy/sklearn (MKL) and torch each ship an OpenMP runtime; loading
# both in one process can fail c10.dll init. Allow co-existence.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import time
from pathlib import Path

import torch  # import torch before numpy/sklearn (MKL) to avoid c10.dll init failure on Windows
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.protocol import load_sequence_folds, load_wunder_dataframe, resolve_dataset_path
from src.evaluation.stepwise import StepwiseScorer
from src.models.ensemble_predictor import EnsemblePredictionModel
from src.models.sequence_inference import GRUStatefulPredictionModel, load_gru_checkpoint


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/train.parquet")
    ap.add_argument("--folds", default="config/folds.json")
    ap.add_argument("--split", default="holdout", help="'holdout' or a dev fold id like '0'")
    ap.add_argument("--gru", nargs="*", default=[], help="GRU checkpoint .pt paths")
    ap.add_argument("--tabular", nargs="*", default=[],
                    help="joblib paths; schema read from sibling model_manifest.json")
    ap.add_argument("--weights", default="", help="optional blend weights json (per_feature_weights)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--output", default="")
    return ap.parse_args()


def main():
    args = parse_args()
    import torch
    torch.set_num_threads(int(args.threads))

    folds = load_sequence_folds(args.folds)
    if args.split == "holdout":
        seq_ids = folds.final_holdout_seq_ids
    else:
        seq_ids = folds.folds[str(args.split)]
    data_path = resolve_dataset_path(args.data)
    val_df = load_wunder_dataframe(data_path, seq_ids=seq_ids)

    members = []
    names = []
    for ckpt in args.gru:
        model = load_gru_checkpoint(ckpt)
        members.append(GRUStatefulPredictionModel(model))
        names.append(Path(ckpt).parent.parent.name)
    for jb in args.tabular:
        import joblib
        from src.models.baselines import TabularStatefulPredictionModel
        manifest = Path(jb).parent / "model_manifest.json"
        schema = "compact_v1"
        if manifest.exists():
            schema = json.loads(manifest.read_text()).get("feature_schema", "compact_v1")
        members.append(TabularStatefulPredictionModel(joblib.load(jb), feature_schema=schema))
        names.append(Path(jb).parent.name)

    weights = None
    if args.weights:
        w = json.loads(Path(args.weights).read_text())["per_feature_weights"]
        weights = np.asarray(w, dtype=np.float32)

    ensemble = EnsemblePredictionModel(members, weights=weights)
    print(f"split={args.split}  seqs={len(seq_ids)}  members={names}  weighted={weights is not None}")

    t0 = time.perf_counter()
    score = StepwiseScorer(val_df).score(ensemble)
    elapsed = time.perf_counter() - t0
    n = score.n_predictions_scored
    print(f"mean R2 = {score.mean_r2:.6f}   ({elapsed:.0f}s, {1000*elapsed/n:.3f} ms/pred over {n} preds)")

    payload = {
        "split": args.split, "members": names, "weighted": weights is not None,
        "mean_r2": score.mean_r2, "r2_per_feature": score.r2_per_feature,
        "n_predictions_scored": n, "elapsed_seconds": elapsed,
        "ms_per_pred": 1000 * elapsed / n,
    }
    out = args.output or str(ROOT / "outputs" / "baselines" / f"ensemble_{args.split}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
