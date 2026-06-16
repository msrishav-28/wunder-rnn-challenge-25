#!/usr/bin/env python3
"""Evaluate root solution.py through the official-style stepwise replay."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from solution import PredictionModel
from src.data.protocol import load_sequence_folds, load_wunder_dataframe, resolve_dataset_path
from src.evaluation.stepwise import StepwiseScorer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate root solution.py on a locked split")
    parser.add_argument("--data", default="data/raw/train.parquet")
    parser.add_argument("--folds", default="config/folds.json")
    parser.add_argument("--split", choices=["holdout", "dev0"], default="holdout")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = resolve_dataset_path(args.data)
    folds = load_sequence_folds(args.folds)
    seq_ids = folds.final_holdout_seq_ids if args.split == "holdout" else folds.folds["0"]

    df = load_wunder_dataframe(data_path, seq_ids=seq_ids)
    model = PredictionModel()
    start = time.perf_counter()
    score = StepwiseScorer(df).score(model)
    elapsed = time.perf_counter() - start
    per_prediction_ms = elapsed / max(score.n_predictions_requested, 1) * 1000.0

    result = {
        **asdict(score),
        "backend": model.backend,
        "model_count": len(model.models),
        "split": args.split,
        "elapsed_seconds": elapsed,
        "per_prediction_ms": per_prediction_ms,
        "estimated_500seq_minutes": per_prediction_ms * 500 * 899 / 1000 / 60,
    }
    print(f"backend={model.backend} models={len(model.models)}")
    print(f"{args.split} mean_r2={score.mean_r2:.6f}")
    print(f"scored={score.n_predictions_scored} requested={score.n_predictions_requested}")
    print(f"latency={per_prediction_ms:.3f} ms/pred")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Results written to {out}")


if __name__ == "__main__":
    main()
