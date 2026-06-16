#!/usr/bin/env python3
"""Evaluate non-trained causal baselines with the stepwise scorer."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.protocol import create_sequence_folds, load_sequence_folds, load_wunder_dataframe, resolve_dataset_path
from src.evaluation.stepwise import StepwiseScorer
from src.models.baselines import EWMAPredictionModel, MomentumPredictionModel, PersistencePredictionModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 simple baselines")
    parser.add_argument("--data", default="data/raw/train.parquet")
    parser.add_argument("--folds", default="config/folds.json")
    parser.add_argument("--split", choices=["holdout", "dev0"], default="holdout")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def _load_or_create_eval_seq_ids(data_path: Path, folds_path: Path, split: str) -> list[int]:
    if folds_path.exists():
        folds = load_sequence_folds(folds_path)
    else:
        df = load_wunder_dataframe(data_path)
        folds = create_sequence_folds(sorted(int(x) for x in df["seq_ix"].unique()))
    if split == "holdout":
        return folds.final_holdout_seq_ids
    return folds.folds["0"]


def main():
    args = parse_args()
    data_path = resolve_dataset_path(args.data)
    seq_ids = _load_or_create_eval_seq_ids(data_path, Path(args.folds), args.split)
    df = load_wunder_dataframe(data_path, seq_ids=seq_ids)
    scorer = StepwiseScorer(df)
    candidates = [
        ("persistence", PersistencePredictionModel()),
        ("momentum_alpha_0.25", MomentumPredictionModel(alpha=0.25)),
        ("momentum_alpha_0.50", MomentumPredictionModel(alpha=0.50)),
        ("momentum_alpha_1.00", MomentumPredictionModel(alpha=1.00)),
        ("ewma_alpha_0.10", EWMAPredictionModel(alpha=0.10)),
        ("ewma_alpha_0.25", EWMAPredictionModel(alpha=0.25)),
        ("ewma_alpha_0.50", EWMAPredictionModel(alpha=0.50)),
    ]
    results = {}
    for name, model in candidates:
        start = time.perf_counter()
        score = scorer.score(model)
        elapsed = time.perf_counter() - start
        per_prediction_ms = elapsed / max(score.n_predictions_requested, 1) * 1000.0
        results[name] = {
            **asdict(score),
            "elapsed_seconds": elapsed,
            "per_prediction_ms": per_prediction_ms,
            "estimated_500seq_minutes": per_prediction_ms * 500 * 900 / 1000 / 60,
        }
        print(f"{name}: mean_r2={score.mean_r2:.6f}, {per_prediction_ms:.3f} ms/pred")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Results written to {out}")


if __name__ == "__main__":
    main()
