#!/usr/bin/env python3
"""Create deterministic locked folds by seq_ix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.protocol import (
    create_dataset_manifest,
    create_sequence_folds,
    load_wunder_dataframe,
    resolve_dataset_path,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Create Wunder sequence folds")
    parser.add_argument("--data", default="data/raw/train.parquet")
    parser.add_argument("--output", default="config/folds.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--final-holdout-fold", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = resolve_dataset_path(args.data)
    manifest = create_dataset_manifest(data_path)
    df = load_wunder_dataframe(data_path)
    seq_ids = sorted(int(x) for x in df["seq_ix"].unique())
    folds = create_sequence_folds(
        seq_ids,
        n_folds=args.n_folds,
        seed=args.seed,
        final_holdout_fold=args.final_holdout_fold,
        dataset_sha256=manifest.sha256,
    )
    write_json(args.output, folds)
    print(f"Folds written to {args.output}")
    print(f"  train/dev sequences: {len(folds.train_dev_seq_ids)}")
    print(f"  final holdout sequences: {len(folds.final_holdout_seq_ids)}")


if __name__ == "__main__":
    main()
