#!/usr/bin/env python3
"""Validate Wunder data and optionally write a dataset manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.protocol import create_dataset_manifest, resolve_dataset_path, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Validate train.parquet protocol")
    parser.add_argument("--data", default="data/raw/train.parquet", help="Preferred parquet path")
    parser.add_argument(
        "--manifest",
        default="outputs/protocol/dataset_manifest.json",
        help="Manifest path to write when --write is set",
    )
    parser.add_argument("--write", action="store_true", help="Write manifest JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = resolve_dataset_path(args.data)
    manifest = create_dataset_manifest(data_path)
    print("Protocol validation passed")
    print(f"  path: {manifest.path}")
    print(f"  sha256: {manifest.sha256}")
    print(f"  rows: {manifest.row_count}")
    print(f"  sequences: {manifest.sequence_count}")
    print(f"  features: {len(manifest.feature_columns)}")
    true_steps = [int(k) for k, v in manifest.need_prediction_counts_by_step.items() if v > 0]
    if true_steps:
        print(f"  need_prediction steps: {min(true_steps)}..{max(true_steps)}")
    if args.write:
        write_json(args.manifest, manifest)
        print(f"Manifest written to {args.manifest}")


if __name__ == "__main__":
    main()
