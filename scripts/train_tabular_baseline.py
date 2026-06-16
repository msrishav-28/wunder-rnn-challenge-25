#!/usr/bin/env python3
"""Train Ridge/ElasticNet/LightGBM on causal raw-32 tabular features."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.protocol import (
    create_dataset_manifest,
    create_sequence_folds,
    current_git_commit,
    get_feature_columns,
    load_sequence_folds,
    load_wunder_dataframe,
    resolve_dataset_path,
    sha256_file,
)
from src.data.causal_features import feature_schema_hash, feature_schema_payload
from src.evaluation.stepwise import StepwiseScorer
from src.models.baselines import TabularStatefulPredictionModel
from src.training.baseline_training import build_tabular_supervised_matrix, create_tabular_estimator
from src.utils.reproducibility import set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train causal tabular baseline")
    parser.add_argument("--data", default="data/raw/train.parquet")
    parser.add_argument("--folds", default="config/folds.json")
    parser.add_argument("--model-type", choices=["ridge", "ridge_cv", "elasticnet", "lightgbm"], default="ridge")
    parser.add_argument("--feature-schema", choices=["compact_v1", "expanded_v2"], default="compact_v1")
    parser.add_argument("--dev-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--hardware-profile", default="outputs/hardware/max_safe_profile.json")
    parser.add_argument("--output-dir", default="models/phase2_tabular")
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed, seed_torch=False)
    data_path = resolve_dataset_path(args.data)
    dataset_manifest = create_dataset_manifest(data_path)
    hardware_profile_path = Path(args.hardware_profile)
    folds_path = Path(args.folds)
    if folds_path.exists():
        folds = load_sequence_folds(folds_path)
    else:
        df = load_wunder_dataframe(data_path)
        folds = create_sequence_folds(sorted(int(x) for x in df["seq_ix"].unique()), seed=args.seed)

    if str(args.dev_fold) == str(folds.final_holdout_fold):
        raise ValueError("dev-fold cannot be the untouched final holdout fold")

    val_seq_ids = folds.folds[str(args.dev_fold)]
    train_seq_ids = sorted(
        seq_id for seq_id in folds.train_dev_seq_ids if seq_id not in set(val_seq_ids)
    )
    feature_start = time.perf_counter()
    x_train, y_train = build_tabular_supervised_matrix(
        str(data_path),
        train_seq_ids,
        feature_schema=args.feature_schema,
    )
    feature_build_seconds = time.perf_counter() - feature_start
    estimator = create_tabular_estimator(args.model_type, seed=args.seed, n_jobs=args.n_jobs)

    start = time.perf_counter()
    estimator.fit(x_train, y_train)
    train_seconds = time.perf_counter() - start

    val_df = load_wunder_dataframe(data_path, seq_ids=val_seq_ids)
    score = StepwiseScorer(val_df).score(
        TabularStatefulPredictionModel(estimator, feature_schema=args.feature_schema)
    )

    output_dir = Path(args.output_dir) / args.model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    manifest_path = output_dir / "model_manifest.json"
    joblib.dump(estimator, model_path)
    manifest = {
        "model_type": args.model_type,
        "feature_schema": args.feature_schema,
        "feature_schema_hash": feature_schema_hash(args.feature_schema),
        "feature_schema_payload": feature_schema_payload(args.feature_schema),
        "seed": args.seed,
        "n_jobs": args.n_jobs,
        "hardware_profile_path": str(hardware_profile_path),
        "hardware_profile_sha256": sha256_file(hardware_profile_path) if hardware_profile_path.exists() else None,
        "git_commit": current_git_commit(),
        "data_path": str(data_path),
        "dataset_sha256": dataset_manifest.sha256,
        "folds_path": str(folds_path),
        "folds_dataset_sha256": folds.dataset_sha256,
        "dev_fold": args.dev_fold,
        "final_holdout_fold": folds.final_holdout_fold,
        "feature_columns": get_feature_columns(val_df),
        "train_seq_ids": train_seq_ids,
        "val_seq_ids": val_seq_ids,
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "train_rows": int(x_train.shape[0]),
        "feature_width": int(x_train.shape[1]),
        "val_mean_r2": score.mean_r2,
        "val_score": asdict(score),
        "feature_build_seconds": feature_build_seconds,
        "train_seconds": train_seconds,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved model: {model_path}")
    print(f"Saved manifest: {manifest_path}")
    print(f"Validation mean R2: {score.mean_r2:.6f}")


if __name__ == "__main__":
    main()
