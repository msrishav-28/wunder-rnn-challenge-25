#!/usr/bin/env python3
"""Train causal tabular models across dev folds and save OOF predictions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.causal_features import feature_schema_hash, feature_schema_payload
from src.data.protocol import (
    create_dataset_manifest,
    current_git_commit,
    get_feature_columns,
    load_sequence_folds,
    load_wunder_dataframe,
    resolve_dataset_path,
    sha256_file,
)
from src.evaluation.stepwise import StepwiseScorer
from src.models.baselines import TabularStatefulPredictionModel
from src.training.baseline_training import build_tabular_supervised_matrix, create_tabular_estimator
from src.utils.hardware import assert_within_max_safe, build_hardware_profile
from src.utils.reproducibility import set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train OOF causal tabular baselines")
    parser.add_argument("--data", default="data/raw/train.parquet")
    parser.add_argument("--folds", default="config/folds.json")
    parser.add_argument("--model-type", choices=["ridge", "ridge_cv", "elasticnet", "lightgbm"], default="ridge_cv")
    parser.add_argument("--feature-schema", choices=["compact_v1", "expanded_v2"], default="compact_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--output-dir", default="models/phase2_oof")
    parser.add_argument("--oof-dir", default="outputs/oof")
    parser.add_argument("--dev-folds", default="",
                        help="Comma-separated dev fold ids to run (default: all non-holdout)")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed, seed_torch=False)
    profile = build_hardware_profile("max_safe")
    assert_within_max_safe(profile)

    data_path = resolve_dataset_path(args.data)
    dataset_manifest = create_dataset_manifest(data_path)
    folds = load_sequence_folds(args.folds)
    output_root = Path(args.output_dir) / f"{args.model_type}_{args.feature_schema}"
    oof_root = Path(args.oof_dir) / f"{args.model_type}_{args.feature_schema}"
    output_root.mkdir(parents=True, exist_ok=True)
    oof_root.mkdir(parents=True, exist_ok=True)

    selected_folds = (
        {int(x) for x in args.dev_folds.split(",") if x.strip() != ""}
        if args.dev_folds else None
    )

    fold_results = []
    for fold_id, val_seq_ids in sorted(folds.folds.items(), key=lambda item: int(item[0])):
        if int(fold_id) == folds.final_holdout_fold:
            continue
        if selected_folds is not None and int(fold_id) not in selected_folds:
            continue
        fold_model_dir = output_root / f"fold_{fold_id}"
        pred_path = oof_root / f"fold_{fold_id}_predictions.npy"
        target_path = oof_root / f"fold_{fold_id}_targets.npy"
        fold_manifest_path = fold_model_dir / "model_manifest.json"
        if args.resume and pred_path.exists() and target_path.exists() and fold_manifest_path.exists():
            fold_results.append(json.loads(fold_manifest_path.read_text(encoding="utf-8")))
            print(f"Fold {fold_id}: resume, existing artifacts found")
            continue

        train_seq_ids = sorted(
            seq_id for seq_id in folds.train_dev_seq_ids if seq_id not in set(val_seq_ids)
        )
        feature_start = time.perf_counter()
        x_train, y_train = build_tabular_supervised_matrix(
            str(data_path),
            train_seq_ids,
            feature_schema=args.feature_schema,
        )
        feature_seconds = time.perf_counter() - feature_start
        estimator = create_tabular_estimator(args.model_type, seed=args.seed, n_jobs=args.n_jobs)

        train_start = time.perf_counter()
        estimator.fit(x_train, y_train)
        train_seconds = time.perf_counter() - train_start

        val_df = load_wunder_dataframe(data_path, seq_ids=val_seq_ids)
        model = TabularStatefulPredictionModel(estimator, feature_schema=args.feature_schema)
        score = StepwiseScorer(val_df).score(model)

        model = TabularStatefulPredictionModel(estimator, feature_schema=args.feature_schema)
        predictions = []
        targets = []
        for _, seq_df in val_df.groupby("seq_ix", sort=True):
            pending = None
            states = seq_df[get_feature_columns(val_df)].to_numpy(dtype=np.float32)
            for pos, row in enumerate(seq_df.itertuples(index=False)):
                if pending is not None:
                    predictions.append(pending)
                    targets.append(states[pos])
                data_point = type(
                    "DataPoint",
                    (),
                    {
                        "seq_ix": int(row.seq_ix),
                        "step_in_seq": int(row.step_in_seq),
                        "need_prediction": bool(row.need_prediction),
                        "state": states[pos],
                    },
                )()
                pending = model.predict(data_point)

        fold_model_dir.mkdir(parents=True, exist_ok=True)
        model_path = fold_model_dir / "model.joblib"
        joblib.dump(estimator, model_path)
        np.save(pred_path, np.asarray(predictions, dtype=np.float32))
        np.save(target_path, np.asarray(targets, dtype=np.float32))

        fold_manifest = {
            "model_type": args.model_type,
            "feature_schema": args.feature_schema,
            "feature_schema_hash": feature_schema_hash(args.feature_schema),
            "feature_schema_payload": feature_schema_payload(args.feature_schema),
            "seed": args.seed,
            "n_jobs": args.n_jobs,
            "fold_id": int(fold_id),
            "dataset_sha256": dataset_manifest.sha256,
            "folds_path": args.folds,
            "git_commit": current_git_commit(),
            "train_seq_ids": train_seq_ids,
            "val_seq_ids": [int(x) for x in val_seq_ids],
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "oof_predictions_path": str(pred_path),
            "oof_targets_path": str(target_path),
            "train_rows": int(x_train.shape[0]),
            "feature_width": int(x_train.shape[1]),
            "feature_build_seconds": feature_seconds,
            "train_seconds": train_seconds,
            "val_score": asdict(score),
            "val_mean_r2": score.mean_r2,
        }
        fold_manifest_path.write_text(json.dumps(fold_manifest, indent=2, sort_keys=True), encoding="utf-8")
        fold_results.append(fold_manifest)
        print(f"Fold {fold_id}: mean_r2={score.mean_r2:.6f}, model={model_path}")

    summary = {
        "model_type": args.model_type,
        "feature_schema": args.feature_schema,
        "dataset_sha256": dataset_manifest.sha256,
        "folds_path": args.folds,
        "fold_results": fold_results,
        "mean_oof_r2": float(np.mean([fold["val_mean_r2"] for fold in fold_results])) if fold_results else None,
    }
    summary_path = oof_root / "oof_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"OOF summary written to {summary_path}")
    print(f"Mean OOF R2: {summary['mean_oof_r2']}")


if __name__ == "__main__":
    main()
