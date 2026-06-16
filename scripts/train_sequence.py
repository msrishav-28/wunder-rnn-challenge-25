#!/usr/bin/env python3
"""Train the causal GRU next-state model on a dev fold and save artifacts.

Saves:
  models/phase2_seq/<run>/fold_<k>/model.pt        (weights + config)
  outputs/oof/<run>/fold_<k>_predictions.npy        (scorer-order OOF preds)
  outputs/oof/<run>/fold_<k>_targets.npy
  models/phase2_seq/<run>/fold_<k>/model_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.protocol import (
    create_dataset_manifest,
    current_git_commit,
    load_sequence_folds,
    load_wunder_dataframe,
    resolve_dataset_path,
)
from src.evaluation.stepwise import StepwiseScorer
from src.models.sequence_inference import GRUStatefulPredictionModel
from src.training.sequence_trainer import TrainConfig, train_sequence_model


def parse_args():
    p = argparse.ArgumentParser(description="Train causal GRU next-state model")
    p.add_argument("--data", default="data/raw/train.parquet")
    p.add_argument("--folds", default="config/folds.json")
    p.add_argument("--dev-fold", type=int, default=0)
    p.add_argument("--train-all-dev", action="store_true",
                   help="Train on all dev folds (0..3) with an internal early-stop split; "
                        "fold 4 holdout stays untouched. Use for the final holdout-eval model.")
    p.add_argument("--train-all", action="store_true",
                   help="Train on ALL sequences (dev + holdout) with an internal early-stop "
                        "split. Use ONLY for the final packaged submission model — never report "
                        "a local score from it (it has seen the holdout).")
    p.add_argument("--internal-val-frac", type=float, default=0.15)
    p.add_argument("--run-name", default="gru_v1")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--rnn-type", default="gru", choices=["gru", "lstm"])
    p.add_argument("--arch", default="rnn", choices=["rnn", "tcn"])
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--head-hidden", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="models/phase2_seq")
    p.add_argument("--oof-dir", default="outputs/oof")
    p.add_argument("--confirm-stepwise", action="store_true",
                   help="Also replay val set through StepwiseScorer to confirm batched R2.")
    return p.parse_args()


def main():
    args = parse_args()
    data_path = resolve_dataset_path(args.data)
    folds = load_sequence_folds(args.folds)

    if args.train_all_dev or args.train_all:
        # Final-model mode: carve a deterministic internal validation slice for
        # early stopping. train_all_dev keeps the holdout untouched; train_all
        # uses every sequence (for the packaged submission only).
        import numpy as _np
        if args.train_all:
            dev_ids = sorted(int(s) for s in (list(folds.train_dev_seq_ids) + list(folds.final_holdout_seq_ids)))
        else:
            dev_ids = sorted(int(s) for s in folds.train_dev_seq_ids)
        rng = _np.random.RandomState(args.seed)
        shuffled = _np.array(dev_ids); rng.shuffle(shuffled)
        n_val = max(1, int(round(len(dev_ids) * args.internal_val_frac)))
        val_ids = sorted(int(x) for x in shuffled[:n_val])
        train_ids = sorted(int(x) for x in shuffled[n_val:])
        _mode = "train_all (dev+holdout)" if args.train_all else "train_all_dev (holdout untouched)"
        print(f"{_mode}: train={len(train_ids)} internal_val={len(val_ids)}")
    else:
        if args.dev_fold == folds.final_holdout_fold:
            raise SystemExit(f"--dev-fold {args.dev_fold} is the final holdout; pick another")
        val_ids = sorted(int(x) for x in folds.folds[str(args.dev_fold)])
        train_ids = sorted(s for s in folds.train_dev_seq_ids if s not in set(val_ids))

    cfg = TrainConfig(
        d_model=args.d_model, n_layers=args.n_layers, dropout=args.dropout,
        head_hidden=(args.head_hidden or None), epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        patience=args.patience, threads=args.threads, seed=args.seed,
        device=args.device, rnn_type=args.rnn_type, arch=args.arch,
        kernel_size=args.kernel_size,
    )
    print(f"== GRU run '{args.run_name}' fold {args.dev_fold}: "
          f"d_model={cfg.d_model} layers={cfg.n_layers} dropout={cfg.dropout} "
          f"epochs={cfg.epochs} bs={cfg.batch_size} lr={cfg.lr} threads={cfg.threads} ==")

    t0 = time.perf_counter()
    result = train_sequence_model(str(data_path), train_ids, val_ids, cfg)
    train_seconds = time.perf_counter() - t0
    print(f"best batched val R2 = {result['best_val_r2']:.6f}  ({train_seconds:.0f}s)")

    out_dir = Path(args.output_dir) / args.run_name / f"fold_{args.dev_fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    oof_dir = Path(args.oof_dir) / args.run_name
    oof_dir.mkdir(parents=True, exist_ok=True)

    if cfg.arch == "tcn":
        model_params = {
            "type": "CausalTCN",
            "params": {
                "n_features": cfg.n_features, "d_model": cfg.d_model,
                "n_layers": cfg.n_layers, "dropout": cfg.dropout,
                "head_hidden": cfg.head_hidden, "kernel_size": cfg.kernel_size,
            },
        }
    else:
        model_params = {
            "type": "CausalGRUForecaster",
            "params": {
                "n_features": cfg.n_features, "d_model": cfg.d_model,
                "n_layers": cfg.n_layers, "dropout": cfg.dropout,
                "head_hidden": cfg.head_hidden, "rnn_type": cfg.rnn_type,
            },
        }
    ckpt = {
        "model_state_dict": result["best_state_dict"],
        "config": {"model": model_params, "train": vars(args)},
        "best_val_r2": result["best_val_r2"],
    }
    model_path = out_dir / "model.pt"
    torch.save(ckpt, model_path)

    y_true, y_pred = result["oof"]
    np.save(oof_dir / f"fold_{args.dev_fold}_targets.npy", y_true.astype(np.float32))
    np.save(oof_dir / f"fold_{args.dev_fold}_predictions.npy", y_pred.astype(np.float32))

    stepwise_r2 = None
    if args.confirm_stepwise:
        from src.models.sequence_inference import load_gru_checkpoint
        model = load_gru_checkpoint(model_path)
        val_df = load_wunder_dataframe(data_path, seq_ids=val_ids)
        score = StepwiseScorer(val_df).score(GRUStatefulPredictionModel(model))
        stepwise_r2 = score.mean_r2
        print(f"stepwise-replay val R2 = {stepwise_r2:.6f} (should match batched)")

    manifest = {
        "run_name": args.run_name,
        "model": model_params,
        "dev_fold": args.dev_fold,
        "train_seq_ids": train_ids,
        "val_seq_ids": [int(x) for x in val_ids],
        "dataset_sha256": create_dataset_manifest(data_path).sha256,
        "git_commit": current_git_commit(),
        "train_config": vars(cfg) if hasattr(cfg, "__dict__") else cfg.__dict__,
        "best_val_r2_batched": result["best_val_r2"],
        "best_val_r2_stepwise": stepwise_r2,
        "best_per_feature": result["best_per_feature"],
        "history": result["history"],
        "train_seconds": train_seconds,
        "model_path": str(model_path),
    }
    (out_dir / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=float), encoding="utf-8")
    print(f"saved -> {model_path}")
    print(f"manifest -> {out_dir/'model_manifest.json'}")


if __name__ == "__main__":
    main()
