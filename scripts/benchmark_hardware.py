#!/usr/bin/env python3
"""Run a lightweight max-safe benchmark and batch calibration."""

from __future__ import annotations

import argparse
import contextlib
import os
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.hardware import (
    assert_within_max_safe,
    atomic_write_json,
    build_hardware_profile,
    calibrate_batch_size,
    dataclass_to_dict,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark local max-safe training settings")
    parser.add_argument("--profile", default="max_safe")
    parser.add_argument("--output", default="outputs/hardware/max_safe_profile.json")
    parser.add_argument("--lookback", type=int, default=128)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--skip-torch-trial", action="store_true")
    return parser.parse_args()


def _make_torch_trial(device: str, lookback: int, features: int, hidden: int):
    with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stderr(devnull):
        import torch

    model = torch.nn.Sequential(
        torch.nn.Linear(features, hidden),
        torch.nn.GELU(),
        torch.nn.Linear(hidden, features),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    use_amp = device == "cuda"

    def trial(batch_size: int) -> None:
        optimizer.zero_grad(set_to_none=True)
        x = torch.randn(batch_size, lookback, features, device=device)
        target = torch.randn(batch_size, features, device=device)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(x[:, -1, :])
                loss = torch.nn.functional.mse_loss(pred, target)
        else:
            pred = model(x[:, -1, :])
            loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        del x, target, pred, loss
        if device == "cuda":
            torch.cuda.empty_cache()

    return trial


def main():
    args = parse_args()
    profile = build_hardware_profile(args.profile)
    assert_within_max_safe(profile)

    benchmark = {
        "profile": dataclass_to_dict(profile),
        "torch_batch_calibration": None,
        "elapsed_seconds": None,
    }
    start = time.perf_counter()

    if not args.skip_torch_trial and profile.torch.installed:
        try:
            with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stderr(devnull):
                import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            trial = _make_torch_trial(device, args.lookback, args.features, args.hidden)
            batch_size = calibrate_batch_size(trial, profile.max_safe.batch_candidates, repeats=2)
            benchmark["torch_batch_calibration"] = {
                "device": device,
                "selected_batch_size": batch_size,
                "candidates": profile.max_safe.batch_candidates,
                "lookback": args.lookback,
                "features": args.features,
                "hidden": args.hidden,
            }
            print(f"Selected torch batch size: {batch_size} on {device}")
        except Exception as exc:
            benchmark["torch_batch_calibration"] = {"error": repr(exc)}
            print(f"Torch calibration skipped/failed: {exc}")
    else:
        print("Torch calibration skipped")

    benchmark["elapsed_seconds"] = time.perf_counter() - start
    atomic_write_json(args.output, benchmark)
    print(f"Max-safe benchmark written to {args.output}")
    print(json.dumps(benchmark["torch_batch_calibration"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
