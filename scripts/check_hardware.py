#!/usr/bin/env python3
"""Inspect local hardware and write a max-safe profile manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.hardware import atomic_write_json, build_hardware_profile, dataclass_to_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Check local hardware for Phase 2 training")
    parser.add_argument("--profile", default="max_safe")
    parser.add_argument("--output", default="outputs/hardware/max_safe_profile.json")
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    profile = build_hardware_profile(args.profile)
    payload = dataclass_to_dict(profile)

    print("Hardware profile")
    print(f"  profile: {profile.profile_name}")
    print(f"  platform: {profile.platform}")
    print(f"  python: {profile.python}")
    print(f"  cpu: {profile.cpu_name}")
    print(f"  logical CPUs: {profile.logical_cpu_count}")
    print(f"  RAM: {profile.memory.total_gb:.2f} GB")
    if profile.gpus:
        for ix, gpu in enumerate(profile.gpus):
            print(
                f"  GPU {ix}: {gpu.name}, {gpu.memory_total_mb} MiB, "
                f"{gpu.temperature_c}C, driver {gpu.driver_version}"
            )
    else:
        print("  GPU: no NVIDIA GPU detected by nvidia-smi")
    print(f"  torch: installed={profile.torch.installed}, version={profile.torch.version}")
    print(f"  torch cuda: {profile.torch.cuda_available}, cuda={profile.torch.cuda_version}")
    print("Max-safe defaults")
    print(f"  cpu_workers: {profile.max_safe.cpu_workers}")
    print(f"  dataloader_workers: {profile.max_safe.dataloader_workers}")
    print(f"  RAM limit: {profile.max_safe.ram_limit_fraction:.0%}")
    print(f"  RAM target: {profile.max_safe.ram_target_gb:.2f} GB")
    print(f"  GPU VRAM target: {profile.max_safe.gpu_target_vram_gb} GB")
    print(f"  GPU temp pause/abort: {profile.max_safe.gpu_pause_temp_c}C/{profile.max_safe.gpu_abort_temp_c}C")

    if not args.no_write:
        atomic_write_json(args.output, payload)
        print(f"Profile written to {args.output}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
