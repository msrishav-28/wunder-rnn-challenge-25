"""Deterministic runtime helpers for Phase 2 experiments."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int, deterministic_torch: bool = True, seed_torch: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch when available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if not seed_torch:
        return

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:
        return


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader workers from PyTorch's worker seed."""
    try:
        import torch

        worker_seed = torch.initial_seed() % 2**32
    except ImportError:
        worker_seed = worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed: int) -> Optional[object]:
    """Return a seeded torch.Generator, or None if torch is unavailable."""
    try:
        import torch

        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator
    except ImportError:
        return None
