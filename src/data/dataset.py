"""
Causal step dataset for the Wunder Fund RNN Challenge.

Provides CausalStepDataset (left-padded lookback windows for the official
stepwise task) and a deterministic train/val loader factory. Targets are
aligned so that the example at step t predicts state t+1, and only scored steps
(need_prediction=True, with an available next state) are emitted.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple

from src.data.protocol import get_feature_columns, load_wunder_dataframe
from src.utils.reproducibility import make_torch_generator, seed_worker


class CausalStepDataset(Dataset):
    """
    Causal online dataset for the official stepwise task.

    Each sample contains only states up to and including step t and predicts
    state t+1. Samples are emitted only where need_prediction=True and t+1 is
    available in the local parquet file. The hidden competition may request a
    final step prediction whose target is unavailable locally; this dataset
    deliberately excludes that row from training and local validation.
    """

    def __init__(
        self,
        parquet_path: str,
        seq_ids: Optional[List[int]] = None,
        lookback: int = 128,
    ):
        self.lookback = int(lookback)
        if self.lookback <= 0:
            raise ValueError("lookback must be positive")

        df = load_wunder_dataframe(parquet_path, seq_ids=seq_ids)
        self.feature_cols = get_feature_columns(df)
        self.n_features = len(self.feature_cols)
        self.samples = []

        for seq_ix, seq_df in df.groupby("seq_ix", sort=True):
            states = seq_df[self.feature_cols].to_numpy(dtype=np.float32)
            need_prediction = seq_df["need_prediction"].to_numpy(dtype=bool)
            steps = seq_df["step_in_seq"].to_numpy(dtype=np.int64)
            for pos, step in enumerate(steps):
                if not need_prediction[pos]:
                    continue
                target_pos = pos + 1
                if target_pos >= len(states):
                    continue
                self.samples.append(
                    {
                        "seq_ix": int(seq_ix),
                        "step_in_seq": int(step),
                        "states": states,
                        "position": int(pos),
                        "target": states[target_pos],
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        states = sample["states"]
        pos = sample["position"]
        start = max(0, pos + 1 - self.lookback)
        history = states[start : pos + 1]

        window = np.zeros((self.lookback, self.n_features), dtype=np.float32)
        mask = np.zeros((self.lookback,), dtype=np.float32)
        window[-len(history) :] = history
        mask[-len(history) :] = 1.0

        return {
            "seq_ix": torch.tensor(sample["seq_ix"], dtype=torch.long),
            "step_in_seq": torch.tensor(sample["step_in_seq"], dtype=torch.long),
            "history": torch.tensor(window, dtype=torch.float32),
            "history_mask": torch.tensor(mask, dtype=torch.float32),
            "target": torch.tensor(sample["target"].astype(np.float32, copy=False), dtype=torch.float32),
        }


def create_causal_dataloaders(
    parquet_path: str,
    train_seq_ids: list[int],
    val_seq_ids: list[int],
    *,
    lookback: int = 128,
    batch_size: int = 256,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create deterministic train/validation loaders for causal step examples."""
    train_dataset = CausalStepDataset(parquet_path, seq_ids=train_seq_ids, lookback=lookback)
    val_dataset = CausalStepDataset(parquet_path, seq_ids=val_seq_ids, lookback=lookback)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader
