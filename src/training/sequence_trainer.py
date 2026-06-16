"""Full-sequence BPTT trainer for the causal GRU next-state model.

Trains on whole 1000-step sequences, computing a masked MSE loss only on the
scored steps (need_prediction True, i.e. current steps 100..998 -> targets
101..999). Validation R2 is computed with a batched full-sequence forward,
which is numerically identical to the stepwise stateful replay because a GRU
is a pure recurrence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.data.protocol import get_feature_columns, load_wunder_dataframe
from src.models.sequence_models import CausalGRUForecaster, count_parameters
from src.models.tcn import CausalTCN
from src.utils.metrics import compute_r2_per_feature, compute_r2_score
from src.utils.reproducibility import set_global_seed


def load_full_sequences(parquet_path: str, seq_ids: list[int]):
    """Return (states, need, seq_ids) as dense arrays.

    states: (N, 1000, 32) float32
    need:   (N, 1000) bool   (need_prediction flag per step)
    """
    df = load_wunder_dataframe(parquet_path, seq_ids=seq_ids)
    cols = get_feature_columns(df)
    states_list, need_list, ids = [], [], []
    for sid, g in df.sort_values(["seq_ix", "step_in_seq"]).groupby("seq_ix", sort=True):
        states_list.append(g[cols].to_numpy(dtype=np.float32))
        need_list.append(g["need_prediction"].to_numpy(dtype=bool))
        ids.append(int(sid))
    states = np.stack(states_list).astype(np.float32)
    need = np.stack(need_list)
    return states, need, ids


@dataclass
class TrainConfig:
    d_model: int = 256
    n_layers: int = 2
    dropout: float = 0.1
    head_hidden: Optional[int] = None
    epochs: int = 40
    batch_size: int = 32
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    warmup_frac: float = 0.1
    seed: int = 42
    threads: int = 8
    patience: int = 12
    n_features: int = 32
    device: str = "cpu"
    rnn_type: str = "gru"
    arch: str = "rnn"        # "rnn" or "tcn"
    kernel_size: int = 3     # tcn only


def _masked_mse(preds: torch.Tensor, states: torch.Tensor, need: torch.Tensor) -> torch.Tensor:
    # preds[:, t] predicts states[:, t+1]
    pred = preds[:, :-1, :]
    target = states[:, 1:, :]
    mask = need[:, :-1]  # (B, T-1)
    diff2 = (pred - target) ** 2  # (B, T-1, F)
    m = mask.unsqueeze(-1).to(diff2.dtype)
    return (diff2 * m).sum() / (m.sum() * preds.shape[-1])


@torch.no_grad()
def _eval_r2(model, states_t, need_t, feature_cols, batch_size=64):
    model.eval()
    preds_all, tgts_all = [], []
    n = states_t.shape[0]
    for i in range(0, n, batch_size):
        sb = states_t[i : i + batch_size]
        nb = need_t[i : i + batch_size]
        preds, _ = model(sb)
        pred = preds[:, :-1, :]
        target = sb[:, 1:, :]
        mask = nb[:, :-1]
        preds_all.append(pred[mask].cpu().numpy())
        tgts_all.append(target[mask].cpu().numpy())
    y_pred = np.concatenate(preds_all).astype(np.float64)
    y_true = np.concatenate(tgts_all).astype(np.float64)
    mean_r2 = compute_r2_score(y_true, y_pred)
    per_feat = compute_r2_per_feature(y_true, y_pred, feature_cols)
    return mean_r2, per_feat, y_true, y_pred


def train_sequence_model(
    data_path: str,
    train_ids: list[int],
    val_ids: list[int],
    cfg: TrainConfig,
    feature_cols: Optional[list[str]] = None,
    log_every: int = 1,
    verbose: bool = True,
):
    set_global_seed(cfg.seed, deterministic_torch=True, seed_torch=True)
    torch.set_num_threads(int(cfg.threads))
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if str(device) != cfg.device:
        print(f"requested device '{cfg.device}' unavailable; using {device}")

    s_tr, n_tr, _ = load_full_sequences(data_path, train_ids)
    s_va, n_va, va_ids = load_full_sequences(data_path, val_ids)
    if feature_cols is None:
        feature_cols = [str(i) for i in range(cfg.n_features)]

    states_tr = torch.from_numpy(s_tr).to(device)
    need_tr = torch.from_numpy(n_tr).to(device)
    states_va = torch.from_numpy(s_va).to(device)
    need_va = torch.from_numpy(n_va).to(device)

    if cfg.arch == "tcn":
        model = CausalTCN(
            n_features=cfg.n_features, d_model=cfg.d_model, n_layers=cfg.n_layers,
            kernel_size=cfg.kernel_size, dropout=cfg.dropout, head_hidden=cfg.head_hidden,
        ).to(device)
    else:
        model = CausalGRUForecaster(
            n_features=cfg.n_features, d_model=cfg.d_model, n_layers=cfg.n_layers,
            dropout=cfg.dropout, head_hidden=cfg.head_hidden, rnn_type=cfg.rnn_type,
        ).to(device)
    if verbose:
        print(f"model params: {count_parameters(model):,}  train_seqs={len(train_ids)} val_seqs={len(val_ids)}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n_batches = max(1, (states_tr.shape[0] + cfg.batch_size - 1) // cfg.batch_size)
    total_steps = cfg.epochs * n_batches
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=total_steps,
        pct_start=cfg.warmup_frac, anneal_strategy="cos",
    )

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    best_r2 = -1e9
    best_state = None
    best_per_feat = None
    best_oof = None
    history = []
    bad_epochs = 0

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(states_tr.shape[0], generator=g)
        ep_loss = 0.0
        t0 = time.perf_counter()
        for bi in range(n_batches):
            idx = perm[bi * cfg.batch_size : (bi + 1) * cfg.batch_size]
            sb = states_tr[idx]
            nb = need_tr[idx]
            opt.zero_grad(set_to_none=True)
            preds, _ = model(sb)
            loss = _masked_mse(preds, sb, nb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            ep_loss += float(loss.detach())
        ep_loss /= n_batches

        val_r2, per_feat, y_true, y_pred = _eval_r2(model, states_va, need_va, feature_cols)
        dt = time.perf_counter() - t0
        history.append({"epoch": epoch, "train_loss": ep_loss, "val_r2": val_r2, "sec": dt})
        if verbose and (epoch % log_every == 0):
            print(f"epoch {epoch:3d}  loss={ep_loss:.5f}  val_R2={val_r2:.5f}  lr={sched.get_last_lr()[0]:.2e}  {dt:.1f}s")

        if val_r2 > best_r2:
            best_r2 = val_r2
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_per_feat = per_feat
            best_oof = (y_true, y_pred)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                if verbose:
                    print(f"early stop at epoch {epoch} (best val_R2={best_r2:.5f})")
                break

    return {
        "best_val_r2": best_r2,
        "best_per_feature": best_per_feat,
        "best_state_dict": best_state,
        "history": history,
        "oof": best_oof,
        "val_seq_ids": va_ids,
        "feature_cols": feature_cols,
    }
