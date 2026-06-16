#!/usr/bin/env python3
"""Diagnostic analysis of the Wunder train.parquet.

Answers the questions that drive modeling choices:
  - Global per-feature scale / tails / NaNs.
  - Persistence (predict current) vs diff structure.
  - Linear ceiling: VAR(p) Ridge for several p, seq-grouped split.
  - Marginal value of history (does memory help linearly?).

Pure tabular/numpy/sklearn — does NOT import torch, so it runs on the
global interpreter even while the project venv is still installing.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent.parent


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")]


def r2_per_feature(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.where(ss_tot == 0, 1, ss_tot)


def build_var_matrix(states_by_seq, need_by_seq, p: int):
    """Build VAR(p) design: X = last p states concatenated, y = next state.

    Only prediction points (need_prediction True) with an available next row
    are emitted, matching the official stepwise scoring convention.
    """
    X, Y = [], []
    for states, need in zip(states_by_seq, need_by_seq):
        n = len(states)
        for pos in range(n - 1):
            if not need[pos]:
                continue
            if pos - (p - 1) < 0:
                continue  # need p history rows
            window = states[pos - p + 1 : pos + 1]  # (p, 32)
            X.append(window.reshape(-1))
            Y.append(states[pos + 1])
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def main():
    t0 = time.perf_counter()
    data_path = ROOT / "data" / "raw" / "train.parquet"
    df = pd.read_parquet(data_path)
    cols = feature_cols(df)
    print(f"Loaded {data_path}")
    print(f"shape={df.shape}  n_features={len(cols)}  cols[:5]={cols[:5]}")
    print(f"seqs={df['seq_ix'].nunique()}  steps/seq={df.groupby('seq_ix').size().unique()}")
    print(f"need_prediction True count={int(df['need_prediction'].sum())}")
    feat = df[cols].to_numpy(dtype=np.float64)
    print(f"NaNs={np.isnan(feat).sum()}  Infs={np.isinf(feat).sum()}")
    print(f"global mean={feat.mean():.4f} std={feat.std():.4f} "
          f"min={feat.min():.3f} max={feat.max():.3f}")
    # tails: excess kurtosis per feature
    z = (feat - feat.mean(0)) / feat.std(0)
    kurt = (z**4).mean(0) - 3.0
    print(f"per-feature std range=[{feat.std(0).min():.3f},{feat.std(0).max():.3f}]  "
          f"excess-kurtosis range=[{kurt.min():.1f},{kurt.max():.1f}] median={np.median(kurt):.1f}")

    # group by sequence (ordered)
    states_by_seq, need_by_seq, seq_ids = [], [], []
    for sid, g in df.sort_values(["seq_ix", "step_in_seq"]).groupby("seq_ix", sort=True):
        states_by_seq.append(g[cols].to_numpy(dtype=np.float32))
        need_by_seq.append(g["need_prediction"].to_numpy(dtype=bool))
        seq_ids.append(int(sid))
    seq_ids = np.array(seq_ids)

    # deterministic 80/20 seq-grouped split
    rng = np.random.RandomState(42)
    order = rng.permutation(len(seq_ids))
    n_val = len(seq_ids) // 5
    val_idx = set(order[:n_val].tolist())
    tr_mask = np.array([i not in val_idx for i in range(len(seq_ids))])

    # Persistence and diff structure on the val split (need points only)
    pers_true, pers_pred = [], []
    diff_var, lvl_var = [], []
    for i, (states, need) in enumerate(zip(states_by_seq, need_by_seq)):
        if tr_mask[i]:
            continue
        for pos in range(len(states) - 1):
            if not need[pos]:
                continue
            pers_true.append(states[pos + 1])
            pers_pred.append(states[pos])
    pers_true = np.asarray(pers_true); pers_pred = np.asarray(pers_pred)
    pr2 = r2_per_feature(pers_true, pers_pred)
    print(f"\n[persistence] mean R2={pr2.mean():.4f}  "
          f"var(next-cur)/var(next) median={np.median(((pers_true-pers_pred).var(0))/pers_true.var(0)):.3f}")

    # lag-1 autocorr per feature (pooled within seq)
    ac = []
    for states, need in zip(states_by_seq, need_by_seq):
        s = states
        ac.append([np.corrcoef(s[:-1, j], s[1:, j])[0, 1] for j in range(s.shape[1])])
    ac = np.nanmean(np.array(ac), axis=0)
    print(f"[lag-1 autocorr] mean={np.nanmean(ac):.3f} range=[{np.nanmin(ac):.3f},{np.nanmax(ac):.3f}]")

    # VAR(p) linear ceiling
    print("\n[VAR(p) Ridge, seq-grouped 80/20, alpha=10]")
    results = {}
    for p in (1, 2, 4, 8, 16):
        Xtr, Ytr = build_var_matrix(
            [s for i, s in enumerate(states_by_seq) if tr_mask[i]],
            [s for i, s in enumerate(need_by_seq) if tr_mask[i]], p)
        Xva, Yva = build_var_matrix(
            [s for i, s in enumerate(states_by_seq) if not tr_mask[i]],
            [s for i, s in enumerate(need_by_seq) if not tr_mask[i]], p)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        model = Ridge(alpha=10.0)
        model.fit((Xtr - mu) / sd, Ytr)
        pred = model.predict((Xva - mu) / sd)
        r2 = r2_per_feature(Yva, pred)
        results[p] = float(r2.mean())
        print(f"  p={p:2d}  Xdim={Xtr.shape[1]:4d}  ntr={Xtr.shape[0]:7d}  "
              f"mean R2={r2.mean():.4f}  per-feat[min,med,max]="
              f"[{r2.min():.3f},{np.median(r2):.3f},{r2.max():.3f}]")

    out = {
        "shape": list(df.shape),
        "persistence_mean_r2": float(pr2.mean()),
        "lag1_autocorr_mean": float(np.nanmean(ac)),
        "var_p_mean_r2": results,
        "kurtosis_median": float(np.median(kurt)),
    }
    outdir = ROOT / "outputs" / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "data_analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {outdir/'data_analysis.json'}  elapsed={time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
