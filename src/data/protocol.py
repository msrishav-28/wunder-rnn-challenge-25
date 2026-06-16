"""Official-protocol checks, dataset manifests, and locked sequence folds."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


REQUIRED_META_COLUMNS = ["seq_ix", "step_in_seq", "need_prediction"]
EXPECTED_N_FEATURES = 32
EXPECTED_SEQUENCE_LENGTH = 1000


@dataclass(frozen=True)
class DatasetManifest:
    path: str
    sha256: str
    size_bytes: int
    row_count: int
    sequence_count: int
    sequence_length: int
    feature_columns: list[str]
    need_prediction_counts_by_step: dict[str, int]
    seq_ix_min: int
    seq_ix_max: int


@dataclass(frozen=True)
class FoldManifest:
    seed: int
    n_folds: int
    final_holdout_fold: int
    folds: dict[str, list[int]]
    train_dev_seq_ids: list[int]
    final_holdout_seq_ids: list[int]
    dataset_sha256: Optional[str] = None
    git_commit: Optional[str] = None


def resolve_dataset_path(preferred: str | Path = "data/raw/train.parquet") -> Path:
    """Find the canonical raw train parquet without mutating the workspace."""
    candidates = [
        Path(preferred),
        Path("data/raw/train.parquet"),
        Path("data/train.parquet"),
        Path("competition_package/datasets/train.parquet"),
        Path("wnn_starterpack/competition_package/datasets/train.parquet"),
        Path("datasets/train.parquet"),
        Path("train.parquet"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No train parquet found. Expected data/raw/train.parquet, "
        "data/train.parquet, or competition_package/datasets/train.parquet."
    )


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature columns in scorer/parquet order, never lexicographic sort."""
    return list(df.columns[len(REQUIRED_META_COLUMNS) :])


def validate_wunder_dataframe(
    df: pd.DataFrame,
    *,
    expected_n_features: int = EXPECTED_N_FEATURES,
    expected_sequence_length: int = EXPECTED_SEQUENCE_LENGTH,
) -> list[str]:
    """Validate the official data contract and return feature columns."""
    if list(df.columns[:3]) != REQUIRED_META_COLUMNS:
        raise ValueError(
            "First columns must be seq_ix, step_in_seq, need_prediction; "
            f"got {list(df.columns[:3])}"
        )

    feature_cols = get_feature_columns(df)
    if len(feature_cols) != expected_n_features:
        raise ValueError(
            f"Expected {expected_n_features} feature columns, got {len(feature_cols)}"
        )
    if df[REQUIRED_META_COLUMNS].isna().any().any():
        raise ValueError("Metadata columns contain missing values")

    lengths = df.groupby("seq_ix", sort=True).size()
    bad_lengths = lengths[lengths != expected_sequence_length]
    if not bad_lengths.empty:
        raise ValueError(
            "Every sequence must have exactly "
            f"{expected_sequence_length} rows; bad seq_ix values: "
            f"{bad_lengths.index.tolist()[:10]}"
        )

    expected_steps = np.arange(expected_sequence_length)
    for seq_ix, seq_df in df.groupby("seq_ix", sort=False):
        steps = seq_df["step_in_seq"].to_numpy()
        if not np.array_equal(steps, expected_steps):
            raise ValueError(
                f"Sequence {seq_ix} must be ordered with steps 0.."
                f"{expected_sequence_length - 1}"
            )

    if df[feature_cols].isna().any().any():
        raise ValueError("Feature columns contain missing values")
    return feature_cols


def load_wunder_dataframe(path: str | Path, seq_ids: Optional[Iterable[int]] = None) -> pd.DataFrame:
    """Read and validate the Wunder parquet file."""
    df = pd.read_parquet(path)
    if seq_ids is not None:
        seq_set = set(int(seq_id) for seq_id in seq_ids)
        df = df[df["seq_ix"].isin(seq_set)].copy()
    df = df.sort_values(["seq_ix", "step_in_seq"]).reset_index(drop=True)
    validate_wunder_dataframe(df)
    return df


def create_dataset_manifest(path: str | Path) -> DatasetManifest:
    """Build a reproducibility manifest for a validated parquet file."""
    path = Path(path)
    df = load_wunder_dataframe(path)
    feature_cols = get_feature_columns(df)
    need_counts = (
        df.groupby("step_in_seq")["need_prediction"]
        .sum()
        .astype(int)
        .to_dict()
    )
    seq_ids = sorted(int(x) for x in df["seq_ix"].unique())
    return DatasetManifest(
        path=str(path),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        row_count=int(len(df)),
        sequence_count=len(seq_ids),
        sequence_length=int(df.groupby("seq_ix").size().iloc[0]),
        feature_columns=feature_cols,
        need_prediction_counts_by_step={str(k): int(v) for k, v in need_counts.items()},
        seq_ix_min=min(seq_ids),
        seq_ix_max=max(seq_ids),
    )


def write_json(path: str | Path, payload: object) -> None:
    """Write dataclasses/dicts as stable JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "__dataclass_fields__"):
        payload = asdict(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def current_git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def create_sequence_folds(
    seq_ids: Iterable[int],
    *,
    n_folds: int = 5,
    seed: int = 42,
    final_holdout_fold: int = 4,
    dataset_sha256: Optional[str] = None,
) -> FoldManifest:
    """Create deterministic folds by seq_ix with one untouched final holdout fold."""
    seq_ids = np.array(sorted(int(x) for x in seq_ids), dtype=int)
    if len(seq_ids) < n_folds:
        raise ValueError("Need at least as many sequences as folds")
    if final_holdout_fold < 0 or final_holdout_fold >= n_folds:
        raise ValueError("final_holdout_fold must be in [0, n_folds)")

    rng = np.random.RandomState(seed)
    shuffled = seq_ids.copy()
    rng.shuffle(shuffled)
    split_ids = np.array_split(shuffled, n_folds)
    folds = {
        str(i): sorted(int(x) for x in split_ids[i].tolist())
        for i in range(n_folds)
    }
    final_holdout = folds[str(final_holdout_fold)]
    train_dev = sorted(
        int(seq_id)
        for fold_id, fold_seq_ids in folds.items()
        if int(fold_id) != final_holdout_fold
        for seq_id in fold_seq_ids
    )
    return FoldManifest(
        seed=seed,
        n_folds=n_folds,
        final_holdout_fold=final_holdout_fold,
        folds=folds,
        train_dev_seq_ids=train_dev,
        final_holdout_seq_ids=final_holdout,
        dataset_sha256=dataset_sha256,
        git_commit=current_git_commit(),
    )


def load_sequence_folds(path: str | Path) -> FoldManifest:
    payload = read_json(path)
    return FoldManifest(**payload)
