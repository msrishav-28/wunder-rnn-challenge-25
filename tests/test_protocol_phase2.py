import numpy as np
import pandas as pd
import pytest

from src.data.protocol import (
    create_dataset_manifest,
    create_sequence_folds,
    validate_wunder_dataframe,
)


def test_validate_wunder_dataframe_preserves_feature_order(sample_dataframe):
    feature_cols = validate_wunder_dataframe(sample_dataframe)
    assert feature_cols[:3] == ["feature_0", "feature_1", "feature_2"]
    assert feature_cols[-1] == "feature_31"


def test_validate_accepts_official_numeric_feature_names(sample_dataframe):
    renamed = sample_dataframe.rename(columns={f"feature_{i}": str(i) for i in range(32)})
    feature_cols = validate_wunder_dataframe(renamed)
    assert feature_cols == [str(i) for i in range(32)]


def test_validate_rejects_missing_feature(sample_dataframe):
    bad_df = sample_dataframe.drop(columns=["feature_31"])
    with pytest.raises(ValueError, match="Expected 32 feature"):
        validate_wunder_dataframe(bad_df)


def test_validate_rejects_incomplete_sequence(sample_dataframe):
    bad_df = sample_dataframe.iloc[:-1].copy()
    with pytest.raises(ValueError, match="exactly 1000"):
        validate_wunder_dataframe(bad_df)


def test_dataset_manifest_and_folds(sample_dataframe, temp_dir):
    parquet_path = temp_dir / "train.parquet"
    sample_dataframe.to_parquet(parquet_path, index=False)
    manifest = create_dataset_manifest(parquet_path)
    assert manifest.row_count == 5000
    assert manifest.sequence_count == 5
    assert len(manifest.sha256) == 64

    folds = create_sequence_folds(range(5), n_folds=5, seed=42, dataset_sha256=manifest.sha256)
    all_fold_ids = sorted(seq for fold in folds.folds.values() for seq in fold)
    assert all_fold_ids == [0, 1, 2, 3, 4]
    assert set(folds.train_dev_seq_ids).isdisjoint(folds.final_holdout_seq_ids)
