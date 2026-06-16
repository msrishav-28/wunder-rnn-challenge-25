import numpy as np

from src.data.dataset import CausalStepDataset


def test_causal_step_dataset_contains_only_prefix(sample_dataframe, temp_dir):
    parquet_path = temp_dir / "sample.parquet"
    sample_dataframe.to_parquet(parquet_path, index=False)
    dataset = CausalStepDataset(str(parquet_path), seq_ids=[0], lookback=8)

    first = dataset[0]
    assert first["step_in_seq"].item() == 100
    assert first["history"].shape == (8, 32)
    assert first["history_mask"].sum().item() == 8

    seq0 = sample_dataframe[sample_dataframe["seq_ix"] == 0].sort_values("step_in_seq")
    feature_cols = [c for c in seq0.columns[3:]]
    state_100 = seq0.iloc[100][feature_cols].to_numpy(dtype=np.float32)
    state_101 = seq0.iloc[101][feature_cols].to_numpy(dtype=np.float32)

    assert np.allclose(first["history"][-1].tolist(), state_100)
    assert np.allclose(first["target"].tolist(), state_101)
    assert not np.allclose(first["history"][-1].tolist(), first["target"].tolist())


def test_causal_step_dataset_drops_unlabeled_final_prediction(sample_dataframe, temp_dir):
    parquet_path = temp_dir / "sample.parquet"
    sample_dataframe.to_parquet(parquet_path, index=False)
    dataset = CausalStepDataset(str(parquet_path), seq_ids=[0], lookback=16)
    assert len(dataset) == 899
    assert dataset[-1]["step_in_seq"].item() == 998
