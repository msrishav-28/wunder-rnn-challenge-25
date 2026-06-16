"""
Pytest configuration and shared fixtures.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture
def device():
    """Get device (CPU for tests)."""
    return 'cpu'


@pytest.fixture
def sample_sequence():
    """Generate a sample sequence for testing."""
    return {
        'seq_ix': 0,
        'states': np.random.randn(1000, 32).astype(np.float32),
        'targets': np.random.randn(1000, 32).astype(np.float32),
        'need_prediction': np.array([False] * 100 + [True] * 900),
    }


@pytest.fixture
def sample_batch():
    """Generate a sample batch of tensors."""
    batch_size = 4
    return {
        'seq_ix': torch.arange(batch_size),
        'states': torch.randn(batch_size, 1000, 32),
        'targets': torch.randn(batch_size, 1000, 32),
        'need_prediction': torch.ones(batch_size, 1000, dtype=torch.bool),
    }


@pytest.fixture
def sample_dataframe():
    """Generate a sample dataframe with multiple sequences."""
    np.random.seed(42)
    data = []
    for seq_id in range(5):
        for step in range(1000):
            row = {
                'seq_ix': seq_id,
                'step_in_seq': step,
                'need_prediction': step >= 100,
            }
            for i in range(32):
                row[f'feature_{i}'] = np.random.randn()
            data.append(row)
    return pd.DataFrame(data)


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def small_model_config():
    """Configuration for a small model (fast testing)."""
    return {
        'n_features': 32,
        'd_model': 64,
        'd_state': 16,
        'n_layers': 1,
        'patch_size': 16,
        'dropout': 0.1,
    }
