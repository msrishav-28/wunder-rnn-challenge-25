"""Data package exports (lazy loading for the Torch-dependent dataset)."""

import importlib

from .protocol import create_dataset_manifest, create_sequence_folds, resolve_dataset_path, validate_wunder_dataframe
from .causal_features import (
    build_causal_tabular_features,
    feature_schema_hash,
    feature_schema_payload,
    feature_width,
)

__all__ = [
    'CausalStepDataset',
    'create_causal_dataloaders',
    'create_dataset_manifest',
    'create_sequence_folds',
    'resolve_dataset_path',
    'validate_wunder_dataframe',
    'build_causal_tabular_features',
    'feature_schema_hash',
    'feature_schema_payload',
    'feature_width',
]


def __getattr__(name):
    if name in {'CausalStepDataset', 'create_causal_dataloaders'}:
        return getattr(importlib.import_module(".dataset", __name__), name)
    raise AttributeError(f"module 'src.data' has no attribute {name!r}")
