"""Training package exports (lazy loading for the Torch-dependent trainer)."""

import importlib

__all__ = [
    'build_tabular_supervised_matrix',
    'create_tabular_estimator',
    'TrainConfig',
    'train_sequence_model',
]


def __getattr__(name):
    if name in {'build_tabular_supervised_matrix', 'create_tabular_estimator'}:
        return getattr(importlib.import_module(".baseline_training", __name__), name)
    if name in {'TrainConfig', 'train_sequence_model'}:
        return getattr(importlib.import_module(".sequence_trainer", __name__), name)
    raise AttributeError(f"module 'src.training' has no attribute {name!r}")
