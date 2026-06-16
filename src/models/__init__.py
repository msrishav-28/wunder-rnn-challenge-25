"""Model package exports (lazy to keep torch optional for tabular-only use)."""

import importlib

__all__ = [
    'PersistencePredictionModel',
    'MomentumPredictionModel',
    'EWMAPredictionModel',
    'TabularStatefulPredictionModel',
    'build_causal_tabular_features',
    'CausalGRUForecaster',
    'CausalTCN',
    'EnsemblePredictionModel',
]


def __getattr__(name):
    if name in {
        'PersistencePredictionModel',
        'MomentumPredictionModel',
        'EWMAPredictionModel',
        'TabularStatefulPredictionModel',
        'build_causal_tabular_features',
    }:
        return getattr(importlib.import_module(".baselines", __name__), name)
    if name == 'CausalGRUForecaster':
        return getattr(importlib.import_module(".sequence_models", __name__), name)
    if name == 'CausalTCN':
        return getattr(importlib.import_module(".tcn", __name__), name)
    if name == 'EnsemblePredictionModel':
        return getattr(importlib.import_module(".ensemble_predictor", __name__), name)
    raise AttributeError(f"module 'src.models' has no attribute {name!r}")
