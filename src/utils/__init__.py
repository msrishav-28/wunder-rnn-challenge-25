from .metrics import compute_manual_r2_score, compute_r2_per_feature, compute_r2_score
from .reproducibility import make_torch_generator, seed_worker, set_global_seed
from .hardware import (
    HardwareProfile,
    MaxSafeConfig,
    ResourceLimitError,
    assert_within_max_safe,
    build_hardware_profile,
    calibrate_batch_size,
)

__all__ = [
    'compute_r2_score',
    'compute_r2_per_feature',
    'compute_manual_r2_score',
    'set_global_seed',
    'seed_worker',
    'make_torch_generator',
    'HardwareProfile',
    'MaxSafeConfig',
    'ResourceLimitError',
    'build_hardware_profile',
    'assert_within_max_safe',
    'calibrate_batch_size',
]
