from src.utils.hardware import (
    GpuInfo,
    HardwareProfile,
    MaxSafeConfig,
    MemoryInfo,
    ResourceLimitError,
    TorchInfo,
    assert_within_max_safe,
    build_max_safe_config,
    calibrate_batch_size,
)


def test_max_safe_config_caps_workers_and_vram():
    cfg = build_max_safe_config(
        MemoryInfo(total_gb=16.0, available_gb=8.0, used_fraction=0.5),
        [GpuInfo("RTX 3050", 4096, 0, 50, "test")],
    )
    assert cfg.cpu_workers <= 12
    assert cfg.dataloader_workers == 2
    assert cfg.max_dataloader_workers == 4
    assert cfg.ram_limit_fraction == 0.90
    assert cfg.gpu_target_vram_gb == 3.4
    assert cfg.gpu_abort_temp_c == 87


def test_batch_calibration_backs_off_after_oom():
    def trial(batch_size):
        if batch_size > 64:
            raise RuntimeError("CUDA out of memory")

    selected = calibrate_batch_size(trial, [512, 256, 128, 64, 32], repeats=2)
    assert selected == 64


def test_resource_guard_aborts_on_hot_gpu():
    profile = HardwareProfile(
        profile_name="test",
        platform="test",
        python="3.11",
        cpu_name="cpu",
        logical_cpu_count=16,
        memory=MemoryInfo(total_gb=16.0, available_gb=8.0, used_fraction=0.5),
        gpus=[GpuInfo("RTX 3050", 4096, 0, 90, "test")],
        torch=TorchInfo(True, "test", False, None, None, None),
        max_safe=MaxSafeConfig(
            cpu_workers=12,
            dataloader_workers=2,
            max_dataloader_workers=4,
            ram_limit_fraction=0.90,
            ram_target_gb=13.0,
            gpu_vram_fraction=0.85,
            gpu_target_vram_gb=3.4,
            gpu_pause_temp_c=82,
            gpu_abort_temp_c=87,
            mixed_precision=True,
            gradient_accumulation=True,
            batch_candidates=[512, 256, 128, 64, 32, 16, 8],
        ),
    )
    try:
        assert_within_max_safe(profile)
    except ResourceLimitError as exc:
        assert "temperature" in str(exc)
    else:
        raise AssertionError("hot GPU should trip max-safe guard")
