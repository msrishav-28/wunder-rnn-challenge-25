"""Hardware profiling and max-safe guard helpers for local-only experiments."""

from __future__ import annotations

import ctypes
import contextlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional


BYTES_PER_GB = 1024**3


@dataclass(frozen=True)
class MemoryInfo:
    total_gb: float
    available_gb: Optional[float]
    used_fraction: Optional[float]


@dataclass(frozen=True)
class GpuInfo:
    name: str
    memory_total_mb: int
    memory_used_mb: Optional[int]
    temperature_c: Optional[int]
    driver_version: Optional[str]


@dataclass(frozen=True)
class TorchInfo:
    installed: bool
    version: Optional[str]
    cuda_available: bool
    cuda_version: Optional[str]
    device_name: Optional[str]
    device_memory_gb: Optional[float]
    error: Optional[str] = None


@dataclass(frozen=True)
class MaxSafeConfig:
    cpu_workers: int
    dataloader_workers: int
    max_dataloader_workers: int
    ram_limit_fraction: float
    ram_target_gb: float
    gpu_vram_fraction: float
    gpu_target_vram_gb: Optional[float]
    gpu_pause_temp_c: int
    gpu_abort_temp_c: int
    mixed_precision: bool
    gradient_accumulation: bool
    batch_candidates: list[int]


@dataclass(frozen=True)
class HardwareProfile:
    profile_name: str
    platform: str
    python: str
    cpu_name: str
    logical_cpu_count: int
    memory: MemoryInfo
    gpus: list[GpuInfo]
    torch: TorchInfo
    max_safe: MaxSafeConfig


class ResourceLimitError(RuntimeError):
    """Raised when a max-safe guard would abort a run."""


def _round_gb(value_bytes: int) -> float:
    return round(value_bytes / BYTES_PER_GB, 2)


def get_memory_info() -> MemoryInfo:
    """Return physical memory using stdlib-only platform APIs."""
    if platform.system().lower() == "windows":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        used_fraction = 1.0 - (status.ullAvailPhys / status.ullTotalPhys)
        return MemoryInfo(
            total_gb=_round_gb(status.ullTotalPhys),
            available_gb=_round_gb(status.ullAvailPhys),
            used_fraction=round(used_fraction, 4),
        )

    if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return MemoryInfo(total_gb=_round_gb(page_size * pages), available_gb=None, used_fraction=None)

    return MemoryInfo(total_gb=0.0, available_gb=None, used_fraction=None)


def get_nvidia_gpus() -> list[GpuInfo]:
    """Read NVIDIA GPU status through nvidia-smi when available."""
    if not shutil.which("nvidia-smi"):
        return []
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,temperature.gpu,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        raw = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=10)
    except Exception:
        return []

    gpus = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        name, total_mb, used_mb, temp_c, driver = parts[:5]
        gpus.append(
            GpuInfo(
                name=name,
                memory_total_mb=int(float(total_mb)),
                memory_used_mb=int(float(used_mb)),
                temperature_c=int(float(temp_c)),
                driver_version=driver,
            )
        )
    return gpus


def get_torch_info() -> TorchInfo:
    """Return PyTorch/CUDA availability without making Torch mandatory."""
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stderr(devnull):
            import torch

        cuda_available = bool(torch.cuda.is_available())
        device_name = None
        device_memory_gb = None
        if cuda_available:
            props = torch.cuda.get_device_properties(0)
            device_name = torch.cuda.get_device_name(0)
            device_memory_gb = round(props.total_memory / BYTES_PER_GB, 2)
        return TorchInfo(
            installed=True,
            version=str(torch.__version__),
            cuda_available=cuda_available,
            cuda_version=getattr(torch.version, "cuda", None),
            device_name=device_name,
            device_memory_gb=device_memory_gb,
        )
    except Exception as exc:
        return TorchInfo(
            installed=False,
            version=None,
            cuda_available=False,
            cuda_version=None,
            device_name=None,
            device_memory_gb=None,
            error=repr(exc),
        )


def build_max_safe_config(memory: MemoryInfo, gpus: list[GpuInfo]) -> MaxSafeConfig:
    """Choose conservative max-safe defaults for this laptop class."""
    logical = os.cpu_count() or 1
    cpu_workers = max(1, min(12, logical - 4 if logical > 4 else logical))
    gpu_target_vram_gb = None
    if gpus:
        gpu_target_vram_gb = round((gpus[0].memory_total_mb / 1024) * 0.85, 2)
    return MaxSafeConfig(
        cpu_workers=cpu_workers,
        dataloader_workers=2,
        max_dataloader_workers=4,
        ram_limit_fraction=0.90,
        ram_target_gb=min(13.0, round(memory.total_gb * 0.82, 2)) if memory.total_gb else 13.0,
        gpu_vram_fraction=0.85,
        gpu_target_vram_gb=gpu_target_vram_gb,
        gpu_pause_temp_c=82,
        gpu_abort_temp_c=87,
        mixed_precision=True,
        gradient_accumulation=True,
        batch_candidates=[512, 256, 128, 64, 32, 16, 8],
    )


def build_hardware_profile(profile_name: str = "max_safe") -> HardwareProfile:
    memory = get_memory_info()
    gpus = get_nvidia_gpus()
    return HardwareProfile(
        profile_name=profile_name,
        platform=platform.platform(),
        python=platform.python_version(),
        cpu_name=platform.processor() or platform.machine(),
        logical_cpu_count=os.cpu_count() or 1,
        memory=memory,
        gpus=gpus,
        torch=get_torch_info(),
        max_safe=build_max_safe_config(memory, gpus),
    )


def assert_within_max_safe(profile: HardwareProfile) -> None:
    """Abort before a run if current memory/temperature already violates guardrails."""
    if profile.memory.used_fraction is not None and profile.memory.used_fraction >= profile.max_safe.ram_limit_fraction:
        raise ResourceLimitError(
            f"System RAM usage {profile.memory.used_fraction:.1%} exceeds "
            f"{profile.max_safe.ram_limit_fraction:.0%} guard"
        )
    if profile.gpus:
        gpu = profile.gpus[0]
        if gpu.temperature_c is not None and gpu.temperature_c >= profile.max_safe.gpu_abort_temp_c:
            raise ResourceLimitError(
                f"GPU temperature {gpu.temperature_c}C exceeds "
                f"{profile.max_safe.gpu_abort_temp_c}C abort guard"
            )


def calibrate_batch_size(
    trial_fn: Callable[[int], None],
    candidates: Iterable[int],
    *,
    repeats: int = 2,
) -> Optional[int]:
    """
    Choose the largest candidate batch that survives repeated trial calls.

    `trial_fn` should raise RuntimeError for OOM-like failures. This helper is
    intentionally framework-agnostic so tests can simulate OOM without Torch.
    """
    for batch_size in candidates:
        try:
            for _ in range(repeats):
                trial_fn(int(batch_size))
            return int(batch_size)
        except RuntimeError:
            continue
    return None


def atomic_write_json(path: str | Path, payload: object) -> None:
    """Write JSON atomically so interrupted runs do not corrupt manifests."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "__dataclass_fields__"):
        payload = asdict(payload)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def dataclass_to_dict(value: object) -> dict:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, dict):
        return value
    raise TypeError(f"Unsupported JSON payload: {type(value)}")
