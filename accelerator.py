"""Minimal accelerator detection shared by GradMesh worker and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Accelerator:
    backend: str
    torch_device: torch.device
    device_name: str
    total_memory_mb: int
    ultralytics_device: str | torch.device

    @property
    def supports_training(self) -> bool:
        return self.backend in {"cuda", "xpu"}

    def synchronize(self) -> None:
        if self.backend == "cuda":
            torch.cuda.synchronize(self.torch_device)
        elif self.backend == "xpu":
            torch.xpu.synchronize(self.torch_device)


def _cuda_accelerator() -> Accelerator:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(0)
    return Accelerator(
        backend="cuda",
        torch_device=device,
        device_name=torch.cuda.get_device_name(0),
        total_memory_mb=int(properties.total_memory / (1024 * 1024)),
        # Preserve the worker's existing Ultralytics CUDA argument.
        ultralytics_device="cuda",
    )


def _xpu_accelerator() -> Accelerator:
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("XPU was requested but torch.xpu.is_available() is False")
    if torch.xpu.device_count() < 1:
        raise RuntimeError("XPU was requested but no XPU devices were found")
    device = torch.device("xpu:0")
    properties = torch.xpu.get_device_properties(0)
    return Accelerator(
        backend="xpu",
        torch_device=device,
        device_name=torch.xpu.get_device_name(0),
        total_memory_mb=int(properties.total_memory / (1024 * 1024)),
        # Ultralytics 8.4.46 rejects XPU strings but accepts torch.device.
        ultralytics_device=device,
    )


def _cpu_accelerator(advertised_memory_mb: int) -> Accelerator:
    device = torch.device("cpu")
    return Accelerator(
        backend="cpu",
        torch_device=device,
        device_name="cpu",
        total_memory_mb=advertised_memory_mb,
        ultralytics_device="cpu",
    )


def detect_accelerator(requested: str = "auto", advertised_memory_mb: int = 24000) -> Accelerator:
    """Detect CUDA, XPU, or CPU, failing rather than falling back for explicit requests."""
    backend = requested.lower().strip()
    if backend not in {"auto", "cuda", "xpu", "cpu"}:
        raise ValueError(f"Unsupported backend '{requested}'; choose auto, cuda, xpu, or cpu")
    if backend == "cuda":
        return _cuda_accelerator()
    if backend == "xpu":
        return _xpu_accelerator()
    if backend == "cpu":
        return _cpu_accelerator(advertised_memory_mb)
    if torch.cuda.is_available():
        return _cuda_accelerator()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return _xpu_accelerator()
    return _cpu_accelerator(advertised_memory_mb)

