from __future__ import annotations

import argparse
import sys
import time

import torch

from accelerator import detect_accelerator


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a GradMesh accelerator backend")
    parser.add_argument("--backend", choices=["auto", "cuda", "xpu", "cpu"], default="auto")
    args = parser.parse_args()

    accelerator = detect_accelerator(args.backend)
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Backend: {accelerator.backend}")
    print(f"Device: {accelerator.torch_device}")
    print(f"Device name: {accelerator.device_name}")
    print(f"Total memory MB: {accelerator.total_memory_mb}")

    left = torch.randn(1024, 1024, device=accelerator.torch_device)
    right = torch.randn(1024, 1024, device=accelerator.torch_device)
    accelerator.synchronize()
    started = time.perf_counter()
    result = left @ right
    accelerator.synchronize()
    elapsed = time.perf_counter() - started
    if result.device.type != accelerator.backend:
        raise RuntimeError(f"Result is on {result.device}, expected {accelerator.backend}")
    print(f"Result device: {result.device}")
    print(f"Elapsed seconds: {elapsed:.6f}")
    print(f"Checksum: {result.mean().item():.8f}")
    print(f"SUCCESS: {accelerator.backend} accelerator operation completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

