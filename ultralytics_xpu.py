"""Narrow Ultralytics compatibility layer for native PyTorch XPU training."""

from __future__ import annotations

import gc
from typing import Any

import torch
from torch.optim import Adam, AdamW, Adamax, NAdam, RAdam
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.obb import OBBTrainer


class _XPUTrainerMixin:
    """Adapt the Ultralytics trainer paths that assume every non-CPU device is CUDA."""

    xpu_model_verified = False
    xpu_foreach_disabled = False

    def _setup_train(self) -> None:
        if self.device.type != "xpu":
            raise RuntimeError(f"XPU trainer received unexpected device: {self.device}")
        if self.args.amp:
            raise RuntimeError("AMP is disabled for the initial Intel XPU validation path")
        super()._setup_train()
        parameter_device = next(self.model.parameters()).device
        if parameter_device.type != "xpu":
            raise RuntimeError(f"Ultralytics model is not on XPU: {parameter_device}")
        self.xpu_model_verified = True

    def build_optimizer(self, *args: Any, **kwargs: Any) -> torch.optim.Optimizer:
        optimizer = super().build_optimizer(*args, **kwargs)
        adam_types = (Adam, AdamW, Adamax, NAdam, RAdam)
        if isinstance(optimizer, adam_types):
            optimizer.defaults["foreach"] = False
            for group in optimizer.param_groups:
                group["foreach"] = False
            self.xpu_foreach_disabled = all(group.get("foreach") is False for group in optimizer.param_groups)
            if not self.xpu_foreach_disabled:
                raise RuntimeError("Failed to disable foreach for the XPU Adam-family optimizer")
        return optimizer

    def optimizer_step(self) -> None:
        self.scaler.unscale_(self.optimizer)
        # clip_grad_norm_ also auto-selects foreach kernels independently of
        # the optimizer. That path crashed the Level Zero runtime in round 3.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0, foreach=False)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)
        torch.xpu.synchronize(self.device)

    def final_eval(self) -> None:
        # BaseTrainer reloads the checkpoint through AutoBackend, which reparses
        # args.device as the unsupported string "xpu:0" even when val=False.
        if self.args.val:
            raise NotImplementedError("Ultralytics final validation on XPU is not enabled yet; use val=False")

    def _get_memory(self, fraction: bool = False) -> float:
        reserved = torch.xpu.memory_reserved(self.device)
        if not fraction:
            return reserved / 2**30
        total = torch.xpu.get_device_properties(self.device).total_memory
        return reserved / total if total else 0.0

    def _clear_memory(self, threshold: float | None = None) -> None:
        if threshold is not None:
            if not 0 <= threshold <= 1:
                raise ValueError("Memory threshold must be between 0 and 1")
            if self._get_memory(fraction=True) <= threshold:
                return
        gc.collect()
        torch.xpu.empty_cache()


class XPUDetectionTrainer(_XPUTrainerMixin, DetectionTrainer):
    """Detection trainer using PyTorch XPU without CUDA-only utility calls."""


class XPUOBBTrainer(_XPUTrainerMixin, OBBTrainer):
    """OBB trainer using PyTorch XPU without CUDA-only utility calls."""


def trainer_for_task(task: str):
    """Return the supported XPU trainer class for an Ultralytics task."""
    trainers = {"detect": XPUDetectionTrainer, "obb": XPUOBBTrainer}
    try:
        return trainers[task]
    except KeyError as exc:
        raise NotImplementedError(f"XPU training is not implemented for Ultralytics task '{task}'") from exc


def xpu_train(model, **kwargs: Any):
    """Train an Ultralytics model explicitly on XPU with the compatibility policy."""
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("Intel XPU was requested but torch.xpu.is_available() is False")
    kwargs["device"] = torch.device("xpu:0")
    kwargs["amp"] = False
    kwargs["trainer"] = trainer_for_task(model.task)
    return model.train(**kwargs)
