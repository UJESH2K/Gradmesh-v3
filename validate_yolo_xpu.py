"""Run a one-epoch Ultralytics OBB training validation on Intel XPU."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from ultralytics_xpu import xpu_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate one small Ultralytics training epoch on XPU")
    parser.add_argument("--model", default="yolov8n-obb.yaml", help="Ultralytics model or checkpoint")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("FAILED: XPU was requested but is unavailable")

    repository = Path(__file__).resolve().parent
    dataset_root = repository / "test_data"
    with tempfile.TemporaryDirectory(prefix="gradmesh-yolo-xpu-") as temp_dir:
        temp_root = Path(temp_dir)
        data_yaml = temp_root / "data.yaml"
        data_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": str(dataset_root),
                    "train": "images/train",
                    "val": "images/val",
                    "nc": 1,
                    "names": {0: "object"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        model = YOLO(args.model)
        seen = {"train_device": None, "optimizer": None, "foreach": None}

        def verify_training_state(trainer) -> None:
            device = next(trainer.model.parameters()).device
            seen["train_device"] = str(device)
            seen["optimizer"] = type(trainer.optimizer).__name__
            seen["foreach"] = [group.get("foreach") for group in trainer.optimizer.param_groups]
            if device.type != "xpu":
                raise RuntimeError(f"FAILED: model parameters are on {device}, not XPU")

        model.add_callback("on_train_start", verify_training_state)
        xpu_train(
            model,
            data=str(data_yaml),
            epochs=1,
            imgsz=args.imgsz,
            batch=args.batch,
            optimizer="Adam",
            workers=0,
            project=str(temp_root / "runs"),
            name="standalone_xpu",
            plots=False,
            verbose=False,
            val=False,
        )

        trainer = model.trainer
        if seen["train_device"] != "xpu:0":
            raise RuntimeError(f"FAILED: training device was {seen['train_device']}")
        if not trainer.xpu_model_verified:
            raise RuntimeError("FAILED: trainer did not verify XPU model placement")
        if not trainer.xpu_foreach_disabled or any(value is not False for value in seen["foreach"]):
            raise RuntimeError(f"FAILED: optimizer foreach values were {seen['foreach']}")

        state = {key: value.detach().cpu() for key, value in model.model.state_dict().items()}
        if any(value.device.type != "cpu" for value in state.values() if isinstance(value, torch.Tensor)):
            raise RuntimeError("FAILED: serialized state contains non-CPU tensors")
        print(f"Device: {seen['train_device']}")
        print(f"Optimizer: {seen['optimizer']}")
        print(f"foreach values: {seen['foreach']}")
        print(f"CPU state tensors: {len(state)}")
        print("SUCCESS: Ultralytics forward, backward, optimizer step, and CPU state export completed on XPU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
