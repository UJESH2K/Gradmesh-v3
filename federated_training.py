from __future__ import annotations

import base64
import io
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable, List, Optional

try:
    import torch
except Exception:  # pragma: no cover - optional dependency for syntax/runtime fallback
    torch = None


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _normalize_zip_path(data: str) -> bytes:
    payload = data.split(",", 1)[1] if data.startswith("data:") and "," in data else data
    return base64.b64decode(payload.encode("ascii"))


def decode_base64_bytes(data: str) -> bytes:
    return _normalize_zip_path(data)


def encode_bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def extract_zip_bytes(zip_bytes: bytes, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        archive.extractall(destination)
    return destination


def _find_dir_with_suffix(root: Path, suffix_parts: tuple[str, str]) -> Optional[Path]:
    for candidate in root.rglob("*"):
        if candidate.is_dir() and tuple(candidate.parts[-2:]) == suffix_parts:
            return candidate
    return None


def discover_yolo_dataset_root(dataset_root: Path) -> Path:
    candidates = list(dataset_root.iterdir())
    if len(candidates) == 1 and candidates[0].is_dir():
        return candidates[0]
    return dataset_root


def discover_yolo_split_dirs(dataset_root: Path) -> dict[str, Path]:
    root = discover_yolo_dataset_root(dataset_root)
    train_images = _find_dir_with_suffix(root, ("images", "train")) or _find_dir_with_suffix(root, ("train", "images"))
    train_labels = _find_dir_with_suffix(root, ("labels", "train")) or _find_dir_with_suffix(root, ("train", "labels"))
    val_images = _find_dir_with_suffix(root, ("images", "val")) or _find_dir_with_suffix(root, ("val", "images"))
    val_labels = _find_dir_with_suffix(root, ("labels", "val")) or _find_dir_with_suffix(root, ("val", "labels"))

    if train_images is None or train_labels is None:
        raise ValueError("Dataset must contain train images and labels folders")

    return {
        "root": root,
        "train_images": train_images,
        "train_labels": train_labels,
        "val_images": val_images,
        "val_labels": val_labels,
    }


def _label_for_image(image_path: Path, image_dir: Path, label_dir: Path) -> Path:
    relative = image_path.relative_to(image_dir)
    return label_dir / relative.with_suffix(".txt")


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_split_pairs(
    image_paths: Iterable[Path],
    image_dir: Path,
    label_dir: Path,
    shard_root: Path,
    split_name: str,
) -> int:
    count = 0
    for image_path in image_paths:
        relative = image_path.relative_to(image_dir)
        label_path = _label_for_image(image_path, image_dir, label_dir)
        target_image = shard_root / "images" / split_name / relative
        target_label = shard_root / "labels" / split_name / relative.with_suffix(".txt")
        _copy_file(image_path, target_image)
        if label_path.exists():
            _copy_file(label_path, target_label)
        else:
            target_label.parent.mkdir(parents=True, exist_ok=True)
            target_label.write_text("", encoding="utf-8")
        count += 1
    return count


def write_data_yaml(dataset_root: Path, class_names: List[str], val_relative: str = "images/val") -> Path:
    yaml_path = dataset_root / "data.yaml"
    names_block = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names)) or "  0: object"
    yaml_text = (
        f"path: {dataset_root.as_posix()}\n"
        f"train: images/train\n"
        f"val: {val_relative}\n"
        f"nc: {len(class_names) if class_names else 1}\n"
        f"names:\n{names_block}\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def build_yolo_shards(dataset_root: Path, shard_output_dir: Path, shard_count: int, class_names: List[str]) -> list[dict[str, Any]]:
    split_dirs = discover_yolo_split_dirs(dataset_root)
    train_images = sorted(
        path for path in split_dirs["train_images"].rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    val_images = []
    if split_dirs["val_images"] is not None:
        val_images = sorted(
            path for path in split_dirs["val_images"].rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if not train_images:
        raise ValueError("No training images were found in the dataset")

    shard_output_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, Any]] = []
    shard_groups = [train_images[index::shard_count] for index in range(shard_count)]

    for shard_index, shard_images in enumerate(shard_groups):
        shard_root = shard_output_dir / f"shard_{shard_index}"
        if shard_root.exists():
            shutil.rmtree(shard_root)
        shard_root.mkdir(parents=True, exist_ok=True)

        train_count = _copy_split_pairs(
            shard_images,
            split_dirs["train_images"],
            split_dirs["train_labels"],
            shard_root,
            "train",
        )

        val_count = 0
        if split_dirs["val_images"] is not None and split_dirs["val_labels"] is not None:
            val_count = _copy_split_pairs(
                val_images,
                split_dirs["val_images"],
                split_dirs["val_labels"],
                shard_root,
                "val",
            )

        # Some datasets ship without a dedicated val split. In that case, validate on train images.
        val_relative = "images/val" if val_count > 0 else "images/train"
        data_yaml = write_data_yaml(shard_root, class_names, val_relative=val_relative)
        shard_zip = shard_output_dir / f"shard_{shard_index}.zip"
        if shard_zip.exists():
            shard_zip.unlink()
        with zipfile.ZipFile(shard_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in shard_root.rglob("*"):
                if file_path.is_file():
                    archive.write(file_path, file_path.relative_to(shard_root))

        shards.append(
            {
                "shard_index": shard_index,
                "root": shard_root,
                "zip_path": shard_zip,
                "data_yaml_path": data_yaml,
                "train_count": train_count,
                "val_count": val_count,
            }
        )

    return shards


def _state_dict_from_bytes(state_bytes: bytes) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("torch is required for weight synchronization")
    buffer = io.BytesIO(state_bytes)
    payload = torch.load(buffer, map_location="cpu")
    if isinstance(payload, dict) and all(hasattr(value, "shape") or not hasattr(value, "dtype") for value in payload.values()):
        return payload
    raise ValueError("Serialized payload does not contain a state_dict")


def encode_state_dict(state_dict: dict[str, Any]) -> str:
    if torch is None:
        raise RuntimeError("torch is required for weight synchronization")
    buffer = io.BytesIO()
    torch.save({key: value.detach().cpu() if hasattr(value, "detach") else value for key, value in state_dict.items()}, buffer)
    return encode_bytes_to_base64(buffer.getvalue())


def decode_state_dict(data_b64: str) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("torch is required for weight synchronization")
    raw_bytes = decode_base64_bytes(data_b64)
    return _state_dict_from_bytes(raw_bytes)


def average_state_dicts(encoded_state_dicts: List[str]) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("torch is required for weight synchronization")
    if not encoded_state_dicts:
        raise ValueError("At least one state dict is required")

    state_dicts = [decode_state_dict(item) for item in encoded_state_dicts]
    averaged: dict[str, Any] = {}
    reference = state_dicts[0]

    for key, reference_value in reference.items():
        if hasattr(reference_value, "detach") and hasattr(reference_value, "float"):
            tensor_values = [state_dict[key].detach().float().cpu() for state_dict in state_dicts if key in state_dict and hasattr(state_dict[key], "detach")]
            if tensor_values:
                averaged_tensor = sum(tensor_values) / float(len(tensor_values))
                averaged[key] = averaged_tensor.to(reference_value.dtype)
                continue

        averaged[key] = reference_value

    return averaged
