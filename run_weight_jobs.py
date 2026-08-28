import argparse
import base64
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from urllib import request

from federated_training import discover_yolo_split_dirs


def banner(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")


def now_ts() -> str:
    return time.strftime("%H:%M:%S")


def http_get(base: str, path: str, timeout: int = 20) -> dict:
    req = request.Request(f"{base}{path}", method="GET")
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post(base: str, path: str, payload: dict, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{base}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def file_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def create_limited_dataset_zip(source_zip: Path, max_images: int, temp_dir: Path) -> Path:
    extracted = temp_dir / "dataset"
    with zipfile.ZipFile(source_zip, "r") as archive:
        archive.extractall(extracted)

    split_dirs = discover_yolo_split_dirs(extracted)
    train_images = sorted(
        path
        for path in split_dirs["train_images"].rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )[:max_images]
    if not train_images:
        raise ValueError("No training images found in dataset ZIP")

    limited_root = temp_dir / "limited_dataset"
    train_image_dir = limited_root / "train" / "images"
    train_label_dir = limited_root / "train" / "labels"
    for image_path in train_images:
        relative = image_path.relative_to(split_dirs["train_images"])
        destination = train_image_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, destination)
        label_path = split_dirs["train_labels"] / relative.with_suffix(".txt")
        if label_path.exists():
            label_destination = train_label_dir / relative.with_suffix(".txt")
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label_path, label_destination)

    if split_dirs["val_images"] is not None and split_dirs["val_labels"] is not None:
        for image_path in split_dirs["val_images"].rglob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            relative = image_path.relative_to(split_dirs["val_images"])
            destination = limited_root / "valid" / "images" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, destination)
            label_path = split_dirs["val_labels"] / relative.with_suffix(".txt")
            if label_path.exists():
                label_destination = limited_root / "valid" / "labels" / relative.with_suffix(".txt")
                label_destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(label_path, label_destination)

    limited_zip = temp_dir / "strawberry_limited.zip"
    with zipfile.ZipFile(limited_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in limited_root.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(limited_root))
    return limited_zip


def wait_training_done(base: str, job_id: str, poll_seconds: float = 5.0) -> tuple[dict, float]:
    start = time.time()
    while True:
        status = http_get(base, f"/training_status/{job_id}", timeout=20)
        progress = status.get("progress", {})
        workers = progress.get("workers", [])
        worker_progress = ", ".join(
            f"{worker.get('display_name') or worker.get('node_id', '')[:8]}: epoch {worker.get('epoch', 0)}/{worker.get('total_epochs', 0)}"
            for worker in workers
        ) or "no active worker"
        elapsed = time.time() - start
        print(
            f"[{now_ts()}] job={job_id[:8]} status={status.get('status'):<7} "
            f"round={progress.get('round')}/{progress.get('total_rounds')} "
            f"done={progress.get('done')}/{progress.get('total')} "
            f"assigned={progress.get('assigned')} elapsed={elapsed:6.1f}s {worker_progress}"
        )
        if status.get("status") == "done":
            return status, time.time() - start
        if status.get("status") == "failed":
            errors = status.get("errors") or []
            detail = " | ".join(str(item) for item in errors) or "No worker error was reported."
            raise RuntimeError(f"Training job failed: {job_id}. {detail}")
        time.sleep(poll_seconds)


def wait_for_active_workers(base: str, required_workers: int, timeout_seconds: int = 120) -> None:
    start = time.time()
    while True:
        nodes_payload = http_get(base, "/nodes", timeout=20)
        nodes = nodes_payload.get("nodes", {})
        active = [n for n in nodes.values() if n.get("active") and n.get("supports_training", True)]
        print(f"[{now_ts()}] active_training_workers={len(active)} required={required_workers}")
        if len(active) >= required_workers:
            return
        if time.time() - start > timeout_seconds:
            raise TimeoutError(
                f"Timed out waiting for {required_workers} active workers. Found {len(active)}."
            )
        time.sleep(5)


def save_weights(base: str, job_id: str, out_path: Path) -> None:
    payload = http_get(base, f"/training_jobs/{job_id}/weights", timeout=30)
    weights_b64 = payload.get("weights_b64")
    if not weights_b64:
        raise RuntimeError(f"No weights returned for job {job_id}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(weights_b64.encode("ascii")))


def submit_training_job(
    base: str,
    dataset_zip: Path,
    job_name: str,
    base_model: str,
    epochs: int,
    imgsz: int,
    batch_size: int,
    worker_count: int,
) -> str:
    payload = http_post(
        base,
        "/submit_training_job",
        {
            "job_name": job_name,
            "dataset_zip_b64": file_to_b64(dataset_zip),
            "dataset_zip_name": dataset_zip.name,
            "base_model": base_model,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch_size": batch_size,
            "worker_count": worker_count,
            "class_names": ["object"],
        },
        timeout=120,
    )
    return payload["job_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 1-worker and 2-worker training jobs and save weights")
    parser.add_argument("--server", default="http://127.0.0.1:8000", help="Coordinator base URL")
    parser.add_argument("--dataset-zip", default="strawberry_dataset.zip", help="Path to dataset zip")
    parser.add_argument("--base-model", default="yolov8n.pt", help="Base model for training")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs (rounds) per job")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--max-images", type=int, default=0, help="Limit training images per experiment; 0 uses the full dataset")
    args = parser.parse_args()

    base = args.server.rstrip("/")
    dataset_zip = Path(args.dataset_zip)
    if not dataset_zip.exists():
        raise FileNotFoundError(f"Dataset zip not found: {dataset_zip}")

    temp_dir = tempfile.TemporaryDirectory()
    if args.max_images > 0:
        dataset_zip = create_limited_dataset_zip(dataset_zip, args.max_images, Path(temp_dir.name))
        print(f"using limited dataset: {args.max_images} training images")

    banner("WAITING FOR 1 ACTIVE WORKER")
    wait_for_active_workers(base, required_workers=1)

    banner("SUBMITTING 1-WORKER TRAINING JOB")
    job1 = submit_training_job(
        base,
        dataset_zip,
        job_name="weighted_with_one",
        base_model=args.base_model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        worker_count=1,
    )
    status1, t1 = wait_training_done(base, job1, poll_seconds=5.0)
    out1 = Path("weights") / f"weighted_with_one_{job1}.pt"
    save_weights(base, job1, out1)

    banner("WAITING FOR 2 ACTIVE WORKERS")
    wait_for_active_workers(base, required_workers=2)

    banner("SUBMITTING 2-WORKER TRAINING JOB")
    job2 = submit_training_job(
        base,
        dataset_zip,
        job_name="weighted_with_two",
        base_model=args.base_model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        worker_count=2,
    )
    status2, t2 = wait_training_done(base, job2, poll_seconds=5.0)
    out2 = Path("weights") / f"weighted_with_two_{job2}.pt"
    save_weights(base, job2, out2)

    banner("RESULTS")
    print(f"job1_id={job1}")
    print(f"job1_time_seconds={t1:.2f}")
    print(f"job1_weights={out1}")
    print(f"job2_id={job2}")
    print(f"job2_time_seconds={t2:.2f}")
    print(f"job2_weights={out2}")
    print(f"speedup_x={(t1 / t2) if t2 > 0 else 0:.3f}")
    print(f"job1_status={status1.get('status')}")
    print(f"job2_status={status2.get('status')}")

    comparison_path = Path("weights") / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "one_worker_seconds": round(t1, 3),
                "two_worker_seconds": round(t2, 3),
                "speedup": round((t1 / t2) if t2 > 0 else 0, 3),
                "one_worker_job_id": job1,
                "two_worker_job_id": job2,
                "base_model": args.base_model,
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"comparison={comparison_path}")


if __name__ == "__main__":
    main()
