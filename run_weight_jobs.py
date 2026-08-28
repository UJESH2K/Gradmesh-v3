import argparse
import base64
import json
import time
from pathlib import Path
from urllib import request


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


def wait_training_done(base: str, job_id: str, poll_seconds: float = 5.0) -> tuple[dict, float]:
    start = time.time()
    while True:
        status = http_get(base, f"/training_status/{job_id}", timeout=20)
        progress = status.get("progress", {})
        elapsed = time.time() - start
        print(
            f"[{now_ts()}] job={job_id[:8]} status={status.get('status'):<7} "
            f"round={progress.get('round')}/{progress.get('total_rounds')} "
            f"done={progress.get('done')}/{progress.get('total')} "
            f"assigned={progress.get('assigned')} elapsed={elapsed:6.1f}s"
        )
        if status.get("status") == "done":
            return status, time.time() - start
        if status.get("status") == "failed":
            raise RuntimeError(f"Training job failed: {job_id}")
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
    args = parser.parse_args()

    base = args.server.rstrip("/")
    dataset_zip = Path(args.dataset_zip)
    if not dataset_zip.exists():
        raise FileNotFoundError(f"Dataset zip not found: {dataset_zip}")

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
