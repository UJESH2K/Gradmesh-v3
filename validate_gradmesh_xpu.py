"""End-to-end validation of one GradMesh XPU worker over three synchronized rounds."""

from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from urllib import error, request

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import OBBModel

from federated_training import decode_state_dict


def http_get(base: str, path: str, timeout: int = 10) -> dict:
    with request.urlopen(f"{base}{path}", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def http_post(base: str, path: str, payload: dict, timeout: int = 60) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{base}{path}", data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def free_local_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def wait_for_server(base: str, process: subprocess.Popen, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Coordinator exited early with code {process.returncode}")
        try:
            http_get(base, "/network", timeout=2)
            return
        except (OSError, error.URLError):
            time.sleep(0.25)
    raise TimeoutError("Coordinator did not become ready")


def wait_for_worker(base: str, node_id: str, process: subprocess.Popen, timeout: float = 45.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Worker exited early with code {process.returncode}")
        nodes = http_get(base, "/nodes", timeout=5).get("nodes", {})
        if node_id in nodes and nodes[node_id].get("active"):
            return nodes[node_id]
        time.sleep(0.5)
    raise TimeoutError("XPU worker did not register")


def terminate(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return "<no log>"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate three GradMesh rounds on one XPU worker")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("FAILED: XPU is unavailable")

    repository = Path(__file__).resolve().parent
    python = Path(sys.executable)
    node_id = "gradmesh-xpu-validation"
    server_process = None
    worker_process = None

    with tempfile.TemporaryDirectory(prefix="gradmesh-distributed-xpu-", ignore_cleanup_errors=True) as temp_dir:
        temp_root = Path(temp_dir)
        server_log = temp_root / "server.log"
        worker_log = temp_root / "worker.log"
        dataset_zip = temp_root / "test_data_obb.zip"
        base_model = temp_root / "yolov8n-obb-base.pt"

        with zipfile.ZipFile(dataset_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for source in (repository / "test_data").rglob("*"):
                if source.is_file() and source.suffix != ".cache":
                    archive.write(source, source.relative_to(repository / "test_data"))
        YOLO("yolov8n-obb.yaml").save(base_model)

        port = free_local_port()
        base = f"http://127.0.0.1:{port}"
        environment = os.environ.copy()
        environment["TRAINING_JOB_DIR"] = str(temp_root / "training_jobs")
        config_root = temp_root / "config"
        (config_root / "Ultralytics").mkdir(parents=True)
        matplotlib_root = temp_root / "matplotlib"
        matplotlib_root.mkdir()
        environment["YOLO_CONFIG_DIR"] = str(config_root)
        environment["MPLCONFIGDIR"] = str(matplotlib_root)
        environment["YOLO_OFFLINE"] = "true"

        try:
            with server_log.open("w", encoding="utf-8") as server_stream, worker_log.open(
                "w", encoding="utf-8"
            ) as worker_stream:
                server_process = subprocess.Popen(
                    [str(python), "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", str(port)],
                    cwd=repository,
                    env=environment,
                    stdout=server_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                wait_for_server(base, server_process)
                worker_process = subprocess.Popen(
                    [
                        str(python),
                        "worker.py",
                        "--server-url",
                        base,
                        "--backend",
                        "xpu",
                        "--name",
                        "arc-140v-validation",
                        "--node-id",
                        node_id,
                        "--max-batch-size",
                        "1",
                    ],
                    cwd=repository,
                    env=environment,
                    stdout=worker_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                node = wait_for_worker(base, node_id, worker_process)
                if not node.get("supports_training") or "Arc(TM) 140V" not in node.get("gpu", ""):
                    raise RuntimeError(f"Unexpected worker registration: {node}")

                submitted = http_post(
                    base,
                    "/submit_training_job",
                    {
                        "job_name": "xpu-integration-validation",
                        "dataset_zip_b64": base64.b64encode(dataset_zip.read_bytes()).decode("ascii"),
                        "dataset_zip_name": dataset_zip.name,
                        "base_model": str(base_model),
                        "epochs": args.rounds,
                        "imgsz": 320,
                        "batch_size": 1,
                        "worker_count": 1,
                        "class_names": ["object"],
                    },
                    timeout=60,
                )
                job_id = submitted["job_id"]
                deadline = time.time() + args.timeout
                while time.time() < deadline:
                    status = http_get(base, f"/training_status/{job_id}", timeout=10)
                    progress = status.get("progress", {})
                    print(
                        f"status={status.get('status')} round={progress.get('round')}/{progress.get('total_rounds')} "
                        f"done={progress.get('done')}/{progress.get('total')}"
                    )
                    if status.get("status") == "done":
                        break
                    if status.get("status") == "failed":
                        raise RuntimeError(f"Distributed training failed: {status.get('errors')}")
                    if worker_process.poll() is not None:
                        raise RuntimeError(f"Worker exited during training with code {worker_process.returncode}")
                    time.sleep(1)
                else:
                    raise TimeoutError("Distributed training did not finish in time")

                if len(status.get("round_metrics", [])) != args.rounds:
                    raise RuntimeError(f"Expected {args.rounds} round metrics: {status.get('round_metrics')}")
                for round_record in status["round_metrics"]:
                    metrics = round_record.get("metrics") or []
                    if len(metrics) != 1 or metrics[0].get("backend") != "xpu":
                        raise RuntimeError(f"Round did not report XPU execution: {round_record}")

                weights_payload = http_get(base, f"/training_jobs/{job_id}/weights", timeout=30)
                state = decode_state_dict(weights_payload["weights_b64"])
                tensors = [value for value in state.values() if isinstance(value, torch.Tensor)]
                if not tensors or any(value.device.type != "cpu" for value in tensors):
                    raise RuntimeError("Coordinator weights are not a non-empty CPU state_dict")
                reload_model = OBBModel("yolov8n-obb.yaml", nc=1, verbose=False)
                reload_model.load_state_dict(state, strict=True)

                print(f"Worker GPU: {node['gpu']}")
                print(f"Worker memory MB: {node['gpu_memory_mb']}")
                print(f"Rounds completed: {args.rounds}")
                print(f"CPU state tensors: {len(tensors)}")
                print("SUCCESS: GradMesh XPU worker, CPU aggregation, repeated reload, and multiple rounds completed")
        except Exception:
            terminate(worker_process)
            terminate(server_process)
            print("--- worker log ---")
            print(tail(worker_log))
            print("--- server log ---")
            print(tail(server_log))
            raise
        finally:
            terminate(worker_process)
            terminate(server_process)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
