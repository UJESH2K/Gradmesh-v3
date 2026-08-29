import argparse
import base64
import io
import importlib
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib import error, request

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from federated_training import decode_state_dict, encode_state_dict
from accelerator import Accelerator, detect_accelerator

torch = None
ACCELERATOR: Accelerator | None = None


# Global variables that can be overridden by CLI args or env vars
SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:8000").rstrip("/")
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "1.5"))
HEARTBEAT_SECONDS = float(os.getenv("HEARTBEAT_SECONDS", "5"))
NODE_ID = os.getenv("NODE_ID", str(uuid.uuid4()))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "4"))
GPU_MEMORY_MB = int(os.getenv("GPU_MEMORY_MB", "24000"))
PREFERRED_MODELS = [item.strip() for item in os.getenv("PREFERRED_MODELS", "").split(",") if item.strip()]
DISPLAY_NAME = os.getenv("DISPLAY_NAME", None)
BACKEND = os.getenv("BACKEND", "auto")


def http_post(path: str, payload: dict, timeout: int = 10) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{SERVER_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def http_get(path: str, timeout: int = 10) -> dict:
    req = request.Request(f"{SERVER_URL}{path}", method="GET")
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def download_bytes(path: str, timeout: int = 20) -> bytes:
    req = request.Request(f"{SERVER_URL}{path}", method="GET")
    with request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_model(model_name: str, destination: Path) -> Path:
    model_path = Path(model_name)
    if model_path.is_file():
        return model_path

    safe_name = model_path.name
    if model_path.suffix.lower() != ".pt" or safe_name != model_name:
        raise FileNotFoundError(f"Model must be a local .pt file or a coordinator filename: {model_name}")

    destination.mkdir(parents=True, exist_ok=True)
    local_path = destination / safe_name
    local_path.write_bytes(download_bytes(f"/models/{safe_name}", timeout=120))
    return local_path


def init_torch() -> None:
    global torch
    try:
        torch = importlib.import_module("torch")
    except Exception:
        torch = None


def detect_gpu_profile() -> tuple[str, int]:
    if ACCELERATOR is None:
        raise RuntimeError("Accelerator has not been initialized")
    return ACCELERATOR.device_name, ACCELERATOR.total_memory_mb


def register_node(gpu_name: str, gpu_memory_mb: int, supports_training: bool) -> None:
    http_post(
        "/register_node",
        {
            "node_id": NODE_ID,
            "gpu": gpu_name,
            "display_name": DISPLAY_NAME,
            "gpu_memory_mb": gpu_memory_mb,
            "max_batch_size": MAX_BATCH_SIZE,
            "max_parallel_batches": 1,
            "preferred_models": PREFERRED_MODELS,
            "supports_training": supports_training,
        },
        timeout=10,
    )


def current_load(active_batches: int = 0) -> float:
    if ACCELERATOR is not None and ACCELERATOR.supports_training:
        return 0.1 if active_batches == 0 else 0.9
    return 0.05 if active_batches == 0 else 0.8


def send_heartbeat(
    active_batches: int = 0,
    allocated_memory_mb: int = 0,
    training_epoch: int | None = None,
    training_total_epochs: int | None = None,
) -> None:
    http_post(
        "/heartbeat",
        {
            "node_id": NODE_ID,
            "load": current_load(active_batches),
            "active_batches": active_batches,
            "allocated_memory_mb": allocated_memory_mb,
            "training_epoch": training_epoch,
            "training_total_epochs": training_total_epochs,
        },
        timeout=10,
    )


def get_batch() -> dict:
    return http_get(f"/get_batch/{NODE_ID}", timeout=15)


def process_batch(batch: dict) -> list[dict]:
    if batch.get("kind") == "training":
        raise ValueError("training batches must be handled by train_batch")

    tasks = batch["tasks"]
    payloads = [item["data"] for item in tasks]
    model_name = batch.get("model_name", "demo-model")
    adapter_name = batch.get("adapter_name") or "base"

    if torch is not None:
        if ACCELERATOR is None:
            raise RuntimeError("Accelerator has not been initialized")
        device = ACCELERATOR.torch_device
        lengths = [len(text) for text in payloads]
        tensor = torch.tensor(lengths, dtype=torch.float32, device=device)
        features = torch.stack([tensor, tensor * 2.0, tensor.sqrt().clamp_min(1.0)], dim=1)
        _ = features.sum().item()

    time.sleep(0.25 + 0.1 * max(0, len(payloads) - 1))

    results = []
    for index, task in enumerate(tasks):
        output = f"{model_name}:{adapter_name}:{index}:{task['data'].upper()}"
        results.append(
            {
                "task_id": task["task_id"],
                "job_id": task["job_id"],
                "output": output,
            }
        )
    return results


def train_batch(batch: dict) -> dict:
    if YOLO is None:
        raise RuntimeError("ultralytics is required for training batches")

    shard_bytes = download_bytes(batch["shard_url"], timeout=60)
    weights_payload = http_get(f"{batch['weights_url'].replace(SERVER_URL, '')}", timeout=20)

    with TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        shard_zip_path = temp_root / "shard.zip"
        shard_zip_path.write_bytes(shard_bytes)

        extract_dir = temp_root / "shard"
        extract_dir.mkdir(parents=True, exist_ok=True)

        import zipfile

        with zipfile.ZipFile(shard_zip_path, "r") as archive:
            archive.extractall(extract_dir)

        data_yaml = extract_dir / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError("training shard is missing data.yaml")

        # Shards are generated on the coordinator, so their YAML may contain
        # the coordinator's absolute path. Each worker extracts the archive in
        # a different temporary directory and must point YOLO at that copy.
        import yaml

        with data_yaml.open("r", encoding="utf-8") as stream:
            dataset_config = yaml.safe_load(stream) or {}
        dataset_config["path"] = str(extract_dir.resolve())
        with data_yaml.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(dataset_config, stream, sort_keys=False)

        model_path = download_model(batch.get("base_model", "yolov8n-obb.pt"), temp_root / "models")
        model = YOLO(str(model_path))
        weights_b64 = weights_payload.get("weights_b64")
        if weights_b64:
            state_dict = decode_state_dict(weights_b64)
            # Filter out incompatible tensors (e.g., class head shape changes) to avoid load failures.
            current_state = model.model.state_dict()
            compatible_state = {
                key: value
                for key, value in state_dict.items()
                if key in current_state
                and hasattr(value, "shape")
                and hasattr(current_state[key], "shape")
                and tuple(value.shape) == tuple(current_state[key].shape)
            }
            model.model.load_state_dict(compatible_state, strict=False)

        if ACCELERATOR is None:
            raise RuntimeError("Accelerator has not been initialized")
        device = ACCELERATOR.ultralytics_device
        total_epochs = int(batch.get("epochs", 1))
        allocated_memory_mb = int(batch.get("estimated_memory_mb", 0))
        send_heartbeat(
            active_batches=1,
            allocated_memory_mb=allocated_memory_mb,
            training_epoch=0,
            training_total_epochs=total_epochs,
        )

        progress = {"epoch": 0}
        heartbeat_stop = threading.Event()

        def keep_training_alive():
            """YOLO train is blocking, so its worker needs an independent heartbeat."""
            while not heartbeat_stop.wait(max(1.0, HEARTBEAT_SECONDS / 2)):
                try:
                    send_heartbeat(
                        active_batches=1,
                        allocated_memory_mb=allocated_memory_mb,
                        training_epoch=progress["epoch"],
                        training_total_epochs=total_epochs,
                    )
                except Exception as exc:
                    print(f"[worker:{NODE_ID}] heartbeat during training failed: {exc}")

        def report_epoch_end(trainer):
            progress["epoch"] = int(getattr(trainer, "epoch", 0)) + 1
            send_heartbeat(
                active_batches=1,
                allocated_memory_mb=allocated_memory_mb,
                training_epoch=progress["epoch"],
                training_total_epochs=total_epochs,
            )

        model.add_callback("on_fit_epoch_end", report_epoch_end)
        heartbeat_thread = threading.Thread(target=keep_training_alive, daemon=True)
        heartbeat_thread.start()
        try:
            train_options = {
                "data": str(data_yaml),
                "epochs": total_epochs,
                "imgsz": int(batch.get("imgsz", 640)),
                "batch": int(batch.get("batch_size", 8)),
                "project": str(temp_root / "runs"),
                "name": f"round_{batch.get('round_index', 0)}",
                "workers": 0,
                "verbose": False,
            }
            if ACCELERATOR.backend == "xpu":
                from ultralytics_xpu import xpu_train

                xpu_train(model, optimizer="Adam", val=False, plots=False, **train_options)
            else:
                # Preserve the existing CUDA/CPU Ultralytics call unchanged.
                model.train(device=device, **train_options)
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=2)

        encoded_weights = encode_state_dict(model.model.state_dict())

        return {
            "node_id": NODE_ID,
            "batch_id": batch["batch_id"],
            "round_index": int(batch.get("round_index", 0)),
            "weights_b64": encoded_weights,
            "metrics": {
                "trained": True,
                "device": str(ACCELERATOR.torch_device),
                "backend": ACCELERATOR.backend,
                "base_model": batch.get("base_model", "yolov8n-obb.pt"),
            },
        }


def submit_batch_result(batch_id: str, results: list[dict]) -> None:
    http_post(
        "/submit_batch_result",
        {
            "node_id": NODE_ID,
            "batch_id": batch_id,
            "results": results,
        },
        timeout=15,
    )


def submit_training_round_result(payload: dict) -> None:
    http_post(
        "/submit_training_round_result",
        payload,
        timeout=60,
    )


def submit_training_batch_failure(batch_id: str, exc: Exception) -> None:
    http_post(
        "/submit_training_batch_failure",
        {"node_id": NODE_ID, "batch_id": batch_id, "error": f"{type(exc).__name__}: {exc}"[:4000]},
        timeout=30,
    )


def main() -> None:
    global SERVER_URL, POLL_SECONDS, HEARTBEAT_SECONDS, NODE_ID, MAX_BATCH_SIZE, GPU_MEMORY_MB, DISPLAY_NAME, BACKEND, ACCELERATOR
    
    parser = argparse.ArgumentParser(description="GPU worker agent for distributed training")
    parser.add_argument("--server-url", default=SERVER_URL, help="Coordinator base URL")
    parser.add_argument("--name", default=DISPLAY_NAME, help="Display name for this worker")
    parser.add_argument("--poll-seconds", type=float, default=POLL_SECONDS, help="Polling interval in seconds")
    parser.add_argument("--heartbeat-seconds", type=float, default=HEARTBEAT_SECONDS, help="Heartbeat interval in seconds")
    parser.add_argument("--max-batch-size", type=int, default=MAX_BATCH_SIZE, help="Maximum batch size")
    parser.add_argument("--gpu-memory-mb", type=int, default=GPU_MEMORY_MB, help="Advertised GPU memory in MB")
    parser.add_argument("--node-id", default=NODE_ID, help="Stable node ID")
    parser.add_argument("--backend", choices=["auto", "cuda", "xpu", "cpu"], default=BACKEND, help="Accelerator backend")
    
    args = parser.parse_args()
    
    # Update globals from CLI args
    SERVER_URL = args.server_url.rstrip("/")
    POLL_SECONDS = args.poll_seconds
    HEARTBEAT_SECONDS = args.heartbeat_seconds
    NODE_ID = args.node_id
    MAX_BATCH_SIZE = args.max_batch_size
    GPU_MEMORY_MB = args.gpu_memory_mb
    DISPLAY_NAME = args.name
    BACKEND = args.backend
    
    init_torch()
    if torch is None:
        raise RuntimeError("PyTorch could not be imported")
    ACCELERATOR = detect_accelerator(BACKEND, advertised_memory_mb=GPU_MEMORY_MB)
    gpu_name, gpu_memory_mb = detect_gpu_profile()
    supports_training = ACCELERATOR.supports_training
    print(
        f"[worker:{NODE_ID}] registering backend='{ACCELERATOR.backend}' "
        f"gpu='{gpu_name}' memory={gpu_memory_mb}MB at {SERVER_URL}"
    )
    if not supports_training:
        print("[worker] no supported accelerator is available; this worker will not accept training jobs")
    register_node(gpu_name, gpu_memory_mb, supports_training=supports_training)

    last_heartbeat = 0.0
    active_batches = 0
    allocated_memory_mb = 0

    while True:
        now = time.time()
        try:
            if now - last_heartbeat >= HEARTBEAT_SECONDS:
                send_heartbeat(active_batches=active_batches, allocated_memory_mb=allocated_memory_mb)
                last_heartbeat = now

            payload = get_batch()
            batch = payload.get("batch")
            if batch and batch.get("batch_id"):
                batch_id = batch["batch_id"]
                active_batches = 1
                allocated_memory_mb = batch.get("estimated_memory_mb", batch.get("memory_mb", 0))
                send_heartbeat(active_batches=active_batches, allocated_memory_mb=allocated_memory_mb)

                if batch.get("kind") == "training":
                    print(
                        f"[worker:{NODE_ID}] training round={batch.get('round_index')} shard={batch.get('shard_index')} job={batch.get('job_id')}"
                    )
                    # Start before downloads/model initialization. Those steps can take longer
                    # than the coordinator liveness timeout on a cold worker.
                    batch_heartbeat_stop = threading.Event()

                    def keep_assigned_training_batch_alive():
                        while not batch_heartbeat_stop.wait(max(1.0, HEARTBEAT_SECONDS / 2)):
                            try:
                                send_heartbeat(
                                    active_batches=1,
                                    allocated_memory_mb=allocated_memory_mb,
                                )
                            except Exception as exc:
                                print(f"[worker:{NODE_ID}] assignment heartbeat failed: {exc}")

                    batch_heartbeat_thread = threading.Thread(
                        target=keep_assigned_training_batch_alive,
                        daemon=True,
                    )
                    batch_heartbeat_thread.start()
                    try:
                        result = train_batch(batch)
                        submit_training_round_result(result)
                    finally:
                        batch_heartbeat_stop.set()
                        batch_heartbeat_thread.join(timeout=2)
                else:
                    print(
                        f"[worker:{NODE_ID}] batch={batch_id} model={batch.get('model_name')} size={batch.get('batch_size')}"
                    )
                    results = process_batch(batch)
                    submit_batch_result(batch_id=batch_id, results=results)

                print(f"[worker:{NODE_ID}] completed batch={batch_id}")

                active_batches = 0
                allocated_memory_mb = 0
                send_heartbeat(active_batches=active_batches, allocated_memory_mb=allocated_memory_mb)
            else:
                time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            print("[worker] stopped by user")
            break
        except error.HTTPError as exc:
            # Ensure scheduler does not think this worker is still busy after an error.
            was_training_batch = bool(active_batches and 'batch_id' in locals() and batch and batch.get("kind") == "training")
            if was_training_batch:
                try:
                    submit_training_batch_failure(batch_id, exc)
                except Exception:
                    pass
            active_batches = 0
            allocated_memory_mb = 0
            try:
                send_heartbeat(active_batches=active_batches, allocated_memory_mb=allocated_memory_mb)
            except Exception:
                pass
            print(f"[worker:{NODE_ID}] server http error: {exc.code}")
            time.sleep(POLL_SECONDS)
        except Exception as exc:
            # Training failures can happen (bad shard, format mismatch). Reset busy state so worker can continue.
            was_training_batch = bool(active_batches and 'batch_id' in locals() and batch and batch.get("kind") == "training")
            if was_training_batch:
                try:
                    submit_training_batch_failure(batch_id, exc)
                except Exception:
                    pass
            active_batches = 0
            allocated_memory_mb = 0
            try:
                send_heartbeat(active_batches=active_batches, allocated_memory_mb=allocated_memory_mb)
            except Exception:
                pass
            print(f"[worker:{NODE_ID}] error: {type(exc).__name__}: {exc}")
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
