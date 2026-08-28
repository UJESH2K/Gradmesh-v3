import argparse
import base64
import json
import os
from pathlib import Path
import time
import uuid
from urllib import error, request


DEFAULT_SERVER = os.getenv("SERVER_URL", "http://127.0.0.1:8000").rstrip("/")


def http_post(base: str, path: str, payload: dict, timeout: int = 10) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{base}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def http_get(base: str, path: str, timeout: int = 10) -> dict:
    req = request.Request(f"{base}{path}", method="GET")
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Terminal client for the local GPU scheduler")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="Coordinator base URL")

    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover", help="Show the network join endpoint and worker command")
    discover.add_argument("--json", action="store_true", help="Print raw JSON")

    join = subparsers.add_parser("join", help="Join the coordinator as a GPU worker")
    join.add_argument("--gpu-name", default="unknown", help="GPU name shown in the scheduler")
    join.add_argument("--display-name", default=None, help="Friendly name for the machine")
    join.add_argument("--gpu-memory-mb", type=int, default=24000, help="Advertised GPU memory in MB")
    join.add_argument("--max-batch-size", type=int, default=4, help="Maximum batch size to accept")
    join.add_argument("--max-parallel-batches", type=int, default=1, help="Maximum concurrent batches")
    join.add_argument("--preferred-models", default="", help="Comma-separated model names to prefer")
    join.add_argument("--node-id", default=str(uuid.uuid4()), help="Stable node id to register")
    join.add_argument("--poll-seconds", type=float, default=1.5, help="Polling interval for the worker loop")
    join.add_argument("--heartbeat-seconds", type=float, default=5.0, help="Heartbeat interval")
    join.add_argument("--run", action="store_true", help="Keep polling for work after registering")

    submit = subparsers.add_parser("submit", help="Submit a job for inference")
    submit.add_argument("--text", default="this is a distributed gpu system demo", help="Input text")
    submit.add_argument("--chunk-size", type=int, default=1, help="Words per task")
    submit.add_argument("--model-name", default="demo-model", help="Model key")
    submit.add_argument("--adapter-name", default="", help="Optional adapter name")
    submit.add_argument("--priority", type=int, default=5, help="Priority from 1 to 10")
    submit.add_argument(
        "--batching-mode",
        choices=["fast", "balanced", "throughput"],
        default="balanced",
        help="Batching mode",
    )
    submit.add_argument("--estimated-memory-mb", type=int, default=2048, help="Estimated GPU memory per request")
    submit.add_argument("--poll-seconds", type=float, default=2.0, help="Status polling interval")

    train = subparsers.add_parser("train", help="Submit a synchronized training job")
    train.add_argument("--dataset-zip", required=True, help="Path to a YOLO dataset zip")
    train.add_argument("--job-name", default="collaborative-training", help="Training job name")
    train.add_argument("--base-model", default="yolov8n-obb.pt", help="Base checkpoint or model name")
    train.add_argument("--epochs", type=int, default=2, help="Number of synchronized rounds")
    train.add_argument("--imgsz", type=int, default=640, help="Image size")
    train.add_argument("--batch", type=int, default=8, help="Batch size")
    train.add_argument("--workers", type=int, default=2, help="Number of GPU workers to sync")
    train.add_argument("--class-names", default="object", help="Comma-separated class names")
    train.add_argument("--poll-seconds", type=float, default=5.0, help="Status polling interval")

    status = subparsers.add_parser("status", help="Check job status")
    status.add_argument("job_id", help="Job id returned by submit")

    return parser


def print_network_manifest(server: str, as_json: bool = False) -> None:
    payload = http_get(server, "/network", timeout=10)
    if as_json:
        print(json.dumps(payload, indent=2))
        return

    manifest = payload.get("manifest", {})
    print(f"Coordinator: {server}")
    print(f"Join endpoint: {server.rstrip('/')}{payload.get('join', '/join')}")
    print(f"Worker command: {manifest.get('worker_command', 'python worker.py --server http://HOST:8000')}")
    print("Active nodes:")
    for node in payload.get("active_nodes", []):
        label = node.get("display_name") or node.get("node_id")
        print(
            f"- {label} | gpu={node.get('gpu')} | memory={node.get('gpu_memory_mb')}MB | batch={node.get('max_batch_size')} | active={node.get('active')}"
        )



def join_worker(args: argparse.Namespace) -> None:
    preferred_models = [item.strip() for item in args.preferred_models.split(",") if item.strip()]
    payload = {
        "node_id": args.node_id,
        "gpu": args.gpu_name,
        "display_name": args.display_name,
        "gpu_memory_mb": args.gpu_memory_mb,
        "max_batch_size": args.max_batch_size,
        "max_parallel_batches": args.max_parallel_batches,
        "preferred_models": preferred_models,
    }

    response = http_post(args.server, "/join", payload, timeout=10)
    print(
        f"joined node_id={response['node_id']} gpu={args.gpu_name} memory={args.gpu_memory_mb}MB batch={args.max_batch_size}"
    )

    if not args.run:
        print("Use --run to stay online and pull batches from the scheduler.")
        return

    poll_seconds = args.poll_seconds
    heartbeat_seconds = args.heartbeat_seconds
    last_heartbeat = 0.0

    while True:
        now = time.time()
        try:
            if now - last_heartbeat >= heartbeat_seconds:
                http_post(
                    args.server,
                    "/heartbeat",
                    {
                        "node_id": args.node_id,
                        "load": 0.1,
                        "active_batches": 0,
                        "allocated_memory_mb": 0,
                    },
                    timeout=10,
                )
                last_heartbeat = now

            batch_payload = http_get(args.server, f"/get_batch/{args.node_id}", timeout=15)
            batch = batch_payload.get("batch")
            if batch and batch.get("batch_id"):
                print(f"batch={batch['batch_id']} model={batch.get('model_name')} size={batch.get('batch_size')}")
                results = []
                for index, task in enumerate(batch.get("tasks", [])):
                    results.append(
                        {
                            "task_id": task["task_id"],
                            "job_id": task["job_id"],
                            "output": f"{batch.get('model_name', 'demo-model')}:{index}:{task['data'].upper()}",
                        }
                    )

                http_post(
                    args.server,
                    "/submit_batch_result",
                    {
                        "node_id": args.node_id,
                        "batch_id": batch["batch_id"],
                        "results": results,
                    },
                    timeout=15,
                )
                print(f"completed batch={batch['batch_id']}")
            else:
                time.sleep(poll_seconds)
        except KeyboardInterrupt:
            print("worker stopped")
            break
        except error.HTTPError as exc:
            print(f"server error: {exc.code}")
            time.sleep(poll_seconds)
        except Exception as exc:
            print(f"worker error: {exc}")
            time.sleep(poll_seconds)


def _file_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")



def submit_job(args: argparse.Namespace) -> None:
    payload = http_post(
        args.server,
        "/submit_job",
        {
            "text": args.text,
            "chunk_size": args.chunk_size,
            "model_name": args.model_name,
            "adapter_name": args.adapter_name or None,
            "priority": args.priority,
            "batching_mode": args.batching_mode,
            "estimated_memory_mb": args.estimated_memory_mb,
        },
        timeout=10,
    )

    job_id = payload["job_id"]
    print(
        f"submitted job_id={job_id} tasks={payload['task_count']} model={payload.get('model_name')} mode={payload.get('batching_mode')}"
    )

    while True:
        status = http_get(args.server, f"/job_status/{job_id}", timeout=10)
        progress = status["progress"]
        print(
            "status={status} done={done}/{total} pending={pending} failed={failed}".format(
                status=status["status"],
                done=progress["done"],
                total=progress["total"],
                pending=progress.get("pending", 0),
                failed=progress["failed"],
            )
        )

        if status["status"] == "done":
            print("output:", status.get("output", ""))
            break

        if status["status"] == "failed":
            print("job failed after retries")
            break

        time.sleep(args.poll_seconds)


def submit_training_job(args: argparse.Namespace) -> None:
    dataset_zip = Path(args.dataset_zip)
    if not dataset_zip.exists():
        raise FileNotFoundError(f"Dataset zip not found: {dataset_zip}")

    class_names = [item.strip() for item in args.class_names.split(",") if item.strip()]
    payload = http_post(
        args.server,
        "/submit_training_job",
        {
            "job_name": args.job_name,
            "dataset_zip_b64": _file_to_base64(dataset_zip),
            "dataset_zip_name": dataset_zip.name,
            "base_model": args.base_model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch_size": args.batch,
            "worker_count": args.workers,
            "class_names": class_names or ["object"],
        },
        timeout=60,
    )

    job_id = payload["job_id"]
    print(
        f"submitted training_job={job_id} rounds={payload['rounds']} workers={payload['worker_count']} shards={payload['shards']}"
    )

    while True:
        status = http_get(args.server, f"/training_status/{job_id}", timeout=10)
        progress = status["progress"]
        print(
            "status={status} round={round}/{total_rounds} done={done}/{total} assigned={assigned}".format(
                status=status["status"],
                round=progress["round"],
                total_rounds=progress["total_rounds"],
                done=progress["done"],
                total=progress["total"],
                assigned=progress.get("assigned", 0),
            )
        )

        if status["status"] == "done":
            print("final weights:", status.get("weights_url", "<not available>"))
            break

        if status["status"] == "failed":
            print("training job failed")
            break

        time.sleep(args.poll_seconds)



def show_status(args: argparse.Namespace) -> None:
    status = http_get(args.server, f"/job_status/{args.job_id}", timeout=10)
    print(json.dumps(status, indent=2))



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    server = args.server.rstrip("/")

    if args.command == "discover":
        print_network_manifest(server, as_json=args.json)
        return

    if args.command == "join":
        join_worker(args)
        return

    if args.command == "submit":
        submit_job(args)
        return

    if args.command == "train":
        submit_training_job(args)
        return

    if args.command == "status":
        show_status(args)
        return


if __name__ == "__main__":
    main()
