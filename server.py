import base64
import os
import shutil
import time
import uuid
import zipfile
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Deque, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from federated_training import average_state_dicts, build_yolo_shards, decode_base64_bytes, decode_state_dict, encode_state_dict, extract_zip_bytes

app = FastAPI(title="Local GPU Scheduler", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_PATH = BASE_DIR / "dashboard.html"
DATASET_ZIP_PATH = Path(
    os.getenv("GRADMESH_DATASET_ZIP", str(BASE_DIR / "strawberry_dataset.zip"))
).expanduser().resolve()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

nodes: Dict[str, dict] = {}
jobs: Dict[str, dict] = {}
tasks: Dict[str, dict] = {}
batches: Dict[str, dict] = {}
task_queue: Deque[str] = deque()
training_jobs: Dict[str, dict] = {}
training_batches: Dict[str, dict] = {}
training_queue: Deque[str] = deque()

state_lock = Lock()

HEARTBEAT_TIMEOUT_SECONDS = 15
BATCH_LEASE_TIMEOUT_SECONDS = 30
TRAINING_BATCH_LEASE_TIMEOUT_SECONDS = 3600
MAX_TASK_ATTEMPTS = 3
FAST_BATCH_LIMIT = 2
BALANCED_BATCH_LIMIT = 4
THROUGHPUT_BATCH_LIMIT = 8
DEFAULT_ESTIMATED_MEMORY_MB = 2048
TRAINING_JOB_DIR = Path(os.getenv("TRAINING_JOB_DIR", str(BASE_DIR / "training_jobs")))
TRAINING_JOB_DIR.mkdir(parents=True, exist_ok=True)


class RegisterNodeRequest(BaseModel):
    node_id: str
    gpu: str = "unknown"
    display_name: Optional[str] = None
    gpu_memory_mb: int = Field(default=24000, ge=1)
    max_batch_size: int = Field(default=4, ge=1, le=64)
    max_parallel_batches: int = Field(default=1, ge=1, le=8)
    preferred_models: List[str] = Field(default_factory=list)
    supports_training: bool = True


class HeartbeatRequest(BaseModel):
    node_id: str
    load: Optional[float] = None
    active_batches: Optional[int] = None
    allocated_memory_mb: Optional[int] = None
    training_epoch: Optional[int] = None
    training_total_epochs: Optional[int] = None


class SubmitJobRequest(BaseModel):
    text: str = Field(..., min_length=1)
    chunk_size: int = Field(default=1, ge=1, le=16)
    model_name: str = Field(default="demo-model", min_length=1)
    adapter_name: Optional[str] = None
    priority: int = Field(default=5, ge=1, le=10)
    batching_mode: str = Field(default="balanced")
    estimated_memory_mb: int = Field(default=DEFAULT_ESTIMATED_MEMORY_MB, ge=1)


class SubmitTrainingJobRequest(BaseModel):
    job_name: str = Field(default="collaborative-training", min_length=1)
    dataset_zip_b64: str = Field(..., min_length=1)
    dataset_zip_name: str = Field(default="dataset.zip", min_length=1)
    base_model: str = Field(default="yolov8n-obb.pt", min_length=1)
    epochs: int = Field(default=2, ge=1, le=1000)
    imgsz: int = Field(default=640, ge=32, le=4096)
    batch_size: int = Field(default=8, ge=1, le=128)
    worker_count: int = Field(default=2, ge=1, le=16)
    class_names: List[str] = Field(default_factory=lambda: ["object"])
    preferred_models: List[str] = Field(default_factory=list)


class SubmitTrainingRoundResultRequest(BaseModel):
    node_id: str
    batch_id: str
    round_index: int = Field(ge=0)
    weights_b64: str = Field(..., min_length=1)
    metrics: Optional[dict] = None


class SubmitTrainingBatchFailureRequest(BaseModel):
    node_id: str
    batch_id: str
    error: str = Field(..., min_length=1)


class SubmitBatchResultItem(BaseModel):
    task_id: str
    job_id: str
    output: str


class SubmitBatchResultRequest(BaseModel):
    node_id: str
    batch_id: str
    results: List[SubmitBatchResultItem]


def _split_chunks(words: List[str], chunk_size: int) -> List[str]:
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]


def _cache_key(model_name: str, adapter_name: Optional[str]) -> str:
    return f"{model_name}::{adapter_name or 'base'}"


def _training_cache_key(job_id: str, round_index: int) -> str:
    return f"{job_id}::round::{round_index}"


def _refresh_nodes_locked(now: float) -> None:
    for node in nodes.values():
        node["active"] = (now - node["last_seen"]) <= HEARTBEAT_TIMEOUT_SECONDS


def _release_batch_locked(batch_id: str, requeue: bool = False) -> None:
    batch = batches.get(batch_id)
    if batch is None or batch.get("status") not in {"assigned", "running", "queued", "done"}:
        return

    node_id = batch["node_id"]
    task_ids = batch["task_ids"]
    memory_mb = batch["memory_mb"]

    if node_id in nodes:
        node = nodes[node_id]
        node["allocated_memory_mb"] = max(0, node["allocated_memory_mb"] - memory_mb)
        node["active_batches"] = max(0, node["active_batches"] - 1)

    for task_id in task_ids:
        task = tasks.get(task_id)
        if task is None or task["status"] == "done":
            continue

        if requeue and task["attempts"] < MAX_TASK_ATTEMPTS:
            task["status"] = "pending"
            task["assigned_node"] = None
            task["batch_id"] = None
            task["assigned_at"] = None
            task_queue.append(task_id)
        else:
            task["status"] = "failed"
            task["assigned_node"] = None
            task["batch_id"] = None
            task["assigned_at"] = None

    batch["status"] = "requeued" if requeue else "done"
    batch["finished_at"] = time.time()


def _release_training_batch_locked(batch_id: str, requeue: bool = False) -> None:
    batch = training_batches.get(batch_id)
    if batch is None or batch.get("status") not in {"assigned", "running", "queued", "done"}:
        return

    node_id = batch["node_id"]
    memory_mb = batch.get("memory_mb", 0)

    if node_id in nodes:
        node = nodes[node_id]
        node["allocated_memory_mb"] = max(0, node["allocated_memory_mb"] - memory_mb)
        node["active_batches"] = max(0, node["active_batches"] - 1)

    # A requeued training batch must be schedulable again; the training
    # scheduler only selects batches whose status is exactly "queued".
    batch["status"] = "queued" if requeue else "done"
    batch["finished_at"] = time.time()


def _fail_training_batch_locked(batch_id: str, message: str) -> None:
    """Finish a failed training batch without hiding the worker error."""
    batch = training_batches.get(batch_id)
    if batch is None or batch.get("status") not in {"assigned", "running"}:
        return

    node_id = batch["node_id"]
    if node_id in nodes:
        node = nodes[node_id]
        node["allocated_memory_mb"] = max(0, node["allocated_memory_mb"] - batch.get("memory_mb", 0))
        node["active_batches"] = max(0, node["active_batches"] - 1)
    batch["status"] = "failed"
    batch["error"] = message
    batch["finished_at"] = time.time()


def _requeue_stale_batches_locked(now: float) -> None:
    for batch_id, batch in list(batches.items()):
        if batch["status"] not in {"assigned", "running"}:
            continue

        node_id = batch["node_id"]
        assigned_at = batch["assigned_at"]
        lease_expired = (now - assigned_at) > BATCH_LEASE_TIMEOUT_SECONDS
        node_inactive = node_id not in nodes or not nodes[node_id]["active"]

        if lease_expired or node_inactive:
            _release_batch_locked(batch_id, requeue=True)


def _requeue_stale_training_batches_locked(now: float) -> None:
    for batch_id, batch in list(training_batches.items()):
        if batch["status"] not in {"assigned", "running"}:
            continue

        node_id = batch["node_id"]
        assigned_at = batch["assigned_at"]
        lease_expired = (now - assigned_at) > TRAINING_BATCH_LEASE_TIMEOUT_SECONDS
        node_inactive = node_id not in nodes or not nodes[node_id]["active"]

        if lease_expired or node_inactive:
            _release_training_batch_locked(batch_id, requeue=True)
            training_queue.appendleft(batch_id)


def _update_job_status_locked(job_id: str) -> None:
    job = jobs[job_id]
    job_task_ids = job["task_ids"]

    done_count = sum(1 for task_id in job_task_ids if tasks[task_id]["status"] == "done")
    failed_count = sum(1 for task_id in job_task_ids if tasks[task_id]["status"] == "failed")
    assigned_count = sum(1 for task_id in job_task_ids if tasks[task_id]["status"] == "assigned")
    pending_count = sum(1 for task_id in job_task_ids if tasks[task_id]["status"] == "pending")

    job["done_count"] = done_count
    job["failed_count"] = failed_count
    job["pending_count"] = pending_count

    if done_count == len(job_task_ids):
        job["status"] = "done"
    elif failed_count > 0 and pending_count == 0 and assigned_count == 0:
        job["status"] = "failed"
    else:
        job["status"] = "running"


def _maintenance_locked() -> None:
    now = time.time()
    _refresh_nodes_locked(now)
    _requeue_stale_batches_locked(now)
    _requeue_stale_training_batches_locked(now)
    for job_id in jobs:
        _update_job_status_locked(job_id)
    for job_id in list(training_jobs):
        _update_training_job_status_locked(job_id)


def _batch_limit_for_node(node: dict, batching_mode: str) -> int:
    if batching_mode == "fast":
        limit = FAST_BATCH_LIMIT
    elif batching_mode == "throughput":
        limit = THROUGHPUT_BATCH_LIMIT
    else:
        limit = BALANCED_BATCH_LIMIT

    if node.get("load") is not None and node["load"] >= 0.9:
        return 1

    return max(1, min(limit, node["max_batch_size"]))


def _node_can_fit_task(node: dict, task: dict) -> bool:
    preferred_models = node.get("preferred_models") or []
    if preferred_models and task["model_name"] not in preferred_models:
        return False

    available_memory = node["gpu_memory_mb"] - node["allocated_memory_mb"]
    if available_memory <= 0:
        return False

    estimated = task["estimated_memory_mb"] + 256
    return estimated <= available_memory


def _pop_batch_from_queue_locked(node: dict) -> Optional[dict]:
    pending_task_ids: List[str] = []
    for task_id in task_queue:
        task = tasks.get(task_id)
        if task is None or task["status"] != "pending":
            continue
        if not _node_can_fit_task(node, task):
            continue
        pending_task_ids.append(task_id)

    if not pending_task_ids:
        return None

    anchor_task_id = max(
        pending_task_ids,
        key=lambda task_id: (tasks[task_id]["priority"], -tasks[task_id]["created_at"]),
    )
    anchor_task = tasks[anchor_task_id]
    cache_key = _cache_key(anchor_task["model_name"], anchor_task["adapter_name"])
    batch_limit = _batch_limit_for_node(node, anchor_task["batching_mode"])

    selected_task_ids: List[str] = []
    selected_memory_mb = 0
    for task_id in task_queue:
        task = tasks.get(task_id)
        if task is None or task["status"] != "pending":
            continue
        if _cache_key(task["model_name"], task["adapter_name"]) != cache_key:
            continue

        projected_memory = max(selected_memory_mb, task["estimated_memory_mb"]) + 256
        available_memory = node["gpu_memory_mb"] - node["allocated_memory_mb"]
        if projected_memory > available_memory:
            continue

        selected_task_ids.append(task_id)
        selected_memory_mb = max(selected_memory_mb, task["estimated_memory_mb"])

        if len(selected_task_ids) >= batch_limit:
            break

    if not selected_task_ids:
        return None

    batch_memory_mb = selected_memory_mb + 256 * len(selected_task_ids)
    batch_id = str(uuid.uuid4())
    now = time.time()

    for task_id in selected_task_ids:
        task = tasks[task_id]
        task["status"] = "assigned"
        task["assigned_node"] = node["node_id"]
        task["assigned_at"] = now
        task["batch_id"] = batch_id
        task["attempts"] += 1

    nodes[node["node_id"]]["allocated_memory_mb"] += batch_memory_mb
    nodes[node["node_id"]]["active_batches"] += 1
    nodes[node["node_id"]]["last_batch_size"] = len(selected_task_ids)
    nodes[node["node_id"]]["last_batch_model"] = anchor_task["model_name"]

    batches[batch_id] = {
        "batch_id": batch_id,
        "node_id": node["node_id"],
        "task_ids": selected_task_ids,
        "status": "assigned",
        "model_name": anchor_task["model_name"],
        "adapter_name": anchor_task["adapter_name"],
        "assigned_at": now,
        "finished_at": None,
        "memory_mb": batch_memory_mb,
        "batch_size": len(selected_task_ids),
    }

    remaining = deque(task_id for task_id in task_queue if task_id not in selected_task_ids)
    task_queue.clear()
    task_queue.extend(remaining)

    return {
        "batch_id": batch_id,
        "node_id": node["node_id"],
        "model_name": anchor_task["model_name"],
        "adapter_name": anchor_task["adapter_name"],
        "batch_size": len(selected_task_ids),
        "batching_mode": anchor_task["batching_mode"],
        "estimated_memory_mb": batch_memory_mb,
        "tasks": [
            {
                "task_id": task_id,
                "job_id": tasks[task_id]["job_id"],
                "data": tasks[task_id]["payload"],
                "attempt": tasks[task_id]["attempts"],
                "priority": tasks[task_id]["priority"],
                "model_name": tasks[task_id]["model_name"],
                "adapter_name": tasks[task_id]["adapter_name"],
            }
            for task_id in selected_task_ids
        ],
    }


def _job_output_locked(job_id: str) -> str:
    ordered_output = [tasks[task_id]["output"] for task_id in jobs[job_id]["task_ids"] if tasks[task_id]["output"] is not None]
    return " ".join(ordered_output)


def _training_job_dir(job_id: str) -> Path:
    return TRAINING_JOB_DIR / job_id


def _training_shard_url(job_id: str, shard_index: int) -> str:
    return f"/training_jobs/{job_id}/shards/{shard_index}.zip"


def _training_weights_url(job_id: str) -> str:
    return f"/training_jobs/{job_id}/weights"


def _enqueue_training_round_locked(job_id: str) -> None:
    job = training_jobs[job_id]
    round_index = job["current_round"]
    shards = job["shards"]

    for shard in shards:
        batch_id = str(uuid.uuid4())
        estimated_memory_mb = max(job["batch_size"] * 128, 1024)
        training_batches[batch_id] = {
            "batch_id": batch_id,
            "job_id": job_id,
            "node_id": None,
            "status": "queued",
            "kind": "training",
            "round_index": round_index,
            "shard_index": shard["shard_index"],
            "assigned_at": None,
            "finished_at": None,
            "memory_mb": estimated_memory_mb,
            "model_name": job["base_model"],
            "base_model": job["base_model"],
            "imgsz": job["imgsz"],
            "batch_size": job["batch_size"],
            "epochs": 1,
            "shard_zip_path": str(shard["zip_path"]),
            "data_yaml_path": str(shard["data_yaml_path"]),
            "metrics": None,
            "error": None,
            "weights_b64": job.get("current_weights_b64"),
        }
        training_queue.append(batch_id)

    job["pending_batches"] = len(shards)
    job["completed_batches"] = 0


def _finalize_training_round_locked(job_id: str) -> None:
    job = training_jobs[job_id]
    round_index = job["current_round"]
    round_batches = [
        batch
        for batch in training_batches.values()
        if batch["job_id"] == job_id and batch["round_index"] == round_index
    ]

    completed_batches = [batch for batch in round_batches if batch.get("status") == "done" and batch.get("weights_b64")]
    if len(completed_batches) != len(round_batches):
        return

    averaged = average_state_dicts([batch["weights_b64"] for batch in completed_batches])
    job["current_weights_b64"] = encode_state_dict(averaged)
    job.setdefault("round_metrics", []).append(
        {
            "round_index": round_index,
            "metrics": [batch.get("metrics") for batch in completed_batches],
        }
    )
    job["current_round"] += 1
    job["completed_batches"] = 0
    job["pending_batches"] = 0

    if job["current_round"] >= job["total_rounds"]:
        job["status"] = "done"
        job["finished_at"] = time.time()
    else:
        _enqueue_training_round_locked(job_id)


def _update_training_job_status_locked(job_id: str) -> None:
    job = training_jobs[job_id]
    round_index = job["current_round"]
    round_batches = [
        batch
        for batch in training_batches.values()
        if batch["job_id"] == job_id and batch["round_index"] == round_index
    ]

    done_count = sum(1 for batch in round_batches if batch.get("status") == "done")
    failed_count = sum(1 for batch in round_batches if batch.get("status") == "failed")
    pending_count = sum(1 for batch in round_batches if batch.get("status") == "queued")
    assigned_count = sum(1 for batch in round_batches if batch.get("status") == "assigned")

    job["completed_batches"] = done_count
    job["pending_batches"] = pending_count

    if job.get("status") in {"done", "failed"}:
        return

    if failed_count > 0 and pending_count == 0 and assigned_count == 0:
        job["status"] = "failed"
    elif done_count == len(round_batches) and round_batches:
        _finalize_training_round_locked(job_id)
        if job.get("status") not in {"done", "failed"}:
            job["status"] = "running"
    else:
        job["status"] = "running"


def _pop_training_batch_from_queue_locked(node: dict) -> Optional[dict]:
    for batch_id in list(training_queue):
        batch = training_batches.get(batch_id)
        if batch is None or batch["status"] != "queued":
            continue

        job = training_jobs.get(batch["job_id"])
        if job is None or job.get("status") in {"done", "failed"}:
            continue
        if not node.get("supports_training", True):
            continue
        if node.get("preferred_models") and job["base_model"] not in node.get("preferred_models", []):
            continue

        if node.get("load") is not None and node["load"] >= 0.95:
            continue

        available_memory = node["gpu_memory_mb"] - node["allocated_memory_mb"]
        if batch["memory_mb"] > available_memory:
            continue

        batch["status"] = "assigned"
        batch["node_id"] = node["node_id"]
        batch["assigned_at"] = time.time()
        nodes[node["node_id"]]["allocated_memory_mb"] += batch["memory_mb"]
        nodes[node["node_id"]]["active_batches"] += 1
        nodes[node["node_id"]]["last_batch_size"] = 1
        nodes[node["node_id"]]["last_batch_model"] = job["base_model"]
        training_queue.remove(batch_id)
        return {
            "batch_id": batch_id,
            "job_id": batch["job_id"],
            "kind": "training",
            "round_index": batch["round_index"],
            "shard_index": batch["shard_index"],
            "shard_url": _training_shard_url(batch["job_id"], batch["shard_index"]),
            "weights_url": _training_weights_url(batch["job_id"]),
            "base_model": batch["base_model"],
            "imgsz": batch["imgsz"],
            "batch_size": batch["batch_size"],
            "estimated_memory_mb": batch["memory_mb"],
            "epochs": batch["epochs"],
            "job_name": job["job_name"],
            "class_names": job["class_names"],
            "current_round": job["current_round"],
            "total_rounds": job["total_rounds"],
            "weights_b64": batch.get("weights_b64"),
        }

    return None


def _network_manifest() -> dict:
    return {
        "service": "local-gpu-scheduler",
        "version": "1.0.0",
        "endpoints": {
            "join": "/join",
            "register_node": "/register_node",
            "heartbeat": "/heartbeat",
            "submit_job": "/submit_job",
            "submit_training_job": "/submit_training_job",
            "get_batch": "/get_batch/{node_id}",
            "submit_batch_result": "/submit_batch_result",
            "submit_training_round_result": "/submit_training_round_result",
            "job_status": "/job_status/{job_id}",
            "training_status": "/training_status/{job_id}",
            "training_shard": "/training_jobs/{job_id}/shards/{shard_index}.zip",
            "training_weights": "/training_jobs/{job_id}/weights",
            "model": "/models/{model_name}",
            "nodes": "/nodes",
            "metrics": "/metrics",
            "dashboard": "/dashboard",
        },
        "worker_command": "python worker.py --server-url http://HOST:8000",
    }


@app.post("/register_node")
def register_node(req: RegisterNodeRequest):
    with state_lock:
        nodes[req.node_id] = {
            "node_id": req.node_id,
            "gpu": req.gpu,
            "display_name": req.display_name,
            "gpu_memory_mb": req.gpu_memory_mb,
            "max_batch_size": req.max_batch_size,
            "max_parallel_batches": req.max_parallel_batches,
            "preferred_models": req.preferred_models,
            "supports_training": req.supports_training,
            "last_seen": time.time(),
            "active": True,
            "load": None,
            "allocated_memory_mb": 0,
            "active_batches": 0,
            "completed_batches": 0,
            "last_batch_size": 0,
            "last_batch_model": None,
            "training_epoch": 0,
            "training_total_epochs": 0,
        }
    return {"status": "registered", "node_id": req.node_id}


@app.post("/join")
def join(req: RegisterNodeRequest):
    return register_node(req)


@app.post("/heartbeat")
def heartbeat(req: HeartbeatRequest):
    with state_lock:
        if req.node_id not in nodes:
            raise HTTPException(status_code=404, detail="Unknown node_id. Register first.")

        nodes[req.node_id]["last_seen"] = time.time()
        nodes[req.node_id]["active"] = True
        nodes[req.node_id]["load"] = req.load
        if req.active_batches is not None:
            nodes[req.node_id]["active_batches"] = req.active_batches
        if req.allocated_memory_mb is not None:
            nodes[req.node_id]["allocated_memory_mb"] = req.allocated_memory_mb
        if req.training_epoch is not None:
            nodes[req.node_id]["training_epoch"] = req.training_epoch
        if req.training_total_epochs is not None:
            nodes[req.node_id]["training_total_epochs"] = req.training_total_epochs
    return {"status": "ok"}


@app.post("/submit_job")
def submit_job(req: SubmitJobRequest):
    words = req.text.split()
    if not words:
        raise HTTPException(status_code=400, detail="Job text must contain at least one word.")

    if req.batching_mode not in {"fast", "balanced", "throughput"}:
        raise HTTPException(status_code=400, detail="batching_mode must be fast, balanced, or throughput")

    chunks = _split_chunks(words, req.chunk_size)
    job_id = str(uuid.uuid4())

    with state_lock:
        job_task_ids: List[str] = []
        for chunk in chunks:
            task_id = str(uuid.uuid4())
            tasks[task_id] = {
                "task_id": task_id,
                "job_id": job_id,
                "payload": chunk,
                "status": "pending",
                "assigned_node": None,
                "batch_id": None,
                "assigned_at": None,
                "attempts": 0,
                "output": None,
                "created_at": time.time(),
                "model_name": req.model_name,
                "adapter_name": req.adapter_name,
                "priority": req.priority,
                "batching_mode": req.batching_mode,
                "estimated_memory_mb": req.estimated_memory_mb,
            }
            task_queue.append(task_id)
            job_task_ids.append(task_id)

        jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "task_ids": job_task_ids,
            "created_at": time.time(),
            "done_count": 0,
            "failed_count": 0,
            "pending_count": len(job_task_ids),
            "model_name": req.model_name,
            "adapter_name": req.adapter_name,
            "priority": req.priority,
            "batching_mode": req.batching_mode,
        }

    return {
        "job_id": job_id,
        "task_count": len(chunks),
        "model_name": req.model_name,
        "adapter_name": req.adapter_name,
        "batching_mode": req.batching_mode,
    }


@app.post("/submit_training_job")
def submit_training_job(req: SubmitTrainingJobRequest):
    job_id = str(uuid.uuid4())
    job_dir = _training_job_dir(job_id)

    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    zip_bytes = decode_base64_bytes(req.dataset_zip_b64)
    dataset_zip_path = job_dir / req.dataset_zip_name
    dataset_zip_path.write_bytes(zip_bytes)

    extracted_dataset_dir = job_dir / "dataset"
    extract_zip_bytes(zip_bytes, extracted_dataset_dir)

    shard_output_dir = job_dir / "shards"
    shards = build_yolo_shards(extracted_dataset_dir, shard_output_dir, req.worker_count, req.class_names)

    with state_lock:
        training_jobs[job_id] = {
            "job_id": job_id,
            "job_name": req.job_name,
            "status": "queued",
            "created_at": time.time(),
            "finished_at": None,
            "dataset_zip_name": req.dataset_zip_name,
            "dataset_zip_path": str(dataset_zip_path),
            "dataset_root": str(extracted_dataset_dir),
            "base_model": req.base_model,
            "epochs": req.epochs,
            "imgsz": req.imgsz,
            "batch_size": req.batch_size,
            "worker_count": req.worker_count,
            "class_names": req.class_names or ["object"],
            "preferred_models": req.preferred_models,
            "total_rounds": req.epochs,
            "current_round": 0,
            "current_weights_b64": None,
            "round_metrics": [],
            "shards": shards,
            "completed_batches": 0,
            "pending_batches": 0,
        }
        _enqueue_training_round_locked(job_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "rounds": req.epochs,
        "worker_count": req.worker_count,
        "shards": len(shards),
    }


@app.get("/get_batch/{node_id}")
def get_batch(node_id: str):
    with state_lock:
        if node_id not in nodes:
            raise HTTPException(status_code=404, detail="Unknown node_id. Register first.")

        _maintenance_locked()

        node = nodes[node_id]
        if not node["active"] or node["active_batches"] >= node["max_parallel_batches"]:
            return {"batch": None}

        training_batch = _pop_training_batch_from_queue_locked(node)
        if training_batch is not None:
            return {"batch": training_batch}

        batch = _pop_batch_from_queue_locked(node)
        if batch is None:
            return {"batch": None}

        return {"batch": batch}


@app.get("/get_task/{node_id}")
def get_task(node_id: str):
    payload = get_batch(node_id)
    batch = payload.get("batch")
    if batch is None or not batch.get("tasks"):
        return {"task": None}

    task = batch["tasks"][0]
    return {
        "task_id": task["task_id"],
        "job_id": task["job_id"],
        "data": task["data"],
        "attempt": task["attempt"],
    }


@app.post("/submit_batch_result")
def submit_batch_result(req: SubmitBatchResultRequest):
    with state_lock:
        if req.node_id not in nodes:
            raise HTTPException(status_code=404, detail="Unknown node_id")
        if req.batch_id not in batches:
            raise HTTPException(status_code=404, detail="Unknown batch_id")

        batch = batches[req.batch_id]
        if batch["node_id"] != req.node_id:
            raise HTTPException(status_code=409, detail="Batch is not assigned to this node")

        for item in req.results:
            if item.task_id not in tasks:
                raise HTTPException(status_code=404, detail=f"Unknown task_id: {item.task_id}")

            task = tasks[item.task_id]
            if task["job_id"] != item.job_id:
                raise HTTPException(status_code=400, detail="job_id does not match task_id")
            if task["status"] == "done":
                continue

            task["status"] = "done"
            task["output"] = item.output
            task["assigned_at"] = None
            task["assigned_node"] = req.node_id
            task["batch_id"] = req.batch_id

        batch["status"] = "done"
        batch["finished_at"] = time.time()
        nodes[req.node_id]["completed_batches"] += 1
        _release_batch_locked(req.batch_id, requeue=False)

        for item in req.results:
            _update_job_status_locked(item.job_id)

    return {"status": "received", "batch_id": req.batch_id}


@app.post("/submit_training_round_result")
def submit_training_round_result(req: SubmitTrainingRoundResultRequest):
    with state_lock:
        if req.node_id not in nodes:
            raise HTTPException(status_code=404, detail="Unknown node_id")
        if req.batch_id not in training_batches:
            raise HTTPException(status_code=404, detail="Unknown batch_id")

        batch = training_batches[req.batch_id]
        if batch["node_id"] != req.node_id:
            raise HTTPException(status_code=409, detail="Batch is not assigned to this node")
        if batch["round_index"] != req.round_index:
            raise HTTPException(status_code=400, detail="round_index does not match batch")

        batch["weights_b64"] = req.weights_b64
        batch["metrics"] = req.metrics
        batch["status"] = "done"
        batch["finished_at"] = time.time()

        _release_training_batch_locked(req.batch_id, requeue=False)
        _update_training_job_status_locked(batch["job_id"])

    return {"status": "received", "batch_id": req.batch_id}


@app.post("/submit_training_batch_failure")
def submit_training_batch_failure(req: SubmitTrainingBatchFailureRequest):
    with state_lock:
        batch = training_batches.get(req.batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail="Unknown batch_id")
        if batch.get("node_id") != req.node_id:
            raise HTTPException(status_code=409, detail="Batch is not assigned to this node")
        _fail_training_batch_locked(req.batch_id, req.error)
        _update_training_job_status_locked(batch["job_id"])
    return {"status": "recorded", "batch_id": req.batch_id}


@app.post("/submit_result")
def submit_result(req: SubmitBatchResultRequest):
    return submit_batch_result(req)


@app.get("/job_status/{job_id}")
def job_status(job_id: str):
    with state_lock:
        if job_id in training_jobs:
            _maintenance_locked()
            job = training_jobs[job_id]
            round_batches = [
                batch
                for batch in training_batches.values()
                if batch["job_id"] == job_id and batch["round_index"] == job["current_round"]
            ]
            response = {
                "job_id": job_id,
                "status": job["status"],
                "job_name": job["job_name"],
                "type": "training",
                "progress": {
                    "round": job["current_round"],
                    "total_rounds": job["total_rounds"],
                    "done": sum(1 for batch in round_batches if batch.get("status") == "done"),
                    "pending": sum(1 for batch in round_batches if batch.get("status") == "queued"),
                    "assigned": sum(1 for batch in round_batches if batch.get("status") == "assigned"),
                    "failed": sum(1 for batch in round_batches if batch.get("status") == "failed"),
                    "total": len(round_batches),
                    "workers": [
                        {
                            "node_id": node["node_id"],
                            "display_name": node.get("display_name"),
                            "gpu": node.get("gpu"),
                            "epoch": node.get("training_epoch", 0),
                            "total_epochs": node.get("training_total_epochs", 0),
                        }
                        for node in nodes.values()
                        if node.get("active") and node.get("active_batches", 0) > 0
                    ],
                },
                "base_model": job["base_model"],
                "imgsz": job["imgsz"],
                "batch_size": job["batch_size"],
                "worker_count": job["worker_count"],
                "class_names": job["class_names"],
                "round_metrics": job.get("round_metrics", []),
                "errors": [batch.get("error") for batch in round_batches if batch.get("error")],
            }
            if job["status"] == "done":
                response["weights_url"] = f"/training_jobs/{job_id}/weights"
            return response

        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Unknown job_id")

        _maintenance_locked()
        job = jobs[job_id]

        response = {
            "job_id": job_id,
            "status": job["status"],
            "progress": {
                "done": job["done_count"],
                "failed": job["failed_count"],
                "pending": job["pending_count"],
                "total": len(job["task_ids"]),
            },
            "model_name": job.get("model_name"),
            "adapter_name": job.get("adapter_name"),
            "batching_mode": job.get("batching_mode"),
        }

        if job["status"] == "done":
            response["output"] = _job_output_locked(job_id)

        return response


@app.get("/training_status/{job_id}")
def training_status(job_id: str):
    return job_status(job_id)


@app.get("/training_jobs/{job_id}/shards/{shard_index}.zip")
def training_shard(job_id: str, shard_index: int):
    with state_lock:
        if job_id not in training_jobs:
            raise HTTPException(status_code=404, detail="Unknown training job")
        job = training_jobs[job_id]
        if shard_index < 0 or shard_index >= len(job["shards"]):
            raise HTTPException(status_code=404, detail="Unknown shard")
        shard = job["shards"][shard_index]
        shard_zip_path = Path(shard["zip_path"])
        if not shard_zip_path.exists():
            raise HTTPException(status_code=404, detail="Shard archive not found")

    return FileResponse(shard_zip_path, media_type="application/zip", filename=shard_zip_path.name)


@app.get("/training_jobs/{job_id}/weights")
def training_weights(job_id: str):
    with state_lock:
        if job_id not in training_jobs:
            raise HTTPException(status_code=404, detail="Unknown training job")
        job = training_jobs[job_id]
        return {
            "job_id": job_id,
            "base_model": job["base_model"],
            "weights_b64": job.get("current_weights_b64"),
            "round": job["current_round"],
        }


@app.get("/models/{model_name:path}")
def model_file(model_name: str):
    model_path = (BASE_DIR / model_name).resolve()
    if model_path.parent != BASE_DIR or model_path.suffix.lower() != ".pt" or not model_path.is_file():
        raise HTTPException(status_code=404, detail="Model checkpoint not found")
    return FileResponse(model_path, media_type="application/octet-stream", filename=model_path.name)


@app.get("/nodes")
def list_nodes():
    with state_lock:
        _maintenance_locked()
        return {"nodes": nodes}


@app.get("/network")
def network():
    with state_lock:
        _maintenance_locked()
        return {
            "join": "/join",
            "dashboard": "/dashboard",
            "manifest": _network_manifest(),
            "active_nodes": [
                {
                    "node_id": node["node_id"],
                    "display_name": node.get("display_name"),
                    "gpu": node["gpu"],
                    "gpu_memory_mb": node["gpu_memory_mb"],
                    "max_batch_size": node["max_batch_size"],
                    "supports_training": node.get("supports_training", True),
                    "active": node["active"],
                }
                for node in nodes.values()
            ],
            "training_jobs": [
                {
                    "job_id": job["job_id"],
                    "job_name": job["job_name"],
                    "status": job["status"],
                    "round": job["current_round"],
                    "total_rounds": job["total_rounds"],
                }
                for job in training_jobs.values()
            ],
        }


@app.get("/dashboard", include_in_schema=False)
def dashboard():
    if not DASHBOARD_PATH.exists():
        raise HTTPException(status_code=404, detail="dashboard.html not found")
    return FileResponse(DASHBOARD_PATH, headers={"Cache-Control": "no-store"})


@app.get("/strawberry_dataset.zip", include_in_schema=False)
def download_dashboard_dataset():
    """Serve the dataset used by the browser dashboard's comparison workflow."""
    if not DATASET_ZIP_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Dataset ZIP not found at {DATASET_ZIP_PATH}. "
                "Place strawberry_dataset.zip beside server.py or set "
                "GRADMESH_DATASET_ZIP before starting the coordinator."
            ),
        )
    return FileResponse(
        DATASET_ZIP_PATH,
        media_type="application/zip",
        filename="strawberry_dataset.zip",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/metrics")
def metrics():
    with state_lock:
        _maintenance_locked()
        pending_tasks = sum(1 for task in tasks.values() if task["status"] == "pending")
        assigned_tasks = sum(1 for task in tasks.values() if task["status"] == "assigned")
        done_tasks = sum(1 for task in tasks.values() if task["status"] == "done")
        failed_tasks = sum(1 for task in tasks.values() if task["status"] == "failed")
        pending_training_batches = sum(1 for batch in training_batches.values() if batch["status"] == "queued")
        assigned_training_batches = sum(1 for batch in training_batches.values() if batch["status"] == "assigned")
        done_training_batches = sum(1 for batch in training_batches.values() if batch["status"] == "done")

        return {
            "nodes": len(nodes),
            "active_nodes": sum(1 for node in nodes.values() if node["active"]),
            "pending_tasks": pending_tasks,
            "assigned_tasks": assigned_tasks,
            "done_tasks": done_tasks,
            "failed_tasks": failed_tasks,
            "queued_batches": len(batches),
            "allocated_memory_mb": sum(node["allocated_memory_mb"] for node in nodes.values()),
            "active_batches": sum(node["active_batches"] for node in nodes.values()),
            "training_jobs": len(training_jobs),
            "pending_training_batches": pending_training_batches,
            "assigned_training_batches": assigned_training_batches,
            "done_training_batches": done_training_batches,
        }


@app.get("/")
def root():
    return {
        "message": "Decentralized GPU coordinator is running",
        "dashboard": "/dashboard",
        "endpoints": [
            "/register_node",
            "/heartbeat",
            "/submit_job",
            "/submit_training_job",
            "/get_task/{node_id}",
            "/submit_result",
            "/submit_training_round_result",
            "/job_status/{job_id}",
            "/training_status/{job_id}",
            "/nodes",
        ],
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)
