# GradMesh

> **Collaborative GPU orchestration for distributed AI training over a
> local network.**

GradMesh turns multiple consumer GPUs on different machines into a
coordinated training cluster. One machine acts as the **Coordinator**
--- the control-plane "brain" --- while other machines run lightweight
**Worker Agents** that contribute GPU compute.

The current prototype focuses on **synchronized YOLO training across
NVIDIA GPUs on a LAN**. The coordinator handles worker registration, job
creation, dataset sharding, round scheduling, health monitoring, weight
aggregation, and recovery. Workers perform the actual model training on
their local GPUs.

The long-term research direction is a decentralized GPU marketplace, but
the current implementation deliberately focuses on validating the
coordinator → worker → synchronized-training pipeline first.

## Architecture

``` text
                    USER / CLI
                        |
                        v
              +-------------------+
              |     COORDINATOR   |
              |     FastAPI       |
              |-------------------|
              | Job Manager       |
              | Scheduler         |
              | Worker Registry   |
              | Health Monitor    |
              | Round Barrier     |
              | Aggregator        |
              +---------+---------+
                        |
             +----------+----------+
             |          |          |
             v          v          v
          Worker 1   Worker 2   Worker N
             |          |          |
             v          v          v
           GPU 1      GPU 2      GPU N
             |          |          |
             +----------+----------+
                        |
                 synchronized rounds
                        |
                        v
                  Final Weights
```

The coordinator CPU is responsible for **control and orchestration**.
GPUs perform the expensive neural-network computation.

## Components

### Coordinator --- `server.py`

Central control plane responsible for worker registration, discovery,
heartbeats, job creation, dataset sharding, round scheduling, leases,
aggregation, status, metrics, and recovery.

### Worker --- `worker.py`

Runs on a GPU machine. It detects local GPU/runtime information,
registers with the coordinator, receives a shard and checkpoint, trains
locally, uploads the resulting state, and waits for the next round.

### CLI --- `client.py`

Provides discovery, worker joining, job submission, and status
operations without requiring the browser dashboard.

### Training --- `run_weight_jobs.py` / `federated_training.py`

Runs the controlled YOLO experiments and training/aggregation helpers.

### Dashboard --- `dashboard.html`

Optional visual control surface. It is not the training engine; the
coordinator and workers perform the actual training workflow.

## Process Manager — `run.sh`

`run.sh` starts the coordinator and local worker processes in the
background, records their PIDs, and keeps separate logs under
`.gradmesh/`. Run it from the repository root using a POSIX shell such as
Linux/macOS `sh`, WSL, or Git Bash on Windows.

Start the coordinator and one local worker:

``` sh
sh run.sh start
```

Inspect or stop the managed stack:

``` sh
sh run.sh status
sh run.sh logs server
sh run.sh logs local-worker-1
sh run.sh stop
```

The complete command set is:

``` text
sh run.sh start
sh run.sh start server
sh run.sh start workers [COUNT]
sh run.sh start worker NAME
sh run.sh stop
sh run.sh stop server
sh run.sh stop workers
sh run.sh stop worker NAME
sh run.sh restart
sh run.sh status
sh run.sh logs [server|WORKER_NAME]
```

### Multiple workers on one GPU

Set `WORKER_COUNT` to run multiple independent worker processes against
the same locally detected CUDA or XPU device:

``` sh
GRADMESH_WORKER_COUNT=3 GRADMESH_WORKER_BACKEND=cuda sh run.sh start
```

Or add individually named workers:

``` sh
sh run.sh start server
GRADMESH_WORKER_BACKEND=xpu sh run.sh start worker arc-test-1
GRADMESH_WORKER_BACKEND=xpu sh run.sh start worker arc-test-2
```

Each managed worker receives a stable, unique node ID based on its name,
its own PID file, and its own log. Starting the same name twice is
idempotent; the script reports the already-running process instead of
creating a duplicate.

All workers on one machine still share the same physical GPU and VRAM.
They are separate scheduler participants, not isolated GPU partitions.
Use a conservative `MAX_BATCH_SIZE`—usually `1` while testing—and watch
GPU memory to avoid oversubscription:

``` sh
GRADMESH_WORKER_COUNT=3 GRADMESH_WORKER_BACKEND=cuda GRADMESH_MAX_BATCH_SIZE=1 sh run.sh start
```

Useful configuration variables:

``` text
GRADMESH_PYTHON           Python executable; auto-detects .venv when unset
GRADMESH_HOST             Coordinator bind address (default: 0.0.0.0)
GRADMESH_PORT             Coordinator port (default: 8000)
GRADMESH_SERVER_URL       URL workers use (default: http://127.0.0.1:$GRADMESH_PORT)
GRADMESH_WORKER_COUNT     Workers started by `start` (default: 1)
GRADMESH_WORKER_BACKEND   auto, cuda, xpu, or cpu (default: auto)
GRADMESH_MAX_BATCH_SIZE   Per-worker advertised batch limit (default: 2)
GRADMESH_HEARTBEAT_SECONDS Worker heartbeat interval (default: 5)
GRADMESH_POLL_SECONDS     Worker work-poll interval (default: 1.5)
GRADMESH_RUNTIME_DIR      Optional PID/log directory override
```

For workers on other computers, run `run.sh` on each machine with the
coordinator's LAN URL:

``` sh
GRADMESH_SERVER_URL=http://HOST_IP:8000 GRADMESH_WORKER_BACKEND=cuda sh run.sh start worker remote-3050
```

`run.sh stop` only stops processes recorded in its own runtime directory;
it does not kill unrelated Python or GPU processes.

## Distributed Training Model

The current implementation uses **round-synchronized training**, not
step-level PyTorch DDP.

``` text
Coordinator
     |
     +---- shard 1 ----> Worker 1 ----> GPU 1
     |
     +---- shard 2 ----> Worker 2 ----> GPU 2
     |
     +---- shard N ----> Worker N ----> GPU N
                                      |
                                      v
                              Local training
                                      |
                                      v
                              Updated weights
                                      |
              <-----------------------+
              |
              v
       Coordinator aggregation
              |
              v
        Global model state
              |
              v
          Next round
```

Workers train one local epoch per round, return their model state, and
the coordinator performs **FedAvg-style weighted state-dict averaging**
before beginning the next round.

> **Important:** GradMesh currently uses FastAPI/HTTP as the
> orchestration and synchronization layer. It is not PyTorch DDP. For
> true step-level gradient synchronization, PyTorch Distributed with
> `torchrun` and an appropriate backend such as NCCL should be used.

## GPU and CUDA

The current prototype primarily targets NVIDIA GPUs.

PyTorch checks CUDA availability with:

``` python
torch.cuda.is_available()
```

and can select:

``` python
device = torch.device("cuda")
```

The execution path is:

``` text
Python → PyTorch → CUDA Runtime → NVIDIA Driver → GPU / VRAM
```

Python coordinates the work; optimized PyTorch/CUDA kernels execute the
heavy tensor operations on the GPU.

Future worker backends may support AMD/ROCm or Apple Silicon/MPS, but
the first implementation should remain focused on NVIDIA/CUDA for
predictable benchmarking.

### Intel XPU worker

Intel workers use native PyTorch XPU and a separate dependency profile so
the CUDA requirements cannot replace the XPU build:

``` powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-xpu.txt
.\.venv\Scripts\python.exe test_accelerator.py --backend xpu
```

Start an Intel worker explicitly to prevent fallback:

``` powershell
.\.venv\Scripts\python.exe worker.py --server-url http://HOST_IP:8000 --name arc-worker --backend xpu
```

`ultralytics==8.4.46` rejects the strings `xpu` and `xpu:0`, so GradMesh
passes a validated `torch.device("xpu:0")` through a small custom trainer.
The initial XPU path disables AMP and final Ultralytics validation, and uses
`foreach=False` for both Adam-family optimizers and gradient clipping. CUDA
workers continue through the original Ultralytics trainer unchanged.

Validation commands:

``` powershell
.\.venv\Scripts\python.exe validate_yolo_xpu.py
.\.venv\Scripts\python.exe validate_gradmesh_xpu.py --rounds 3
```

## LAN Setup

### Host

Find the host's private LAN IPv4:

``` powershell
ipconfig
```

Allow port 8000 in an Administrator PowerShell:

``` powershell
New-NetFirewallRule -DisplayName "GradMesh Coordinator 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

Start the coordinator:

``` powershell
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Second Laptop

Activate the environment and install dependencies:

``` powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Check CUDA:

``` powershell
python test_cuda.py
```

Start the worker:

``` powershell
python worker.py --server-url http://HOST_IP:8000 --name remote-gpu --max-batch-size 2
```

Example:

``` powershell
python worker.py --server-url http://192.168.1.25:8000 --name remote-3050 --max-batch-size 2
```

If the host GPU should participate, start another worker on the host:

``` powershell
python worker.py --server-url http://127.0.0.1:8000 --name host-gpu --max-batch-size 2
```

Check workers:

``` powershell
python client.py discover --server http://127.0.0.1:8000
```

Test connectivity from the second laptop:

``` powershell
Test-NetConnection HOST_IP -Port 8000
```

Use the host's **private LAN IPv4**, not its public internet address,
for the normal same-network setup.

## YOLO Dataset

A typical dataset ZIP should contain:

``` text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

The included strawberry labels use standard YOLO detection boxes
(`class x_center y_center width height`), so use a detection model such as
`yolov8n.pt`, not an OBB model.

Always validate the dataset with a single-GPU run before using it for
the distributed benchmark.

## Two-GPU Experiment

The main proof-of-concept compares:

### Test 1 --- Single GPU

``` text
Coordinator → GPU 1
```

### Test 2 --- GradMesh

``` text
              Coordinator
               /       \
              v         v
            GPU 1     GPU 2
```

Keep these identical between runs:

-   dataset
-   model
-   image size
-   batch size
-   epochs
-   optimizer/hyperparameters
-   evaluation procedure

Measure:

-   total wall-clock time
-   epoch time
-   training/validation loss
-   mAP or task-specific accuracy
-   GPU utilization
-   VRAM usage
-   network transfer
-   synchronization time
-   worker completion time

Do not assume two GPUs will produce exactly 2× speedup. Communication,
serialization, synchronization, and stragglers create overhead.

## Example Training

``` powershell
python run_weight_jobs.py --server http://127.0.0.1:8000 --dataset-zip strawberry_dataset.zip --base-model yolov8n.pt --epochs 2 --imgsz 512 --batch 2
```

A larger controlled comparison:

``` powershell
python run_weight_jobs.py --server http://127.0.0.1:8000 --dataset-zip strawberry_dataset.zip --base-model yolov8n.pt --epochs 10 --imgsz 640 --batch 4
```

The batch size is the **per-worker batch size** in the current design.

## Output Structure

``` text
outputs/
├── Test_01/
│   ├── single_gpu/
│   │   ├── weights/
│   │   │   ├── best.pt
│   │   │   └── last.pt
│   │   ├── metrics.json
│   │   ├── training_log.csv
│   │   └── system_metrics.csv
│   │
│   └── gradmesh_2gpu/
│       ├── weights/
│       │   ├── best.pt
│       │   └── last.pt
│       ├── metrics.json
│       ├── training_log.csv
│       └── system_metrics.csv
│
└── comparison/
    ├── comparison.csv
    ├── comparison.json
    └── plots/
        ├── training_time.png
        ├── loss.png
        ├── map.png
        └── gpu_utilization.png
```

Streamlit can later read these outputs and visualize the comparison
without controlling the training process.

## Mathematical Model

### Dataset Allocation

``` text
D = union(D_i)
|D_i| = (C_i / sum(C_j)) |D|
```

where `C_i` is normalized compute capability.

### Worker Scheduling Score

``` text
G_i = alpha*C_i + beta*M_i + gamma*H_i + lambda*T_i - delta*L_i
W* = argmax(G_i)
```

This is a proposed scheduling heuristic combining compute capability,
memory, health, reliability, and latency.

### Weighted Aggregation

``` text
w^(t+1) = sum_i [ |D_i| / |D| * w_i^(t+1) ]
```

This is a FedAvg-style aggregation approach rather than a newly invented
optimization algorithm.

### Dynamic Rebalancing

``` text
R_t = |D| - sum_i |D_i_completed|

D_i' = [ G_i / sum_(j in A) G_j ] R_t
```

This represents redistribution of unfinished work among active workers.

### Performance

``` text
Speedup = T_single / T_distributed
Efficiency = Speedup / N
Accuracy Difference = |A_single - A_distributed|
```

## API Endpoints

``` text
GET  /network
POST /join
POST /register_node
POST /heartbeat

POST /submit_job
GET  /get_batch/{node_id}
POST /submit_batch_result

GET  /job_status/{job_id}
GET  /nodes
GET  /metrics
```

The exact request/response schema in `server.py` is the source of truth.

## Project Structure

``` text
GradMesh/
├── run.sh
├── server.py
├── worker.py
├── client.py
├── run_weight_jobs.py
├── federated_training.py
├── dashboard.html
├── test_cuda.py
├── requirements.txt
├── outputs/
├── host/
└── gpu_supplier/
```

## Current Status

### Implemented

-   [x] FastAPI coordinator
-   [x] Worker registration
-   [x] Worker discovery
-   [x] Heartbeat monitoring
-   [x] Job submission
-   [x] Dataset sharding
-   [x] Worker-side GPU training
-   [x] Round-based synchronization
-   [x] FedAvg-style weight aggregation
-   [x] CLI workflow
-   [x] Browser dashboard
-   [x] LAN worker connectivity
-   [x] Training output handling
-   [x] Coordinator/worker process manager

### Validation in Progress

-   [ ] Reliable two-machine YOLO benchmark
-   [ ] Single-GPU vs 2-GPU speedup
-   [ ] Accuracy equivalence
-   [ ] Communication overhead
-   [ ] Heterogeneous GPU allocation
-   [ ] Worker failure and recovery
-   [ ] Dynamic rebalancing

### Future Research

-   [ ] True step-level gradient synchronization
-   [ ] gRPC internal communication
-   [ ] NCCL / PyTorch Distributed integration
-   [ ] Heterogeneous scheduling
-   [ ] Gradient/update compression
-   [ ] Straggler mitigation
-   [ ] Secure worker isolation
-   [ ] NAT traversal
-   [ ] Worker verification
-   [ ] Decentralized marketplace
-   [ ] Reputation and settlement
-   [ ] Privacy-preserving training

## Limitations

GradMesh is currently a **research prototype**, not a production GPU
marketplace.

Important limitations include:

1.  HTTP/round synchronization is slower than specialized GPU
    communication.
2.  Model-weight aggregation is not equivalent to step-level DDP
    gradient synchronization.
3.  Heterogeneous GPUs can create stragglers.
4.  Large checkpoints and updates can create network overhead.
5.  Worker trust and malicious-result verification are not solved in the
    prototype.
6.  Internet deployment requires additional security and NAT traversal
    work.
7.  The first implementation is primarily designed for NVIDIA/CUDA
    environments.

## Research Objective

The central question is:

> **Can heterogeneous, independently owned consumer GPUs be coordinated
> over an ordinary network to reduce deep-learning training time while
> maintaining comparable model quality, without requiring
> datacenter-grade infrastructure?**

The initial experiment is deliberately controlled:

``` text
                    SAME DATASET
                         |
             +-----------+-----------+
             |                       |
             v                       v
       Single GPU              GradMesh 2 GPU
             |                       |
             v                       v
         YOLO Train               YOLO Train
             |                       |
             +-----------+-----------+
                         |
                         v
                    COMPARISON
                         |
              +----------+----------+
              |          |          |
              v          v          v
             Time       mAP       Overhead
```

The objective is to measure the real trade-off between computation,
communication, synchronization, and model quality.

## Long-Term Vision

``` text
LAN GPU Collaboration
          |
          v
Heterogeneous GPU Scheduling
          |
          v
Fault-Aware Distributed Training
          |
          v
Remote GPU Providers
          |
          v
Decentralized GPU Marketplace
          |
          v
Collaborative AI Compute Network
```

> **GradMesh aims to make unused GPU compute accessible, coordinated,
> and useful without requiring every user to own a dedicated high-end
> machine.**

## License

Add the project's intended license before public release.

## Disclaimer

GradMesh is a research and engineering prototype. Performance depends on
GPU capability, network bandwidth/latency, model size, dataset size,
synchronization strategy, and workload characteristics. Benchmark
results should be reported with the exact hardware, software, dataset,
and training configuration used.
