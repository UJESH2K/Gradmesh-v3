# Local GPU Scheduler MVP

This project is now a terminal-first GPU allocation and batching demo that you can run on a LAN.

Current LAN IP on this machine: `10.62.184.114`

Use that IP in the commands below when your friends connect from the same Wi-Fi or Ethernet network.

It includes:
- A coordinator server in [server.py](server.py) that tracks GPU nodes, batches compatible requests, and applies simple allocation rules
- Worker agents in [worker.py](worker.py) that register local GPU capacity, pull batches, run batched inference, and report results
- A CLI client in [client.py](client.py) for discovery, joining, job submission, and status checks
- A browser dashboard in [dashboard.html](dashboard.html) that is optional and not needed for the main flow

For the two-laptop setup, use the deployment guides in [host](host/README.md) and [gpu_supplier](gpu_supplier/README.md). They are launch guides around the shared implementation, not separate copies of the coordinator and worker logic.

## Local Architecture

1. A client submits a job with `text`, `model_name`, `adapter_name`, `priority`, and `batching_mode`.
2. The coordinator splits the text into micro tasks.
3. Workers register their local GPU memory and max batch size using the join endpoint.
4. When a worker asks for work, the coordinator selects a compatible batch using the model/adapter cache key and the node's memory budget.
5. The worker processes the batch in one pass and submits a batch result list.
6. The coordinator aggregates task outputs back into the job result.
7. If a node stops heartbeating, its batch is re-queued after the lease timeout.

## Allocation Logic

The scheduler uses simple but useful rules for a local demo:
- Prefer compatible nodes that already declare the requested model in `preferred_models`
- Limit batch size by `batching_mode` and node `max_batch_size`
- Respect the node's advertised GPU memory budget
- Re-queue stale batches when heartbeats stop

This gives you a concrete architecture to explain in a demo without pretending to be a full production cluster.

## Fast Start

If you only want the shortest possible path, do this:

1. Start the coordinator on your machine.

```powershell
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

2. Have your friend join from their machine.

```powershell
python client.py join --server http://10.62.184.114:8000 --gpu-name "NVIDIA RTX 4090" --display-name "friend-pc" --run
```

3. Submit a job from your machine.

```powershell
python client.py submit --server http://10.62.184.114:8000 --text "this is a distributed gpu system demo" --model-name demo-model --batching-mode balanced
```

If you want to see who is connected, run:

```powershell
python client.py discover --server http://10.62.184.114:8000
```

## LAN Join Flow

If your friends are on the same Wi-Fi or Ethernet network, the flow is:

1. You start the coordinator on your machine.
2. They point their terminal at your machine's LAN IP.
3. They run the join command to advertise their GPU.
4. Your scheduler starts handing their machine batches.

Use `python client.py discover --server http://YOUR_IP:8000` to show the join endpoint and the worker command.

## Endpoints

- `GET /network`
- `POST /join`
- `POST /register_node`
- `POST /heartbeat`
- `POST /submit_job`
- `GET /get_batch/{node_id}`
- `POST /submit_batch_result`
- `GET /job_status/{job_id}`
- `GET /nodes`
- `GET /metrics`

## Setup on Windows

1. Activate your virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run Locally

1. Start the coordinator:

```powershell
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

If you want friends to connect from the same network, use your LAN IP instead of `127.0.0.1`:

```powershell
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

2. Discover the join endpoint:

```powershell
python client.py discover --server http://127.0.0.1:8000
```

3. Join a worker from another terminal or another machine on the LAN:

```powershell
python client.py join --server http://YOUR_IP:8000 --gpu-name "NVIDIA RTX 4090" --display-name "friend-pc" --run
```

4. Submit a job:

```powershell
python client.py submit --server http://127.0.0.1:8000 --text "this is a distributed gpu system demo" --model-name demo-model --batching-mode balanced
```

## Worker Mode

If you want a worker process without the CLI wrapper, run:

```powershell
python worker.py
```

## Testing the Allocation Path

1. Start 2 workers with different `--gpu-memory-mb` or `--max-batch-size` values.
2. Submit several jobs quickly with the same `--model-name`.
3. Watch the coordinator batch compatible tasks together.
4. Check `python client.py discover --server http://YOUR_IP:8000` or `GET /metrics` to see active batches and allocated memory.

## Notes


## Synchronized Training (Implemented)

This repo currently supports synchronized multi-worker training with a coordinator barrier per round:

1. Coordinator creates one shard batch per worker for the current round.
2. Every worker trains exactly one local epoch on its shard.
3. Workers submit updated weights.
4. Coordinator averages all returned weights (FedAvg-style state-dict averaging).
5. Next round starts only after all shard batches for the current round are complete.

This keeps workers synchronized at round boundaries and prevents model drift across workers.

### Important

- FastAPI is used as a job orchestrator and aggregation barrier.
- This is round-synchronized training, not per-step DDP all-reduce.
- For true step-level gradient sync, use PyTorch DDP (`torchrun`, NCCL) directly.

## Two-Laptop YOLO Training

The training coordinator uses HTTP polling and a round barrier. It is not PyTorch DDP: each worker trains one local epoch on its shard, uploads its state dict, and the coordinator averages the returned weights before starting the next round.

### Host laptop

Find the host LAN address with `ipconfig`, then allow the coordinator port through Windows Firewall (run PowerShell as Administrator):

```powershell
New-NetFirewallRule -DisplayName "GradMesh Coordinator 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

Start the coordinator from this project directory. `0.0.0.0` is required so another laptop can reach it:

```powershell
uvicorn server:app --host 0.0.0.0 --port 8000
```

Confirm locally:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/network
```

### Second laptop

Install the same `requirements.txt`, then verify that its NVIDIA driver and PyTorch can see the RTX 3050 or GTX 1650:

```powershell
python test_cuda.py
```

Replace `HOST_IP` with the host laptop's IPv4 address. The worker automatically downloads `yolov8n.pt` or `yolov8n-obb.pt` from the coordinator, so the checkpoint does not need to be copied manually:

```powershell
python worker.py --server-url http://HOST_IP:8000 --name remote-3050 --max-batch-size 2
```

On the host, start a second worker if the host GPU should participate too:

```powershell
python worker.py --server-url http://127.0.0.1:8000 --name host-1650 --max-batch-size 2
```

Check that both nodes are active before submitting training:

```powershell
python client.py discover --server http://127.0.0.1:8000
```

For the first test, use conservative settings for the 4 GB GTX 1650. The `worker_count` must equal the number of active workers, and `batch` is the per-worker batch size, not the combined batch:

```powershell
python run_weight_jobs.py --server http://127.0.0.1:8000 --dataset-zip strawberry_dataset.zip --base-model yolov8n-obb.pt --epochs 2 --imgsz 512 --batch 2
```

Create the zip first if it does not exist:

```powershell
Compress-Archive -Path strawberry_dataset_extract\* -DestinationPath strawberry_dataset.zip -Force
```

The 1-worker run is completed first, then the 2-worker run. Both timings are wall-clock times measured by `run_weight_jobs.py`; include dataset transfer, shard download, training, weight upload, and synchronization. Do not compare runs with different `epochs`, `imgsz`, `batch`, model, or dataset.

If the remote worker does not appear, test the port from the second laptop:

```powershell
Test-NetConnection HOST_IP -Port 8000
```

If `TcpTestSucceeded` is false, fix the Windows Firewall rule or ensure both laptops are on the same non-isolated Wi-Fi network. If it is true but the worker fails, inspect its terminal for the first HTTP or CUDA error.

## 10-Epoch Comparison Command

Use this command to run 1-worker and 2-worker training sequentially, save both final weights, and print timing:

```powershell
python run_weight_jobs.py --server http://127.0.0.1:8000 --dataset-zip strawberry_dataset.zip --base-model yolov8n.pt --epochs 10 --imgsz 640 --batch 4
```

Output files are saved as:

- `weights/weighted_with_one_<job_id>.pt`
- `weights/weighted_with_two_<job_id>.pt`

#   g p u - i n t e r s e c t i o n 
 
 