# GPU supplier laptop

This laptop contributes its NVIDIA GPU. It does not need the strawberry dataset, Streamlit, or the coordinator state. It needs the repository's Python worker code and the same Python dependencies.

Copy the whole project folder to this laptop, or at minimum keep these files together:

- `worker.py`
- `federated_training.py`
- `gpu_supplier\worker.py`
- `requirements.txt`

Install dependencies and verify CUDA:

```powershell
pip install -r requirements.txt
python test_cuda.py
```

Find the host laptop's IPv4 address with `ipconfig`. Test connectivity before starting work:

```powershell
Test-NetConnection HOST_IP -Port 8000
```

Start the supplier worker from the repository root:

```powershell
.\gpu_supplier\start-worker.ps1 -HostIp HOST_IP -Name gpu-3050 -MaxBatchSize 2
```

Or run the Python entry point directly:

```powershell
python gpu_supplier\worker.py --server-url http://HOST_IP:8000 --name gpu-3050 --max-batch-size 2
```

Keep this terminal open. A successful startup prints `registering gpu=...` and the host will show this laptop in `python client.py discover --server http://127.0.0.1:8000`.

The worker automatically downloads the selected `.pt` checkpoint and its dataset shard from the host for each training job. Do not use `127.0.0.1` here: that address means the supplier laptop itself, not the host.
