# Host laptop

The host owns the coordinator, fixed strawberry dataset, base model, experiment runner, and Streamlit dashboard.

From the repository root, install `requirements.txt`, then allow TCP 8000 through Windows Firewall once:

```powershell
New-NetFirewallRule -DisplayName "GradMesh Coordinator 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

Start the coordinator:

```powershell
.\host\start-coordinator.ps1
```

Start the dashboard in a second terminal:

```powershell
.\host\start-dashboard.ps1
```

After the supplier worker appears in `python client.py discover --server http://127.0.0.1:8000`, run the experiment from the repository root:

```powershell
python run_weight_jobs.py --server http://127.0.0.1:8000 --dataset-zip strawberry_dataset.zip --base-model yolov8n.pt --epochs 2 --imgsz 512 --batch 2
```

The host must retain `server.py`, `federated_training.py`, the strawberry dataset, and the selected `.pt` model. The coordinator sends shards and the base model to workers.
