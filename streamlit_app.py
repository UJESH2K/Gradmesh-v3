import math
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="GPU Weight Comparison", layout="wide")

WEIGHTS_DIR = Path("weights")
TRAINING_JOBS_DIR = Path("training_jobs")
DEFAULT_EVALUATION_YAML = Path("strawberry_dataset_extract") / "data.yaml"
AUTOMATED_RESULTS_PATH = WEIGHTS_DIR / "comparison_results.json"

# ==============================
# Helpers
# ==============================

def list_weight_files():
    if WEIGHTS_DIR.exists():
        return sorted(WEIGHTS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return []


def load_comparison_timings():
    comparison_path = WEIGHTS_DIR / "comparison.json"
    if not comparison_path.exists():
        return {}
    try:
        return json.loads(comparison_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def run_live_training_comparison(server_url, dataset_zip, base_model, epochs, imgsz, batch, max_images):
    command = [
        sys.executable,
        "-u",
        "run_weight_jobs.py",
        "--server",
        server_url.rstrip("/"),
        "--dataset-zip",
        str(dataset_zip),
        "--base-model",
        base_model,
        "--epochs",
        str(epochs),
        "--imgsz",
        str(imgsz),
        "--batch",
        str(batch),
    ]
    if max_images > 0:
        command.extend(["--max-images", str(max_images)])
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines = []
    started_at = time.perf_counter()
    status_box = st.empty()
    log_box = st.empty()

    while process.poll() is None:
        line = process.stdout.readline()
        if line:
            output_lines.append(line.rstrip())
        elapsed = time.perf_counter() - started_at
        phase = output_lines[-1] if output_lines else "Starting training comparison..."
        status_box.info(f"Live training: {elapsed:.1f} seconds\n\n{phase}")
        log_box.code("\n".join(output_lines[-12:]))
        time.sleep(0.2)

    remaining = process.stdout.read()
    if remaining:
        output_lines.extend(remaining.splitlines())
    elapsed = time.perf_counter() - started_at
    log_box.code("\n".join(output_lines[-20:]))

    if process.returncode != 0:
        status_box.error(f"Training comparison failed after {elapsed:.1f} seconds.")
        st.error("\n".join(output_lines[-10:]))
        return None

    status_box.success(f"Training comparison completed in {elapsed:.1f} seconds.")
    return output_lines

def extract_job_id(weight_path):
    if not weight_path:
        return None
    match = re.search(r"([0-9a-f\-]{36})\.pt$", weight_path.name)
    return match.group(1) if match else None

def infer_data_yaml(weight_path):
    job_id = extract_job_id(weight_path)
    if not job_id:
        return None
    candidate = TRAINING_JOBS_DIR / job_id / "dataset" / "data.yaml"
    return candidate if candidate.exists() else None

# ==============================
# Dataset Fix
# ==============================

def fix_dataset_yaml(data_yaml_path):
    import yaml

    data_yaml_path = Path(data_yaml_path)
    if not data_yaml_path.is_file():
        raise FileNotFoundError(f"Evaluation YAML is not a file: {data_yaml_path}")

    with data_yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    dataset_root = data_yaml_path.parent

    train_path = dataset_root / "train" / "images"
    val_path = dataset_root / "valid" / "images"

    st.write("🔍 Dataset Debug")
    st.write("Train:", train_path, train_path.exists())
    st.write("Val:", val_path, val_path.exists())

    if not train_path.exists():
        raise FileNotFoundError(f"Train folder missing: {train_path}")

    if not val_path.exists():
        st.warning("⚠️ No validation dataset → using TRAIN as VAL")
        val_path = train_path

    data["train"] = str(train_path.resolve())
    data["val"] = str(val_path.resolve())

    fixed_yaml = dataset_root / "fixed_data.yaml"

    with fixed_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    return fixed_yaml

# ==============================
# Safe Model Loader
# ==============================

def load_model_safely(weight_path, data_yaml):
    import torch
    from ultralytics.nn.tasks import DetectionModel
    import yaml

    try:
        return YOLO(str(weight_path))
    except Exception:
        st.warning("⚠️ Auto-load failed → rebuilding model")

        with open(data_yaml, "r") as f:
            nc = yaml.safe_load(f).get("nc", 80)

        model_cfg = "yolov8n.yaml"
        model = YOLO(model_cfg)
        model.model = DetectionModel(cfg=model_cfg, nc=nc)

        ckpt = torch.load(weight_path, map_location="cpu")

        ckpt_state = ckpt.get("model", ckpt)
        model_state = model.model.state_dict()

        compatible = {
            k: v for k, v in ckpt_state.items()
            if k in model_state and v.shape == model_state[k].shape
        }

        model.model.load_state_dict(compatible, strict=False)

        return model

# ==============================
# Evaluation
# ==============================

@st.cache_data(show_spinner=False)
def evaluate(weight_path, data_yaml, imgsz, batch, device, conf, iou):
    start = time.perf_counter()

    fixed_yaml = fix_dataset_yaml(data_yaml)
    model = load_model_safely(weight_path, data_yaml)

    results = model.val(
        data=str(fixed_yaml),
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf,
        iou=iou,
        plots=False,
        save=False,
        verbose=False,
    )

    elapsed = time.perf_counter() - start

    box = getattr(results, "box", None)
    speed = getattr(results, "speed", {}) or {}

    return {
        "weight": Path(weight_path).name,
        "precision": float(getattr(box, "mp", math.nan)) if box else math.nan,
        "recall": float(getattr(box, "mr", math.nan)) if box else math.nan,
        "map50": float(getattr(box, "map50", math.nan)) if box else math.nan,
        "map50_95": float(getattr(box, "map", math.nan)) if box else math.nan,
        "eval_time": elapsed,
        "inference_ms": float(speed.get("inference", math.nan)),
    }


def automated_comparison(timings: dict, evaluation_yaml: Path) -> pd.DataFrame:
    """Evaluate the two checkpoints created by the runner without user input."""
    job1 = timings.get("one_worker_job_id")
    job2 = timings.get("two_worker_job_id")
    if not job1 or not job2:
        raise RuntimeError("Training completed but the runner did not provide both job IDs.")

    weight1 = WEIGHTS_DIR / f"weighted_with_one_{job1}.pt"
    weight2 = WEIGHTS_DIR / f"weighted_with_two_{job2}.pt"
    missing = [str(path) for path in (weight1, weight2) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Training completed but checkpoint(s) are missing: {', '.join(missing)}")
    if not evaluation_yaml.is_file():
        raise FileNotFoundError(f"Automatic evaluation YAML not found: {evaluation_yaml}")

    try:
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    with st.spinner("Evaluating the 1-GPU checkpoint..."):
        one_worker = evaluate(str(weight1), str(evaluation_yaml), int(timings["imgsz"]), int(timings["batch"]), device, 0.001, 0.6)
    with st.spinner("Evaluating the 2-GPU checkpoint..."):
        two_worker = evaluate(str(weight2), str(evaluation_yaml), int(timings["imgsz"]), int(timings["batch"]), device, 0.001, 0.6)

    result = pd.DataFrame([one_worker, two_worker])
    result["label"] = ["1 GPU", "2 GPUs"]
    result["train_time"] = [float(timings["one_worker_seconds"]), float(timings["two_worker_seconds"])]
    result["evaluation_device"] = device
    return result


def save_automated_results(timings: dict, result: pd.DataFrame) -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    AUTOMATED_RESULTS_PATH.write_text(
        json.dumps({"timings": timings, "results": result.to_dict(orient="records")}, indent=2),
        encoding="utf-8",
    )


def load_automated_results() -> pd.DataFrame | None:
    if not AUTOMATED_RESULTS_PATH.is_file():
        return None
    try:
        return pd.DataFrame(json.loads(AUTOMATED_RESULTS_PATH.read_text(encoding="utf-8"))["results"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None


def render_comparison(result: pd.DataFrame) -> None:
    st.subheader("Automatic 1-GPU vs 2-GPU Comparison")
    st.dataframe(result, use_container_width=True, hide_index=True)

    one_time, two_time = result["train_time"].iloc[0], result["train_time"].iloc[1]
    speedup = one_time / two_time if two_time else 0.0
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("1-GPU training time", f"{one_time:.2f} s")
    metric2.metric("2-GPU training time", f"{two_time:.2f} s")
    metric3.metric("Measured speedup", f"{speedup:.2f}x")

    time_chart = alt.Chart(result).mark_bar().encode(
        x=alt.X("label:N", title="Run"),
        y=alt.Y("train_time:Q", title="Training time (seconds)"),
        color="label:N",
        tooltip=["label", "train_time", "map50", "map50_95"],
    )
    st.altair_chart(time_chart, use_container_width=True)

    accuracy_chart = alt.Chart(result).mark_bar().encode(
        x=alt.X("label:N", title="Run"),
        y=alt.Y("map50:Q", title="mAP@50"),
        color="label:N",
        tooltip=["label", "map50", "map50_95", "precision", "recall"],
    )
    st.altair_chart(accuracy_chart, use_container_width=True)

# ==============================
# UI
# ==============================

st.title("🚀 Federated GPU Weight Comparison")

st.subheader("Live Training Comparison")
st.caption("The runner measures both jobs automatically. Keep one worker active for the first run and both workers active for the second run.")
live_server = st.text_input("Coordinator URL", value="http://127.0.0.1:8000")
live_dataset = st.text_input("Training Dataset ZIP", value="strawberry_dataset.zip")
live_model = st.text_input("Training Base Model", value="yolov8n.pt")
live_col1, live_col2, live_col3 = st.columns(3)
with live_col1:
    live_epochs = st.number_input("Training Rounds", min_value=1, max_value=1000, value=2)
with live_col2:
    live_imgsz = st.number_input("Training Image Size", min_value=32, max_value=4096, value=512)
with live_col3:
    live_batch = st.number_input("Per-worker Batch", min_value=1, max_value=128, value=2)
live_max_images = st.number_input(
    "Training Images (0 = full strawberry dataset)",
    min_value=0,
    max_value=10000,
    value=20,
    step=5,
)

if st.button("Start Live Training Comparison"):
    if not Path(live_dataset).exists():
        st.error(f"Dataset ZIP not found: {live_dataset}")
    elif not Path(live_model).exists():
        st.error(f"Base model not found: {live_model}")
    else:
        completed_logs = run_live_training_comparison(
            live_server,
            live_dataset,
            live_model,
            live_epochs,
            live_imgsz,
            live_batch,
            live_max_images,
        )
        if completed_logs is not None:
            try:
                timings = load_comparison_timings()
                result = automated_comparison(timings, DEFAULT_EVALUATION_YAML)
                save_automated_results(timings, result)
                st.success(f"Training and automatic comparison complete. Saved: {AUTOMATED_RESULTS_PATH}")
            except Exception as exc:
                st.error(f"Training completed, but automatic evaluation failed: {type(exc).__name__}: {exc}")
            st.rerun()

latest_results = load_automated_results()
if latest_results is not None and not latest_results.empty:
    render_comparison(latest_results)
    st.caption(f"Saved automatically to {AUTOMATED_RESULTS_PATH}.")
else:
    st.info("Start the live comparison once. The checkpoints, timings, validation metrics, and charts will appear here automatically.")
st.stop()

weights = list_weight_files()

if not weights:
    st.error("No weights found")
    st.stop()

timing_record = load_comparison_timings()

col1, col2 = st.columns(2)

with col1:
    w1 = st.selectbox("Checkpoint A", weights, format_func=lambda x: x.name)

with col2:
    w2 = st.selectbox("Checkpoint B", weights, format_func=lambda x: x.name)

yaml_auto = infer_data_yaml(w1) or infer_data_yaml(w2)

data_yaml = st.text_input("Dataset YAML", value=str(yaml_auto) if yaml_auto else "")

imgsz = st.number_input("Image Size", 64, 1280, 640)
batch = st.number_input("Batch", 1, 64, 4)
conf = st.slider("Confidence", 0.0, 1.0, 0.001)
iou = st.slider("IoU", 0.1, 1.0, 0.6)

device = st.selectbox("Device", ["cpu", "0"])

# ==============================
# Training Time + Labels
# ==============================

st.subheader("⏱ Measured Training Time")

col_t1, col_t2 = st.columns(2)

with col_t1:
    train_time_1 = float(timing_record.get("one_worker_seconds", 0.0))
    st.metric("1-worker time", f"{train_time_1:.2f} seconds" if train_time_1 else "Not measured")

with col_t2:
    train_time_2 = float(timing_record.get("two_worker_seconds", 0.0))
    st.metric("2-worker time", f"{train_time_2:.2f} seconds" if train_time_2 else "Not measured")

if timing_record:
    st.caption("Loaded measured timings from weights/comparison.json.")

st.subheader("🏷️ Label Runs")

col_l1, col_l2 = st.columns(2)

with col_l1:
    label_1 = st.text_input("Label for A", value="1 GPU")

with col_l2:
    label_2 = st.text_input("Label for B", value="2 GPU")

# ==============================
# RUN
# ==============================

if st.button("Run Comparison"):

    if not Path(data_yaml).exists():
        st.error("Dataset YAML not found")
        st.stop()

    with st.spinner("Evaluating A..."):
        m1 = evaluate(str(w1), data_yaml, imgsz, batch, device, conf, iou)

    with st.spinner("Evaluating B..."):
        m2 = evaluate(str(w2), data_yaml, imgsz, batch, device, conf, iou)

    df = pd.DataFrame([m1, m2])

    # Add training time + labels
    df["train_time"] = [train_time_1, train_time_2]
    df["label"] = [label_1, label_2]

    # ==============================
    # Metrics Table
    # ==============================
    st.subheader("📊 Metrics")
    st.dataframe(df)

    # ==============================
    # Training Time Chart
    # ==============================
    st.subheader("📊 Training Time Comparison")

    train_chart = alt.Chart(df).mark_bar().encode(
        x="label:N",
        y="train_time:Q",
        color="label:N"
    )

    st.altair_chart(train_chart, use_container_width=True)

    # ==============================
    # Accuracy vs Training Time
    # ==============================
    st.subheader("📈 Accuracy vs Training Time")

    acc_chart = alt.Chart(df).mark_circle(size=200).encode(
        x="train_time:Q",
        y="map50:Q",
        color="label:N",
        tooltip=["label", "train_time", "map50"]
    )

    st.altair_chart(acc_chart, use_container_width=True)

    # ==============================
    # Speed Comparison
    # ==============================
    st.subheader("⚡ Speed Comparison")

    speed_df = df[["label", "eval_time", "inference_ms"]].melt(id_vars=["label"])

    speed_chart = alt.Chart(speed_df).mark_bar().encode(
        x="variable",
        y="value",
        color="label",
        xOffset="label"
    )

    st.altair_chart(speed_chart, use_container_width=True)

    # ==============================
    # Speedup Insight
    # ==============================
    st.subheader("🚀 Training Performance Insight")

    if len(df) == 2:
        t1 = df.iloc[0]["train_time"]
        t2 = df.iloc[1]["train_time"]

        faster = df.iloc[0]["label"] if t1 < t2 else df.iloc[1]["label"]
        speedup = max(t1, t2) / min(t1, t2)

        st.metric("Training Speedup", f"{speedup:.2f}x")
        st.write(f"⚡ Faster Run: {faster}")
        st.write(f"⏱ Time Saved: {abs(t1 - t2):.2f} seconds")
