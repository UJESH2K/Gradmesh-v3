import math
import json
import re
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="GPU Weight Comparison", layout="wide")

WEIGHTS_DIR = Path("weights")
TRAINING_JOBS_DIR = Path("training_jobs")

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

    with open(data_yaml_path, "r") as f:
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

    with open(fixed_yaml, "w") as f:
        yaml.dump(data, f)

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

# ==============================
# UI
# ==============================

st.title("🚀 Federated GPU Weight Comparison")

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

st.subheader("⏱ Training Time Input")

col_t1, col_t2 = st.columns(2)

with col_t1:
    train_time_1 = st.number_input(
        "Training Time A (seconds)",
        value=float(timing_record.get("one_worker_seconds", 155.0)),
    )

with col_t2:
    train_time_2 = st.number_input(
        "Training Time B (seconds)",
        value=float(timing_record.get("two_worker_seconds", 145.0)),
    )

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