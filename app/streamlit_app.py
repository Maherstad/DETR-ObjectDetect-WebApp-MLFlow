"""Demo UI. Talks to the model server over HTTP like any other client, and reads the registry
to show which version is live and how it scored on the hold-out set."""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import requests
import streamlit as st
from PIL import Image, ImageDraw

from detr_lifecycle.codec import encode_image, image_to_bytes

MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://127.0.0.1:7100")
ASSETS = Path(__file__).resolve().parent.parent / "assets"
SAMPLES = ["street.jpg", "safari.jpg", "bedroom.jpg"]
PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#008080", "#9a6324", "#800000"]


def detect(image_bytes: bytes, threshold: float) -> tuple[list[dict], float]:
    payload = {
        "dataframe_records": [{"image": encode_image(image_bytes)}],
        "params": {"threshold": threshold},
    }
    start = time.perf_counter()
    resp = requests.post(f"{MODEL_SERVER_URL}/invocations", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["predictions"][0]["detections"], (time.perf_counter() - start) * 1000


def draw(image: Image.Image, detections: list[dict]) -> Image.Image:
    out = image.copy()
    pen = ImageDraw.Draw(out)
    width = max(2, image.width // 300)
    for det in detections:
        color = PALETTE[det["label_id"] % len(PALETTE)]
        x0, y0, x1, y1 = det["box"]
        pen.rectangle([x0, y0, x1, y1], outline=color, width=width)
        caption = f"{det['label']} {det['score']:.2f}"
        tx0, ty0, tx1, ty1 = pen.textbbox((x0, y0), caption)
        pen.rectangle([tx0 - 2, ty0 - 2, tx1 + 2, ty1 + 2], fill=color)
        pen.text((x0, y0), caption, fill="white")
    return out


@st.cache_data(ttl=30)
def live_model() -> dict | None:
    """The registry's view of what is live. Best effort: the demo works without it."""
    try:
        from mlflow import MlflowClient

        from detr_lifecycle.config import CHAMPION, Settings

        settings = Settings()
        mv = MlflowClient(settings.tracking_uri).get_model_version_by_alias(
            settings.model_name, CHAMPION
        )
        return {"version": mv.version, **mv.tags}
    except Exception:
        return None


st.set_page_config(page_title="DETR lifecycle demo", layout="wide")

with st.sidebar:
    st.subheader("Input")
    choice = st.radio("Image", ["Sample", "Upload"], horizontal=True, label_visibility="collapsed")
    if choice == "Sample":
        name = st.selectbox("Sample image", SAMPLES)
        raw: bytes | None = (ASSETS / name).read_bytes()
    else:
        upload = st.file_uploader("JPEG or PNG", type=["jpg", "jpeg", "png"])
        raw = upload.getvalue() if upload else None
    threshold = st.slider("Confidence threshold", 0.1, 0.95, 0.7, 0.05)

    st.subheader("Live model")
    model = live_model()
    if model:
        st.markdown(f"`@champion` = **v{model['version']}**")
        st.caption(model.get("checkpoint") or f"fine-tuned from v{model.get('parent_version')}")
        if "eval.map" in model:
            left, right = st.columns(2)
            left.metric("Hold-out mAP", f"{float(model['eval.map']):.3f}")
            right.metric("p95, CPU", f"{float(model['eval.latency_p95_ms']):.0f} ms")
    else:
        st.caption("Registry not reachable; showing detections only.")

st.markdown("#### Object detection served from `models:/detr-object-detector@champion`")

if raw:
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    try:
        detections, ms = detect(raw, threshold)
    except requests.RequestException as exc:
        st.error(f"Model server at {MODEL_SERVER_URL} did not answer: {exc}")
        st.stop()
    left, right = st.columns([3, 2])
    annotated = draw(image, detections)
    left.image(annotated, width="stretch")
    right.caption(f"{len(detections)} objects at threshold {threshold:.2f}, {ms:.0f} ms round trip")
    right.dataframe(
        [{"label": d["label"], "score": d["score"], "box": d["box"]} for d in detections],
        hide_index=True,
        height=min(36 * (len(detections) + 1) + 2, 420),
    )
    right.download_button(
        "Download annotated image", image_to_bytes(annotated, "PNG"), "detections.png", "image/png"
    )
