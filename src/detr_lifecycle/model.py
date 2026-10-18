"""The detector and its MLflow PyFunc wrapper.

`Detector` is the only place that runs the network. Serving, evaluation and training all go
through it, so the number the gate sees comes from the same code path the API serves.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from mlflow.pyfunc import PythonModel
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from detr_lifecycle.codec import Detection, decode_image
from detr_lifecycle.config import resolve_device

DEFAULT_THRESHOLD = 0.7


class Detector:
    def __init__(self, model: Any, processor: Any, device: str = "cpu") -> None:
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.processor = processor

    @classmethod
    def from_pretrained(cls, source: str | Path, device: str = "cpu") -> Detector:
        """Load from a Hugging Face Hub id or a local `save_pretrained` directory."""
        processor = AutoImageProcessor.from_pretrained(str(source))
        model = AutoModelForObjectDetection.from_pretrained(str(source))
        return cls(model, processor, device)

    @property
    def id2label(self) -> dict[int, str]:
        return {int(k): v for k, v in self.model.config.id2label.items()}

    @torch.inference_mode()
    def detect(self, images: list[Image.Image], threshold: float) -> list[list[Detection]]:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=[img.size[::-1] for img in images]
        )
        labels = self.id2label
        return [
            [
                Detection(
                    label=labels.get(int(label), str(int(label))),
                    label_id=int(label),
                    score=float(score),
                    box=tuple(float(v) for v in box.tolist()),  # type: ignore[arg-type]
                )
                for score, label, box in zip(r["scores"], r["labels"], r["boxes"], strict=True)
            ]
            for r in results
        ]

    def save(self, path: Path) -> None:
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)


class DetrPyFunc(PythonModel):
    """Served by `mlflow models serve`.

    Input: a DataFrame with a base64 `image` column. Optional param `threshold`.
    Output: one record per image with its size and a list of detections.
    """

    def load_context(self, context: Any) -> None:
        config = context.model_config or {}
        device = os.getenv("DETR_DEVICE", config.get("device", "cpu"))
        self.detector = Detector.from_pretrained(context.artifacts["weights"], device)
        self.default_threshold = float(config.get("threshold", DEFAULT_THRESHOLD))

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        threshold = float((params or {}).get("threshold", self.default_threshold))
        images = [decode_image(b64) for b64 in model_input["image"]]
        detections = self.detector.detect(images, threshold)
        return [
            {"width": img.width, "height": img.height, "detections": [d.to_dict() for d in dets]}
            for img, dets in zip(images, detections, strict=True)
        ]
