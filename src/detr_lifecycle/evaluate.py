"""Evaluate a registered version on a fixed hold-out set: COCO mAP plus CPU latency.

The report is stored on the model version as tags, so the registry itself carries the evidence
for every promotion decision.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import mlflow
import torch
from torchmetrics.detection import MeanAveragePrecision

from detr_lifecycle.config import Settings
from detr_lifecycle.data import CocoSubset
from detr_lifecycle.model import Detector
from detr_lifecycle.registry import connect, load_detector

WARMUP_IMAGES = 3


@dataclass(frozen=True)
class EvalReport:
    dataset: str
    n_images: int
    map: float  # COCO mAP@[.5:.95]
    map_50: float
    map_75: float
    latency_p50_ms: float
    latency_p95_ms: float
    device: str
    per_class_ap: dict[str, float] = field(default_factory=dict)
    hardware: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> EvalReport:
        return cls(**json.loads(raw))


def _xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = box
    return [x, y, x + w, y + h]


def _percentile(values: list[float], pct: int) -> float:
    if len(values) < 2:
        return round(values[0], 1) if values else 0.0
    return round(statistics.quantiles(values, n=100, method="inclusive")[pct - 1], 1)


Sample = dict[str, torch.Tensor]


@dataclass
class Predictions:
    preds: list[Sample]
    targets: list[Sample]
    latencies_ms: list[float]
    category_names: dict[int, str]


def predict_dataset(
    detector: Detector, dataset: CocoSubset, limit: int | None = None
) -> Predictions:
    """Run the detector image by image (batch size 1, as served), timing each call."""
    records = dataset.records()[:limit]
    # Match classes by name: checkpoints number classes differently (DETR uses COCO ids 1-90,
    # RT-DETR uses 0-79), the names are what the business means.
    category_ids = {c["name"]: c["id"] for c in dataset.load()["categories"]}
    out = Predictions([], [], [], {v: k for k, v in category_ids.items()})
    for i, (info, anns) in enumerate(records):
        image = dataset.open_image(info)
        start = time.perf_counter()
        # Threshold 0: mAP needs the full ranked list, not the served cut-off.
        (dets,) = detector.detect([image], threshold=0.0)
        if i >= WARMUP_IMAGES:
            out.latencies_ms.append((time.perf_counter() - start) * 1000)
        out.preds.append(
            {
                "boxes": torch.tensor([d.box for d in dets], dtype=torch.float32).reshape(-1, 4),
                "scores": torch.tensor([d.score for d in dets], dtype=torch.float32),
                "labels": torch.tensor(
                    [category_ids.get(d.label, -1) for d in dets], dtype=torch.int64
                ),
            }
        )
        out.targets.append(
            {
                "boxes": torch.tensor(
                    [_xywh_to_xyxy(a["bbox"]) for a in anns], dtype=torch.float32
                ).reshape(-1, 4),
                "labels": torch.tensor([a["category_id"] for a in anns], dtype=torch.int64),
                "iscrowd": torch.tensor([a.get("iscrowd", 0) for a in anns], dtype=torch.int64),
                "area": torch.tensor([a["area"] for a in anns], dtype=torch.float32),
            }
        )
    return out


def score(preds: list[Sample], targets: list[Sample]) -> dict[str, torch.Tensor]:
    metric = MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox", class_metrics=True, backend="faster_coco_eval"
    )
    # COCO protocol scores the top 100 detections per image; RT-DETR emits 300 by design.
    metric.warn_on_many_detections = False
    metric.update(preds, targets)
    return metric.compute()


def evaluate_detector(
    detector: Detector, dataset: CocoSubset, limit: int | None = None
) -> EvalReport:
    run = predict_dataset(detector, dataset, limit)
    result = score(run.preds, run.targets)
    per_class = {
        run.category_names.get(int(c), str(int(c))): round(float(ap), 4)
        for c, ap in zip(result["classes"].tolist(), result["map_per_class"].tolist(), strict=True)
        if ap >= 0  # -1 means the class has no ground truth in this set
    }
    return EvalReport(
        dataset=dataset.fingerprint(),
        n_images=len(run.preds),
        map=round(float(result["map"]), 4),
        map_50=round(float(result["map_50"]), 4),
        map_75=round(float(result["map_75"]), 4),
        latency_p50_ms=_percentile(run.latencies_ms, 50),
        latency_p95_ms=_percentile(run.latencies_ms, 95),
        device=detector.device,
        per_class_ap=per_class,
        hardware=f"{platform.system()} {platform.machine()}, {torch.get_num_threads()} threads",
    )


def stored_report(settings: Settings, version: str, dataset: CocoSubset) -> EvalReport | None:
    client = connect(settings)
    mv = client.get_model_version(settings.model_name, version)
    raw = mv.tags.get("eval.report")
    if not raw:
        return None
    report = EvalReport.from_json(raw)
    return report if report.dataset == dataset.fingerprint() else None


def evaluate_version(
    settings: Settings,
    version: str,
    dataset: CocoSubset,
    *,
    device: str = "cpu",
    force: bool = False,
    limit: int | None = None,
) -> EvalReport:
    """Evaluate a registered version, reusing a stored report for the same dataset."""
    if not force and (cached := stored_report(settings, version, dataset)) is not None:
        return cached

    client = connect(settings)
    detector = load_detector(settings, version, device)
    report = evaluate_detector(detector, dataset, limit=limit)

    with mlflow.start_run(run_name=f"eval-v{version}"):
        mlflow.set_tags({"model_version": version, "dataset": report.dataset})
        mlflow.log_params({"device": report.device, "n_images": report.n_images})
        metrics: dict[str, Any] = {
            k: getattr(report, k)
            for k in ("map", "map_50", "map_75", "latency_p50_ms", "latency_p95_ms")
        }
        mlflow.log_metrics(metrics)
        mlflow.log_dict(json.loads(report.to_json()), "eval_report.json")

    for key, value in {
        "eval.dataset": report.dataset,
        "eval.map": str(report.map),
        "eval.map_50": str(report.map_50),
        "eval.latency_p95_ms": str(report.latency_p95_ms),
        "eval.report": report.to_json(),
    }.items():
        client.set_model_version_tag(settings.model_name, version, key, value)
    return report
