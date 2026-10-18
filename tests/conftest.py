"""Fixtures: a tiny randomly initialised DETR and a toy COCO dataset, so CI needs no downloads."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor, ResNetConfig

from detr_lifecycle.config import Settings
from detr_lifecycle.data import CocoSubset

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

LABELS = {0: "N/A", 1: "person", 2: "bicycle", 3: "car"}


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tiny-detr")
    backbone = ResNetConfig(
        embedding_size=8,
        hidden_sizes=[8, 8, 16, 16],
        depths=[1, 1, 1, 1],
        layer_type="basic",
        out_features=["stage4"],
    )
    config = DetrConfig(
        backbone_config=backbone,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
        num_queries=8,
        id2label=LABELS,
        label2id={v: k for k, v in LABELS.items()},
    )
    DetrForObjectDetection(config).save_pretrained(out)
    DetrImageProcessor(size={"shortest_edge": 64, "longest_edge": 96}).save_pretrained(out)
    return out


def _make_dataset(root: Path, n: int, seed: int) -> CocoSubset:
    rng = np.random.default_rng(seed)
    (root / "images").mkdir(parents=True)
    images, annotations = [], []
    for i in range(n):
        img = Image.fromarray(rng.integers(0, 255, (64, 96, 3), dtype=np.uint8))
        x, y = int(rng.integers(0, 50)), int(rng.integers(0, 30))
        ImageDraw.Draw(img).rectangle([x, y, x + 30, y + 20], fill=(255, 0, 0))
        name = f"{seed}_{i}.jpg"
        img.save(root / "images" / name)
        images.append({"id": seed * 1000 + i, "file_name": name, "width": 96, "height": 64})
        annotations.append(
            {
                "id": seed * 1000 + i,
                "image_id": seed * 1000 + i,
                "category_id": 1 if i % 2 else 3,
                "bbox": [x, y, 30, 20],
                "area": 600,
                "iscrowd": 0,
            }
        )
    categories = [{"id": k, "name": v} for k, v in LABELS.items() if k]
    (root / "annotations.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories})
    )
    return CocoSubset(root)


@pytest.fixture(scope="session")
def holdout(tmp_path_factory: pytest.TempPathFactory) -> CocoSubset:
    return _make_dataset(tmp_path_factory.mktemp("data") / "holdout", n=6, seed=1)


@pytest.fixture(scope="session")
def batch(tmp_path_factory: pytest.TempPathFactory) -> CocoSubset:
    return _make_dataset(tmp_path_factory.mktemp("data") / "batch-001", n=4, seed=2)


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.chdir(tmp_path)  # MLflow writes artifacts relative to the working directory
    return Settings(
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        model_name="test-detector",
        experiment="test",
        data_dir=tmp_path,
        device="cpu",
    )
