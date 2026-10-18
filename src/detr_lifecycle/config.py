"""Settings shared by the CLI, the services and the tests, all overridable from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

CHAMPION = "champion"
CHALLENGER = "challenger"
PREVIOUS = "previous"


@dataclass(frozen=True)
class Settings:
    tracking_uri: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050")
    )
    model_name: str = field(
        default_factory=lambda: os.getenv("DETR_MODEL_NAME", "detr-object-detector")
    )
    experiment: str = field(default_factory=lambda: os.getenv("DETR_EXPERIMENT", "detr-lifecycle"))
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DETR_DATA_DIR", "data")).expanduser()
    )
    device: str = field(default_factory=lambda: os.getenv("DETR_DEVICE", "cpu"))


def resolve_device(requested: str) -> str:
    """Map 'auto' to the best local accelerator; pass anything else through."""
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
