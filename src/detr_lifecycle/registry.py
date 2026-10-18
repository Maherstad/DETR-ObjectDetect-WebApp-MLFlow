"""Model registry: log a detector as a versioned PyFunc and move aliases between versions.

Aliases, not stages: `@champion` is what the server loads, `@challenger` is the candidate under
evaluation, `@previous` is the last champion so a rollback is one alias move.
"""

from __future__ import annotations

import contextlib
import tempfile
from collections.abc import Iterator
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, ParamSchema, ParamSpec, Schema

from detr_lifecycle.config import CHAMPION, PREVIOUS, Settings
from detr_lifecycle.model import DEFAULT_THRESHOLD, Detector, DetrPyFunc

PACKAGE_DIR = Path(__file__).resolve().parent

SIGNATURE = ModelSignature(
    inputs=Schema([ColSpec("string", "image")]),
    params=ParamSchema([ParamSpec("threshold", "double", DEFAULT_THRESHOLD)]),
)


def _requirements() -> list[str]:
    pins = ["mlflow", "torch", "torchvision", "transformers", "timm", "pillow", "pandas"]
    return [f"{name}=={pkg_version(name)}" for name in pins]


def connect(settings: Settings) -> MlflowClient:
    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_experiment(settings.experiment)
    return MlflowClient(settings.tracking_uri)


@contextlib.contextmanager
def _run(run_name: str) -> Iterator[Any]:
    """Log into the active run if there is one, otherwise open a new one."""
    active = mlflow.active_run()
    if active is not None:
        yield active
    else:
        with mlflow.start_run(run_name=run_name) as run:
            yield run


def log_detector(
    detector: Detector,
    settings: Settings,
    *,
    run_name: str,
    tags: dict[str, str] | None = None,
    description: str | None = None,
) -> ModelVersion:
    """Save weights + processor into the model artifact and register a new version.

    The weights travel inside the artifact, so serving never fetches from the Hub and a
    version always means the same bytes.
    """
    client = connect(settings)
    tags = tags or {}
    with tempfile.TemporaryDirectory() as tmp, _run(run_name):
        weights = Path(tmp) / "weights"
        detector.save(weights)
        mlflow.set_tags(tags)
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=DetrPyFunc(),
            artifacts={"weights": str(weights)},
            code_paths=[str(PACKAGE_DIR)],
            signature=SIGNATURE,
            pip_requirements=_requirements(),
            model_config={"threshold": DEFAULT_THRESHOLD, "device": "cpu"},
            registered_model_name=settings.model_name,
        )
    mv_version = str(info.registered_model_version)
    for key, value in tags.items():
        client.set_model_version_tag(settings.model_name, mv_version, key, value)
    if description:
        client.update_model_version(settings.model_name, mv_version, description=description)
    return client.get_model_version(settings.model_name, mv_version)


def register_checkpoint(source: str, settings: Settings) -> ModelVersion:
    """Register a pretrained checkpoint (Hub id or local dir) as a new version."""
    detector = Detector.from_pretrained(source)
    return log_detector(
        detector,
        settings,
        run_name=f"register-{Path(source).name}",
        tags={"source": "checkpoint", "checkpoint": source},
        description=f"Pretrained checkpoint {source}, no fine-tuning.",
    )


def load_detector(settings: Settings, version: str, device: str = "cpu") -> Detector:
    """Load a registered version through MLflow, i.e. exactly what the server would load."""
    connect(settings)
    model = mlflow.pyfunc.load_model(
        f"models:/{settings.model_name}/{version}", model_config={"device": device}
    )
    detector: Detector = model.unwrap_python_model().detector
    return detector


def get_alias(client: MlflowClient, settings: Settings, alias: str) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(settings.model_name, alias)
    except MlflowException:
        return None


def set_alias(client: MlflowClient, settings: Settings, alias: str, version: str) -> None:
    client.set_registered_model_alias(settings.model_name, alias, version)


def promote(client: MlflowClient, settings: Settings, version: str) -> None:
    """Make `version` the champion and keep the old champion as `@previous`."""
    current = get_alias(client, settings, CHAMPION)
    if current is not None and str(current.version) != version:
        set_alias(client, settings, PREVIOUS, str(current.version))
    set_alias(client, settings, CHAMPION, version)


def rollback(client: MlflowClient, settings: Settings) -> str:
    previous = get_alias(client, settings, PREVIOUS)
    if previous is None:
        raise RuntimeError("No @previous version to roll back to.")
    current = get_alias(client, settings, CHAMPION)
    set_alias(client, settings, CHAMPION, str(previous.version))
    if current is not None:
        set_alias(client, settings, PREVIOUS, str(current.version))
    return str(previous.version)
