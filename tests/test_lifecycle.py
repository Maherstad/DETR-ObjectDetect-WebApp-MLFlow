"""End to end on a tiny model: register, serve, evaluate, retrain, gate, roll back."""

import mlflow
import pandas as pd

from detr_lifecycle.codec import encode_image
from detr_lifecycle.config import CHALLENGER, CHAMPION, PREVIOUS
from detr_lifecycle.evaluate import evaluate_version
from detr_lifecycle.gate import GatePolicy
from detr_lifecycle.pipeline import gate_version, retrain
from detr_lifecycle.registry import connect, get_alias, register_checkpoint, rollback, set_alias
from detr_lifecycle.train import TrainConfig

ALWAYS = GatePolicy(min_map_gain=-1.0, max_latency_increase=100.0, critical_classes=())
NEVER = GatePolicy(min_map_gain=1.0)
SMOKE = TrainConfig(epochs=1, batch_size=2, max_steps=2)


def _bootstrap(settings, tiny_checkpoint):
    mv = register_checkpoint(str(tiny_checkpoint), settings)
    set_alias(connect(settings), settings, CHAMPION, str(mv.version))
    return str(mv.version)


def test_registered_model_serves_json_detections(settings, tiny_checkpoint, holdout):
    version = _bootstrap(settings, tiny_checkpoint)
    model = mlflow.pyfunc.load_model(f"models:/{settings.model_name}@{CHAMPION}")
    image = (holdout.images_dir / holdout.load()["images"][0]["file_name"]).read_bytes()
    out = model.predict(pd.DataFrame({"image": [encode_image(image)]}), params={"threshold": 0.0})
    assert version == "1"
    assert out[0]["width"] == 96 and out[0]["height"] == 64
    assert len(out[0]["detections"]) > 0
    det = out[0]["detections"][0]
    assert set(det) == {"label", "label_id", "score", "box"} and len(det["box"]) == 4


def test_evaluation_is_stored_on_the_version_and_reused(settings, tiny_checkpoint, holdout):
    version = _bootstrap(settings, tiny_checkpoint)
    report = evaluate_version(settings, version, holdout)
    assert report.n_images == 6 and 0.0 <= report.map <= 1.0
    mv = connect(settings).get_model_version(settings.model_name, version)
    assert mv.tags["eval.dataset"] == holdout.fingerprint()
    again = evaluate_version(settings, version, holdout)
    assert again == report  # cached: same dataset fingerprint


def test_retrain_promotes_when_gate_passes_and_rolls_back(
    settings, tiny_checkpoint, holdout, batch
):
    _bootstrap(settings, tiny_checkpoint)
    result = retrain(settings, batch, holdout, SMOKE, ALWAYS, train_device="cpu")
    assert result is not None and result.decision.promote
    client = connect(settings)
    assert str(get_alias(client, settings, CHAMPION).version) == result.version == "2"
    assert str(get_alias(client, settings, PREVIOUS).version) == "1"
    assert get_alias(client, settings, CHALLENGER) is None
    mv = client.get_model_version(settings.model_name, "2")
    assert mv.tags["parent_version"] == "1" and mv.tags["gate.decision"] == "promote"

    # Same batch again: nothing new to learn, the job is a no-op.
    assert retrain(settings, batch, holdout, SMOKE, ALWAYS, train_device="cpu") is None

    assert rollback(client, settings) == "1"
    assert str(get_alias(client, settings, CHAMPION).version) == "1"


def test_rejected_challenger_never_reaches_champion(settings, tiny_checkpoint, holdout):
    _bootstrap(settings, tiny_checkpoint)
    candidate = str(register_checkpoint(str(tiny_checkpoint), settings).version)
    result = gate_version(settings, candidate, holdout, NEVER)
    client = connect(settings)
    assert not result.decision.promote
    assert str(get_alias(client, settings, CHAMPION).version) == "1"
    assert str(get_alias(client, settings, CHALLENGER).version) == candidate
    assert (
        client.get_model_version(settings.model_name, candidate).tags["gate.decision"] == "reject"
    )
