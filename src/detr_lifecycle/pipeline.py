"""Lifecycle jobs: gate a challenger, and retrain-then-gate when a new labelled batch lands.

Both are idempotent, so a scheduler (cron, a CI workflow, Airflow) can call them blindly.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow

from detr_lifecycle.config import CHALLENGER, CHAMPION, Settings
from detr_lifecycle.data import CocoSubset
from detr_lifecycle.evaluate import EvalReport, evaluate_version
from detr_lifecycle.gate import GateDecision, GatePolicy, decide
from detr_lifecycle.registry import (
    connect,
    get_alias,
    load_detector,
    log_detector,
    promote,
    set_alias,
)
from detr_lifecycle.train import TrainConfig, fine_tune


def _log_loss(step: int, loss: float) -> None:
    mlflow.log_metric("loss", loss, step=step)


@dataclass(frozen=True)
class GateResult:
    version: str
    champion_version: str | None
    champion: EvalReport | None
    challenger: EvalReport
    decision: GateDecision


def gate_version(
    settings: Settings,
    version: str,
    holdout: CocoSubset,
    policy: GatePolicy,
    *,
    apply: bool = True,
    eval_device: str = "cpu",
) -> GateResult:
    """Evaluate `version` against the champion; promote it if every check passes."""
    client = connect(settings)
    champ = get_alias(client, settings, CHAMPION)
    champ_version = str(champ.version) if champ is not None else None
    if champ_version == version:
        raise ValueError(f"v{version} is already the champion.")

    set_alias(client, settings, CHALLENGER, version)
    champion = (
        evaluate_version(settings, champ_version, holdout, device=eval_device)
        if champ_version
        else None
    )
    challenger = evaluate_version(settings, version, holdout, device=eval_device)
    decision = decide(champion, challenger, policy)

    client.set_model_version_tag(
        settings.model_name, version, "gate.decision", "promote" if decision.promote else "reject"
    )
    client.set_model_version_tag(settings.model_name, version, "gate.summary", decision.summary())
    if apply and decision.promote:
        promote(client, settings, version)
        client.delete_registered_model_alias(settings.model_name, CHALLENGER)
    return GateResult(version, champ_version, champion, challenger, decision)


def find_trained_on(settings: Settings, batch: CocoSubset, cfg: TrainConfig) -> str | None:
    """Version already fine-tuned on exactly this batch with exactly this recipe, if any."""
    client = connect(settings)
    fp = batch.fingerprint()
    for mv in client.search_model_versions(f"name='{settings.model_name}'"):
        if mv.tags.get("train.dataset") == fp and mv.tags.get("train.config") == cfg.digest():
            return str(mv.version)
    return None


def retrain(
    settings: Settings,
    batch: CocoSubset,
    holdout: CocoSubset,
    cfg: TrainConfig,
    policy: GatePolicy,
    *,
    train_device: str = "auto",
    eval_device: str = "cpu",
) -> GateResult | None:
    """Fine-tune the champion on `batch`, register the result, and run it through the gate.

    Returns None when this batch was already used with this recipe (nothing new to try).
    """
    if find_trained_on(settings, batch, cfg) is not None:
        return None
    client = connect(settings)
    champ = get_alias(client, settings, CHAMPION)
    if champ is None:
        raise RuntimeError("No @champion to start from. Run `detr-lifecycle bootstrap` first.")

    parent = str(champ.version)
    detector = load_detector(settings, parent, device=train_device)
    with mlflow.start_run(run_name=f"retrain-from-v{parent}"):
        mlflow.log_params(
            {**cfg.as_params(), "parent_version": parent, "train_device": detector.device}
        )
        mlflow.set_tags({"train.dataset": batch.fingerprint(), "train.config": cfg.digest()})
        final_loss = fine_tune(detector, batch, cfg, on_step=_log_loss)
        mlflow.log_metric("final_epoch_loss", final_loss)
        detector.model.to("cpu")
        mv = log_detector(
            detector,
            settings,
            run_name="unused",
            tags={
                "source": "retrain",
                "parent_version": parent,
                "train.dataset": batch.fingerprint(),
                "train.config": cfg.digest(),
            },
            description=f"v{parent} fine-tuned on {batch.fingerprint()}.",
        )
    return gate_version(settings, str(mv.version), holdout, policy, eval_device=eval_device)
