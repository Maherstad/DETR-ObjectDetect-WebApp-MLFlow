"""Pick a fine-tuning recipe on a validation split of the batch, never on the hold-out.

The hold-out decides promotion. Choosing hyperparameters on it as well would leak it into the
decision, so the batch is split 80/20 and recipes are compared on the 20.

    python scripts/sweep_recipes.py --batch data/batches/2026-10-batch-01 --steps 250
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import replace
from pathlib import Path

import mlflow

from detr_lifecycle.config import CHAMPION, Settings
from detr_lifecycle.data import CocoSubset
from detr_lifecycle.evaluate import evaluate_detector
from detr_lifecycle.registry import connect, get_alias, load_detector
from detr_lifecycle.train import TrainConfig, fine_tune

RECIPES = {
    "lr1e-5_no-warmup": TrainConfig(
        lr=1e-5, lr_backbone=1e-6, warmup_steps=0, freeze_backbone=False
    ),
    "lr1e-5_warmup": TrainConfig(lr=1e-5, lr_backbone=1e-6, freeze_backbone=False),
    "lr2e-6_frozen-backbone": TrainConfig(lr=2e-6, freeze_backbone=True),
}


def split(batch: CocoSubset, out: Path, val_fraction: float, seed: int) -> tuple[CocoSubset, ...]:
    data = batch.load()
    ids = sorted(im["id"] for im in data["images"])
    val_ids = set(random.Random(seed).sample(ids, int(len(ids) * val_fraction)))
    parts = []
    for name, keep in (("train", lambda i: i not in val_ids), ("val", lambda i: i in val_ids)):
        root = out / f"{batch.root.name}-{name}"
        root.mkdir(parents=True, exist_ok=True)
        part = {
            **data,
            "images": [im for im in data["images"] if keep(im["id"])],
            "annotations": [a for a in data["annotations"] if keep(a["image_id"])],
        }
        (root / "annotations.json").write_text(json.dumps(part))
        if not (root / "images").exists():
            (root / "images").symlink_to(batch.images_dir.resolve())
        parts.append(CocoSubset(root))
    return tuple(parts)


def main() -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="docs/results/recipe_sweep.json")
    args = parser.parse_args()

    settings = Settings()
    client = connect(settings)
    champion = str(get_alias(client, settings, CHAMPION).version)  # type: ignore[union-attr]
    train, val = split(CocoSubset(Path(args.batch)), Path("data/splits"), 0.2, seed=0)

    results = {}
    with mlflow.start_run(run_name=f"recipe-sweep-v{champion}"):
        mlflow.log_params({"parent_version": champion, "steps": args.steps})
        base = evaluate_detector(load_detector(settings, champion), val)
        results["champion (no training)"] = base.map
        mlflow.log_metric("val_map_champion", base.map)
        for name, cfg in RECIPES.items():
            cfg = replace(cfg, epochs=99, max_steps=args.steps)
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_params(cfg.as_params())
                detector = load_detector(settings, champion, device=args.device)
                loss = fine_tune(detector, train, cfg)
                detector.model.to("cpu")
                detector.device = "cpu"
                val_map = evaluate_detector(detector, val).map
                mlflow.log_metrics({"val_map": val_map, "final_loss": loss})
            results[name] = val_map
            print(f"{name:<26} val mAP {val_map:.4f}  (champion {base.map:.4f})", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(
            {"val_set": val.fingerprint(), "steps": args.steps, "val_map": results}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
