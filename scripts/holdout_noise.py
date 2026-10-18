"""How big must an mAP gain be before the hold-out set can see it?

Bootstraps the hold-out images (paired: both models scored on the same resample) and reports
the spread of one model's mAP and of the difference between two models. The gate's
`min_map_gain` is set from this.

    python scripts/holdout_noise.py 1 2 --holdout data/holdout --n-boot 200
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import numpy as np
import torch

from detr_lifecycle.config import Settings
from detr_lifecycle.data import CocoSubset
from detr_lifecycle.evaluate import Predictions, predict_dataset, score
from detr_lifecycle.registry import load_detector

CACHE = Path(".mlflow/predictions")


def predictions(settings: Settings, version: str, holdout: CocoSubset) -> Predictions:
    path = CACHE / f"{holdout.fingerprint()}-v{version}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    run = predict_dataset(load_detector(settings, version), holdout)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(run, path)
    return run


def main() -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--holdout", default="data/holdout")
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--out", default="docs/results/holdout_noise.json")
    args = parser.parse_args()

    settings, holdout = Settings(), CocoSubset(Path(args.holdout))
    a, b = predictions(settings, args.a, holdout), predictions(settings, args.b, holdout)
    n = len(a.targets)
    rng = np.random.default_rng(0)
    maps_a, diffs = [], []
    for _ in range(args.n_boot):
        idx = rng.integers(0, n, n)
        targets = [a.targets[i] for i in idx]
        m_a = float(score([a.preds[i] for i in idx], targets)["map"])
        m_b = float(score([b.preds[i] for i in idx], targets)["map"])
        maps_a.append(m_a)
        diffs.append(m_b - m_a)

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    result = {
        "holdout": holdout.fingerprint(),
        "n_images": n,
        "n_boot": args.n_boot,
        "versions": [args.a, args.b],
        "map_a_std": round(statistics.stdev(maps_a), 4),
        "diff_mean": round(statistics.mean(diffs), 4),
        "diff_std": round(statistics.stdev(diffs), 4),
        "diff_ci95": [round(float(lo), 4), round(float(hi), 4)],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
