"""Plot every registered version as an operating point: hold-out mAP against p95 CPU latency.

Reads the evaluation and gate tags straight from the registry, so the chart is the registry.
The shaded area is where a challenger passes the default gate against the first champion.

    python scripts/plot_versions.py --out docs/results
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mlflow import MlflowClient

from detr_lifecycle.config import Settings
from detr_lifecycle.gate import GatePolicy

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
              "zone": "#e8f0fb", "baseline": "#52514e", "promoted": "#2a78d6",
              "rejected": "#eb6834"},
    "dark": {"surface": "#0d1117", "ink": "#f0efec", "muted": "#c3c2b7", "grid": "#30363d",
             "zone": "#15243a", "baseline": "#c3c2b7", "promoted": "#3987e5",
             "rejected": "#d95926"},
}  # fmt: skip

NAMES = {
    "facebook/detr-resnet-50": "DETR-R50",
    "facebook/detr-resnet-101": "DETR-R101",
    "PekingU/rtdetr_v2_r50vd": "RT-DETRv2-R50",
}


def _describe(mv: Any) -> tuple[str, str]:
    """(name, outcome) for one version, from its registry tags."""
    if ck := mv.tags.get("checkpoint"):
        name = NAMES.get(ck, ck.split("/")[-1])
    else:
        bug = " (padding bug)" if "note" in mv.tags else ""
        name = f"v{mv.tags.get('parent_version')} fine-tuned{bug}"
    decision = mv.tags.get("gate.decision")
    if decision is None:
        return name, "first champion"
    if decision == "promote":
        return name, "promoted"
    why = []
    for line in mv.tags.get("gate.summary", "").splitlines():
        if "[FAIL]" not in line:
            continue
        if "mAP gain" in line and (m := re.search(r"\(([+-][\d.]+) pt", line)):
            why.append(f"mAP {m.group(1)} pt")
        elif "latency" in line and (m := re.search(r"\(([+-]\d+%)", line)):
            why.append(f"{m.group(1)} latency")
        elif "critical class" in line:
            why.append("person AP drop")
    return name, "rejected: " + ", ".join(why) if why else "rejected"


def main() -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/results")
    args = parser.parse_args()

    settings = Settings()
    client = MlflowClient(settings.tracking_uri)
    versions = sorted(
        (
            mv
            for mv in client.search_model_versions(f"name='{settings.model_name}'")
            if "eval.map" in mv.tags
        ),
        key=lambda v: int(v.version),
    )
    points = [
        (mv, float(mv.tags["eval.latency_p95_ms"]), float(mv.tags["eval.map"]) * 100)
        for mv in versions
    ]
    first = points[0]
    policy = GatePolicy()
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    x_lo, x_hi = min(xs) * 0.85, max(xs) * 1.45
    y_lo, y_hi = min(ys) - 2, max(ys) + 2

    for theme, c in THEMES.items():
        fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=200)
        fig.patch.set_facecolor(c["surface"])
        ax.set_facecolor(c["surface"])

        zone_x = first[1] * (1 + policy.max_latency_increase)
        zone_y = first[2] + policy.min_map_gain * 100
        ax.add_patch(
            Rectangle((x_lo, zone_y), zone_x - x_lo, y_hi - zone_y, color=c["zone"], zorder=0)
        )
        ax.text(
            x_lo + 3, (zone_y + y_hi) / 2 - 1,
            "passes the gate vs v1:\n+1 pt mAP, at most +15% latency",
            ha="left", va="center", linespacing=1.4,
            fontsize=7.5, color=c["muted"],
        )  # fmt: skip

        for mv, x, y in points:
            name, outcome = _describe(mv)
            kind = (
                "baseline"
                if outcome == "first champion"
                else "promoted"
                if outcome == "promoted"
                else "rejected"
            )
            ax.scatter(
                x, y, s=70, color=c[kind], edgecolor=c["surface"], linewidth=2,
                zorder=4 if kind == "baseline" else 3,
            )  # fmt: skip
            # Points that sit on top of each other get their labels pushed apart vertically.
            close = [
                p for p in points if p[0] is not mv and abs(p[1] - x) < 12 and abs(p[2] - y) < 2
            ]
            dy = 0 if not close else (9 if y >= max(p[2] for p in close) else -9)
            ax.annotate(
                f"v{mv.version} {name}\n{outcome}",
                (x, y),
                xytext=(9, dy),
                textcoords="offset points",
                fontsize=7.5,
                color=c["ink"],
                va="center",
                linespacing=1.3,
            )

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(
            "p95 latency per image on CPU (ms), lower is better", color=c["muted"], fontsize=8
        )
        ax.set_ylabel("Hold-out mAP@[.5:.95] (%)", color=c["muted"], fontsize=8)
        ax.tick_params(colors=c["muted"], labelsize=8, length=0)
        ax.grid(color=c["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        out = Path(args.out) / f"operating-points-{theme}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, facecolor=c["surface"])
        print(out)


if __name__ == "__main__":
    main()
