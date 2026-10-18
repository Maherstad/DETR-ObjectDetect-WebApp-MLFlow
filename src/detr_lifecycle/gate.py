"""The promotion gate: a pure function from two eval reports to a decision with reasons.

Kept free of MLflow and torch so the rules are unit-tested on their own and easy to read in review.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from detr_lifecycle.evaluate import EvalReport


@dataclass(frozen=True)
class GatePolicy:
    # Smallest mAP gain worth a deploy. Scored on the same 500 images, two related models differ by
    # ~0.15 pt of noise (std, paired bootstrap), so 1 pt is a business bar well clear of noise.
    # See docs/decisions/0003-promotion-gate.md and scripts/holdout_noise.py.
    min_map_gain: float = 0.01
    # p95 latency may grow by at most this fraction over the champion.
    max_latency_increase: float = 0.15
    # Absolute p95 budget per image on the serving hardware. None = no absolute budget.
    latency_budget_ms: float | None = None
    # Classes the business cannot afford to get worse at, whatever the average does.
    critical_classes: tuple[str, ...] = ("person",)
    max_critical_drop: float = 0.01


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class GateDecision:
    promote: bool
    checks: list[Check] = field(default_factory=list)

    def summary(self) -> str:
        verdict = "PROMOTE" if self.promote else "REJECT"
        lines = [verdict] + [
            f"  [{'pass' if c.passed else 'FAIL'}] {c.name}: {c.detail}" for c in self.checks
        ]
        return "\n".join(lines)


def decide(champion: EvalReport | None, challenger: EvalReport, policy: GatePolicy) -> GateDecision:
    checks: list[Check] = []

    if policy.latency_budget_ms is not None:
        checks.append(
            Check(
                "latency budget",
                challenger.latency_p95_ms <= policy.latency_budget_ms,
                f"p95 {challenger.latency_p95_ms:.0f} ms, budget {policy.latency_budget_ms:.0f} ms",
            )
        )

    if champion is None:
        checks.append(Check("champion", True, "no champion yet, first version goes live"))
        return GateDecision(all(c.passed for c in checks), checks)

    if champion.dataset != challenger.dataset:
        checks.append(
            Check(
                "same hold-out set",
                False,
                f"champion scored on {champion.dataset}, challenger on {challenger.dataset}",
            )
        )
        return GateDecision(False, checks)

    gain = challenger.map - champion.map
    checks.append(
        Check(
            "mAP gain",
            gain >= policy.min_map_gain,
            f"{champion.map:.3f} -> {challenger.map:.3f} ({gain * 100:+.1f} pt, "
            f"need {policy.min_map_gain * 100:+.1f})",
        )
    )

    if champion.latency_p95_ms > 0:
        growth = challenger.latency_p95_ms / champion.latency_p95_ms - 1
        checks.append(
            Check(
                "latency regression",
                growth <= policy.max_latency_increase,
                f"p95 {champion.latency_p95_ms:.0f} -> {challenger.latency_p95_ms:.0f} ms "
                f"({growth:+.0%}, max {policy.max_latency_increase:+.0%})",
            )
        )

    for name in policy.critical_classes:
        before = champion.per_class_ap.get(name)
        after = challenger.per_class_ap.get(name)
        if before is None or after is None:
            continue
        drop = before - after
        checks.append(
            Check(
                f"critical class '{name}'",
                drop <= policy.max_critical_drop,
                f"AP {before:.3f} -> {after:.3f}",
            )
        )

    return GateDecision(all(c.passed for c in checks), checks)
