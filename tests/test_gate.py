from dataclasses import replace

from detr_lifecycle.evaluate import EvalReport
from detr_lifecycle.gate import GatePolicy, decide

CHAMPION = EvalReport(
    dataset="holdout@abc",
    n_images=500,
    map=0.420,
    map_50=0.620,
    map_75=0.440,
    latency_p50_ms=140,
    latency_p95_ms=160,
    device="cpu",
    per_class_ap={"person": 0.540, "car": 0.410},
)
POLICY = GatePolicy()


def _failed(decision):
    return {c.name for c in decision.checks if not c.passed}


def test_promotes_clear_improvement():
    challenger = replace(CHAMPION, map=0.440, latency_p95_ms=170)
    decision = decide(CHAMPION, challenger, POLICY)
    assert decision.promote, decision.summary()


def test_rejects_gain_below_threshold():
    challenger = replace(CHAMPION, map=0.425)
    decision = decide(CHAMPION, challenger, POLICY)
    assert not decision.promote
    assert _failed(decision) == {"mAP gain"}


def test_rejects_slower_model_even_if_more_accurate():
    challenger = replace(CHAMPION, map=0.450, latency_p95_ms=260)
    decision = decide(CHAMPION, challenger, POLICY)
    assert _failed(decision) == {"latency regression"}


def test_rejects_when_critical_class_regresses():
    challenger = replace(CHAMPION, map=0.450, per_class_ap={"person": 0.500, "car": 0.600})
    decision = decide(CHAMPION, challenger, POLICY)
    assert _failed(decision) == {"critical class 'person'"}


def test_refuses_to_compare_different_holdout_sets():
    challenger = replace(CHAMPION, dataset="holdout@def", map=0.9)
    decision = decide(CHAMPION, challenger, POLICY)
    assert not decision.promote
    assert _failed(decision) == {"same hold-out set"}


def test_first_version_needs_only_the_budget():
    assert decide(None, CHAMPION, POLICY).promote
    tight = GatePolicy(latency_budget_ms=100)
    assert not decide(None, CHAMPION, tight).promote


def test_summary_lists_every_check():
    challenger = replace(CHAMPION, map=0.440)
    text = decide(CHAMPION, challenger, POLICY).summary()
    assert text.startswith("PROMOTE")
    assert "mAP gain" in text and "latency regression" in text and "person" in text
