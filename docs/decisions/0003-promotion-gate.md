# 0003. The promotion gate: what "better" means

Status: accepted (2026-10)

## Context

The 2024–25 code promoted a retrained model if its mAP was above 0.85 and above the production
model's stored number. Three things are wrong with that rule:

1. **Different data, different numbers.** The two mAPs could come from different evaluation sets
   and still be compared.
2. **Noise counts as progress.** On a few hundred images, mAP moves by a point or so depending on
   which images happen to be in the set. A "better" model can be luck.
3. **Accuracy is not the only cost.** A model that is 2 points better and 50 % slower can be the
   wrong trade for a CPU-hosted API, and an average can hide one class getting worse.

## Decision

`gate.decide(champion_report, challenger_report, policy)` is a pure function. A challenger is
promoted only if **every** check passes:

| Check | Default | Why this value |
|---|---|---|
| Same hold-out set | fingerprint (hash of the annotation file) must match | Otherwise the comparison means nothing. Reports are cached per version *and* hold-out hash. |
| mAP@[.5:.95] gain | ≥ +1.0 point | The smallest gain worth a deploy, and about 6× the paired noise between two related models (measured below). |
| p95 latency growth | ≤ +15 % | Roughly what one CPU replica can absorb before the replica count has to go up. A product decision, so it is a flag (`--max-latency-increase`). |
| Critical classes | `person` AP may not drop by more than 1 point | The average over 80 classes can rise while the class the business cares about falls. |
| Latency budget (optional) | off; `--latency-budget-ms` | For edge targets, where there is a hard frame budget. |

Both models are scored by the same code that serves them (`Detector.detect`, batch size 1,
CPU), loaded from the registry, on the same hold-out. The full report (mAP, mAP@.5, mAP@.75,
per-class AP, p50/p95 latency, hardware) and the gate's verdict with one line per check are
stored as tags on the model version.

## How the 1-point threshold was measured

`scripts/holdout_noise.py` resamples the 500 hold-out images with replacement 200 times and
scores v1 (DETR-R50) and v3 (v1 fine-tuned) on each resample
([`docs/results/holdout_noise.json`](../results/holdout_noise.json)):

| Quantity | Value |
|---|---|
| Spread of one model's mAP across resamples (std) | **1.35 pt** |
| Spread of the paired difference v3 − v1 (std) | **0.15 pt** |
| 95 % interval of v3 − v1 | −0.1 to +0.5 pt |

Two conclusions:

1. **Unpaired comparisons are useless at this size.** A model's mAP moves by more than a point
   depending on which 500 images are drawn. Comparing against a number measured on other data
   would mostly measure the data. Hence the hard "same hold-out" check.
2. **Paired, the set is sensitive.** Scored on the same images, two related models can be told
   apart at about half a point. The 1-point bar is therefore a materiality decision (a gain big
   enough to pay for a rollout and its risk), not a statistical floor. v3's +0.2 is below it,
   and its interval includes zero anyway.

Caveat: the paired spread was measured on two closely related models. Models with different
architectures disagree on more images, so their paired noise is larger. For a close call between
architectures, re-run the script on that pair before trusting a gain near the bar.

## Consequences

- A rejection is a normal, explained outcome, visible in `detr-lifecycle status` and the MLflow
  UI, not a silent "nothing happened".
- Latency is compared on the machine that ran the evaluation. Champion reports are cached, so
  in production the gate should run on the serving hardware class, or re-measure the champion
  (`evaluate --force`) when the hardware changes.
- Changing the policy changes which models ship. It lives in code and in CLI flags, so it is
  reviewed like code.
