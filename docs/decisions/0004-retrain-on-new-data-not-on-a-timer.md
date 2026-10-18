# 0004. Retrain when new labelled data lands, and never deploy without the gate

Status: accepted (2026-10). Replaces `automated_retraining.py`.

## Context

The 2024–25 retraining script polled a folder on a 24-hour timer and promoted anything above a
fixed mAP of 0.85. It was a scaffold: the training step and its metrics were placeholders. The
design questions behind it are real, though:

- **What triggers a retrain?** A timer retrains on unchanged data and burns compute for nothing.
  A drift alarm needs labels to act on anyway. The thing that actually makes a better model
  possible is new labelled data.
- **What lets a retrained model go live?** A fixed bar ("mAP ≥ 0.85") says nothing about whether
  the new model is better than the one already serving.

## Decision

- `detr-lifecycle retrain --batch <dir>` is one idempotent job: fine-tune the current champion
  on the batch → register the result → evaluate it and the champion on the frozen hold-out set →
  run the gate ([0003](0003-promotion-gate.md)) → promote only if it passes.
- A batch is identified by the hash of its annotation file. The job records it on the new
  version (`train.dataset`) and does nothing if that batch has already been used. So it is safe
  to call from any scheduler, as often as you like.
- The scheduler is left out on purpose. A cron entry, a GitHub Actions `workflow_dispatch`, or an
  Airflow sensor on the labelling tool's export bucket can all call the same command.

## What the first real retrain showed

The first fine-tune of the champion on the 1,000-image batch (registered as v2) had a falling
training loss and looked healthy. On the hold-out it lost **15.5 mAP points**
(0.432 → 0.277), and `person` AP fell from 0.531 to 0.322. The gate rejected it.

The shape of the drop gave the cause away: mAP@0.75 fell much more than mAP@0.5, so the model
was still finding objects but placing the boxes worse. The Hugging Face processor, when it pads
a batch of differently sized images, re-normalises the target boxes to the padded canvas. The
pretrained DETR predicts boxes relative to the real image. Every padded batch was teaching it a
slightly wrong coordinate system. `train.make_collate` now encodes each image on its own and pads
pixels and mask without touching the boxes, and `tests/test_train.py` pins that behaviour.

Two lessons this ADR keeps:

- **Training loss is not evidence.** The only number that decides is the hold-out comparison
  against what is already serving.
- **Pick recipes on the batch, never on the hold-out.** `scripts/sweep_recipes.py` splits the
  batch 80/20 and compares recipes on the 20 %; the hold-out is only touched by the gate. Result
  in [`docs/results/recipe_sweep.json`](../results/recipe_sweep.json): all recipes land within
  about a point of the untrained champion on the validation split, and the default became the
  best of them (lr 2e-6, frozen backbone, 1 epoch).

## Consequences

- No retrain without new information, and no deploy without a comparison.
- Fine-tuning a model that is already strong on a small batch often does *not* help. The gate
  makes that a normal outcome rather than an incident (see the results table in the README).
- Not covered: drift detection on unlabelled production traffic, and active learning to pick
  which images to label next. Both would decide *what* goes into a batch, not how the batch is
  used.
