# Changelog

## 0.2.0 (2026-10)

Rebuilt around the model lifecycle.

- **Promotion gate.** A new version only becomes `@champion` if it beats the current one on a
  frozen 500-image hold-out set by more than the set's measured noise, stays within 15 % of its
  p95 latency, and doesn't get worse on `person`. Every decision and its reasons are stored on
  the model version.
- **Real retraining.** `detr-lifecycle retrain` fine-tunes the champion on a labelled batch and
  sends the result through the gate. It replaces `automated_retraining.py` and
  `hyperparameter_optimization.py`, which were scaffolds that logged placeholder metrics.
  Recipes are now compared for real on a validation split of the batch
  (`scripts/sweep_recipes.py`).
- **Fixed: batch padding corrupted training targets.** Padding through the image processor
  re-normalised boxes to the padded canvas; the first retrain lost 15.5 mAP points while its loss
  fell. The gate rejected it, and the collate function now pads without touching the boxes.
- **Registry aliases** (`@champion`, `@challenger`, `@previous`) instead of the deprecated
  `Production` stage; one-command rollback.
- **JSON API.** The model server returns labels, scores and boxes instead of a drawn image.
- **Weights inside the model artifact**, so serving never calls the Hugging Face Hub.
- **One-command start**: `docker compose up` brings up MLflow, registers the first champion,
  starts the model server and the UI. The old compose file could not start (a `curl`
  health check in an image without curl, and a UI hard-wired to `127.0.0.1`).
- Package layout (`src/detr_lifecycle`), `uv`, ruff, mypy, pytest, GitHub Actions. The tests
  run the whole lifecycle on a tiny random DETR against a real MLflow registry in ~7 s.
- MLflow 2.10 → 3.17, transformers 4.37 → 5.20, torch 2.1 → 2.14.

## 0.1.0 (2024-02)

DETR-ResNet-50 wrapped as an MLflow PyFunc, served with `mlflow models serve`, with a
Streamlit page that showed the annotated image.
