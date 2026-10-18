# 0001. Serve a custom PyFunc straight from the MLflow registry

Status: accepted (2024-02, revised 2026-10)

## Context

One model, a small team, and CPU-only hosting. The registry already knows which version is
approved. What we want is the shortest path from "this version is approved" to "this version
answers requests", without a second deployment system to keep in sync.

## Options

| Option | For | Against |
|---|---|---|
| **MLflow PyFunc + `mlflow models serve`** | The registry URI *is* the deployment target (`models:/detr-object-detector@champion`). No glue code. Signature validation for free. | One model per process, batch size 1, no dynamic batching, a restart is needed to pick up a new version. |
| FastAPI service that loads weights itself | Full control over the API, async I/O, easy to add auth. | Re-implements loading, versioning and validation that MLflow already does. A second place where "which model is live" can drift. |
| TorchServe / Triton | Dynamic batching, GPU sharing, gRPC, high throughput. | Heavy to run for one model at a few requests per second. Needs a model-archive step that the registry doesn't know about. |

## Decision

Wrap the Hugging Face model in a `PythonModel` (`DetrPyFunc`) and serve it with
`mlflow models serve`. The PyFunc takes base64 images in a DataFrame and returns JSON
detections (see [0005](0005-json-detections-not-images.md)). Weights and processor config are
stored inside the model artifact, so a version always means the same bytes and the server never
calls the Hugging Face Hub.

## Consequences

- Promotion is an alias move plus a restart of `model-server`. That restart is the deploy step,
  and it is explicit (see [0002](0002-aliases-not-stages.md)).
- Throughput is bounded by one CPU process: about 7–8 images/s per worker at 125–160 ms each.
  Scaling out is more replicas behind a load balancer, which is fine up to a few dozen
  requests per second.
- **When to revisit:** sustained load above ~50 req/s, a GPU in the budget, or a latency target
  under ~50 ms. Then: export to ONNX/TensorRT and move to Triton with dynamic batching, and keep
  the registry as the source of truth by having the deploy job read `@champion`.
