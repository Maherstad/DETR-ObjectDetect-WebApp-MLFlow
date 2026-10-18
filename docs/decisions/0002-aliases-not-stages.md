# 0002. Registry aliases (`@champion`, `@challenger`, `@previous`) instead of stages

Status: accepted (2026-10). Replaces the 2024 setup that moved versions to the `Production` stage.

## Context

The 2024 version registered every model and immediately moved it to the `Production` stage.
Two problems:

1. MLflow deprecated stages in 2.9. They are a fixed set of four labels, and a version can only
   sit in one of them.
2. More importantly, *anything* registered went live. There was no step where a new version had
   to prove itself.

## Decision

Use aliases with clear jobs:

| Alias | Meaning | Who moves it |
|---|---|---|
| `@champion` | What the model server loads | The gate, or a person running `rollback` |
| `@challenger` | The candidate being evaluated | `gate` / `retrain`, removed after a promotion |
| `@previous` | The champion before the last promotion | `promote`, so rollback is one alias move |

The model server resolves `@champion` once, at start-up. It does not poll the registry.

## Consequences

- Promotion and rollback are atomic metadata changes. Nothing is copied or rebuilt.
- A rejected challenger stays registered, with its evaluation and the gate's reasons in its
  version tags, so "why didn't v3 ship?" has an answer in the registry.
- The live model only changes when someone (or a deploy job) restarts the server. That is
  deliberate: a registry write by itself never changes production behaviour. The cost is one
  extra step, which a CD job would automate by watching alias changes.
