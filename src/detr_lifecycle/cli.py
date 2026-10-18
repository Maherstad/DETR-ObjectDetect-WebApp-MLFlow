"""`detr-lifecycle` command line: data, register, evaluate, gate, retrain, rollback, status."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING

from detr_lifecycle.config import CHAMPION, Settings

if TYPE_CHECKING:  # heavy imports stay lazy so `--help` is instant
    from detr_lifecycle.data import CocoSubset
    from detr_lifecycle.gate import GatePolicy

DEFAULT_CHECKPOINT = "facebook/detr-resnet-50"
CACHE = Path(os.getenv("DETR_CACHE", "~/.cache/detr-lifecycle")).expanduser()


def _subset(path: str) -> CocoSubset:
    from detr_lifecycle.data import CocoSubset

    subset = CocoSubset(Path(path))
    if not subset.annotations_path.exists():
        raise SystemExit(
            f"{subset.annotations_path} not found. Build it with `detr-lifecycle data`."
        )
    return subset


def _policy(args: argparse.Namespace) -> GatePolicy:
    from detr_lifecycle.gate import GatePolicy

    return GatePolicy(
        min_map_gain=args.min_map_gain,
        max_latency_increase=args.max_latency_increase,
        latency_budget_ms=args.latency_budget_ms,
    )


def cmd_data(args: argparse.Namespace, settings: Settings) -> None:
    from detr_lifecycle.data import (
        CocoSubset,
        build_subset,
        download_images,
        fetch_coco_annotations,
    )

    source = (
        Path(args.source)
        if args.source
        else fetch_coco_annotations(CACHE) / "instances_val2017.json"
    )
    exclude = [CocoSubset(Path(p)) for p in args.exclude]
    subset = build_subset(source, Path(args.out), args.n, args.seed, exclude)
    fetched = download_images(subset)
    print(f"{subset.fingerprint()}: {args.n} images ({fetched} downloaded) in {subset.root}")


def cmd_register(args: argparse.Namespace, settings: Settings) -> None:
    from detr_lifecycle.registry import connect, register_checkpoint, set_alias

    mv = register_checkpoint(args.checkpoint, settings)
    print(f"Registered {args.checkpoint} as {settings.model_name} v{mv.version}")
    if args.alias:
        set_alias(connect(settings), settings, args.alias, str(mv.version))
        print(f"@{args.alias} -> v{mv.version}")


def cmd_bootstrap(args: argparse.Namespace, settings: Settings) -> None:
    """Idempotent first start: register the default checkpoint as champion if there is none."""
    from detr_lifecycle.registry import connect, get_alias, register_checkpoint, set_alias

    client = connect(settings)
    if (champ := get_alias(client, settings, CHAMPION)) is not None:
        print(f"@champion is v{champ.version}, nothing to do.")
        return
    mv = register_checkpoint(args.checkpoint, settings)
    set_alias(client, settings, CHAMPION, str(mv.version))
    print(f"Registered {args.checkpoint} as v{mv.version} and set @champion.")


def cmd_evaluate(args: argparse.Namespace, settings: Settings) -> None:
    from detr_lifecycle.evaluate import evaluate_version

    r = evaluate_version(
        settings,
        args.version,
        _subset(args.holdout),
        device=args.device,
        force=args.force,
        limit=args.limit,
    )
    print(
        f"v{args.version} on {r.dataset} ({r.n_images} images): mAP {r.map:.3f}, "
        f"mAP50 {r.map_50:.3f}, p50 {r.latency_p50_ms:.0f} ms, "
        f"p95 {r.latency_p95_ms:.0f} ms ({r.device})"
    )


def cmd_gate(args: argparse.Namespace, settings: Settings) -> None:
    from detr_lifecycle.pipeline import gate_version

    result = gate_version(
        settings,
        args.version,
        _subset(args.holdout),
        _policy(args),
        apply=not args.dry_run,
        eval_device=args.device,
    )
    print(f"v{result.version} vs champion v{result.champion_version}:")
    print(result.decision.summary())


def cmd_retrain(args: argparse.Namespace, settings: Settings) -> None:
    from detr_lifecycle.pipeline import retrain
    from detr_lifecycle.train import TrainConfig

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        warmup_steps=args.warmup_steps,
        freeze_backbone=args.freeze_backbone,
        max_steps=args.max_steps,
    )
    result = retrain(
        settings,
        _subset(args.batch),
        _subset(args.holdout),
        cfg,
        _policy(args),
        train_device=args.train_device,
        eval_device=args.device,
    )
    if result is None:
        print("This batch was already trained on with this recipe. Nothing to do.")
        return
    print(f"Retrained v{result.champion_version} -> v{result.version}:")
    print(result.decision.summary())


def cmd_rollback(args: argparse.Namespace, settings: Settings) -> None:
    from detr_lifecycle.registry import connect, rollback

    version = rollback(connect(settings), settings)
    print(f"@champion -> v{version}. Restart the model server to serve it.")


def cmd_status(args: argparse.Namespace, settings: Settings) -> None:
    from detr_lifecycle.registry import connect

    client = connect(settings)
    aliases: dict[str, list[str]] = {}
    for alias, version in client.get_registered_model(settings.model_name).aliases.items():
        aliases.setdefault(str(version), []).append(alias)
    versions = sorted(
        client.search_model_versions(f"name='{settings.model_name}'"), key=lambda v: int(v.version)
    )
    print(f"{'ver':>4}  {'aliases':<22} {'source':<44} {'mAP':>6} {'p95 ms':>7}  gate")
    for mv in versions:
        source = mv.tags.get("checkpoint") or f"retrain of v{mv.tags.get('parent_version', '?')}"
        tags = ",".join(sorted(aliases.get(str(mv.version), [])))
        print(
            f"{mv.version:>4}  {tags:<22} {source:<44} "
            f"{mv.tags.get('eval.map', '-'):>6} {mv.tags.get('eval.latency_p95_ms', '-'):>7}  "
            f"{mv.tags.get('gate.decision', '-')}"
        )


def _add_gate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--holdout", default="data/holdout")
    p.add_argument("--device", default="cpu", help="evaluation device (latency is measured here)")
    p.add_argument("--min-map-gain", type=float, default=0.01)
    p.add_argument("--max-latency-increase", type=float, default=0.15)
    p.add_argument("--latency-budget-ms", type=float, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="detr-lifecycle", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data", help="build a reproducible COCO subset and download its images")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source", help="COCO annotation json (default: COCO val2017, downloaded once)")
    p.add_argument(
        "--exclude", nargs="*", default=[], help="subsets whose images must not be reused"
    )
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("register", help="register a pretrained checkpoint as a new version")
    p.add_argument("checkpoint", help="Hugging Face Hub id or local directory")
    p.add_argument("--alias")
    p.set_defaults(func=cmd_register)

    p = sub.add_parser(
        "bootstrap", help="register the default checkpoint as champion if none exists"
    )
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.set_defaults(func=cmd_bootstrap)

    p = sub.add_parser("evaluate", help="score a version on the hold-out set")
    p.add_argument("version")
    p.add_argument("--holdout", default="data/holdout")
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser(
        "gate", help="evaluate a version against the champion and promote if it passes"
    )
    p.add_argument("version")
    p.add_argument("--dry-run", action="store_true")
    _add_gate_args(p)
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("retrain", help="fine-tune the champion on a new batch, then gate it")
    p.add_argument("--batch", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--lr-backbone", type=float, default=1e-6)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--train-device", default="auto")
    _add_gate_args(p)
    p.set_defaults(func=cmd_retrain)

    p = sub.add_parser("rollback", help="point @champion back at @previous")
    p.set_defaults(func=cmd_rollback)

    p = sub.add_parser("status", help="list versions, aliases and their evaluation")
    p.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    args = build_parser().parse_args(argv)
    args.func(args, Settings())


if __name__ == "__main__":
    main()
