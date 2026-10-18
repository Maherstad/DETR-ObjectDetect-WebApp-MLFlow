"""Fine-tune a detector on a labelled COCO-format batch (the standard DETR recipe, scaled down)."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from detr_lifecycle.data import CocoSubset, TrainingSet
from detr_lifecycle.model import Detector


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 1
    batch_size: int = 4
    # Defaults picked by scripts/sweep_recipes.py on a validation split of the batch.
    lr: float = 2e-6
    lr_backbone: float = 1e-6
    weight_decay: float = 1e-4
    max_grad_norm: float = 0.1
    warmup_steps: int = 50
    freeze_backbone: bool = True
    max_steps: int | None = None  # cap for smoke runs
    seed: int = 0

    def as_params(self) -> dict[str, Any]:
        return {f"train.{k}": v for k, v in asdict(self).items()}

    def digest(self) -> str:
        """Identifies the recipe, so the same batch can be retried with a different one."""
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:8]


def make_collate(
    processor: Any, device: str
) -> Callable[[list[tuple[Any, dict[str, Any]]]], dict[str, Any]]:
    def collate(batch: list[tuple[Any, dict[str, Any]]]) -> dict[str, Any]:
        # Encode one image at a time, then pad here. Padding through the processor re-normalises
        # the target boxes to the padded canvas, but DETR predicts boxes relative to the real
        # image; training on the padded convention cost 15 mAP points (docs/decisions/0004).
        encoded = [processor(images=img, annotations=t, return_tensors="pt") for img, t in batch]
        height = max(e["pixel_values"].shape[-2] for e in encoded)
        width = max(e["pixel_values"].shape[-1] for e in encoded)
        pixel_values = torch.zeros(len(encoded), 3, height, width)
        pixel_mask = torch.zeros(len(encoded), height, width, dtype=torch.long)
        for i, e in enumerate(encoded):
            h, w = e["pixel_values"].shape[-2:]
            pixel_values[i, :, :h, :w] = e["pixel_values"][0]
            pixel_mask[i, :h, :w] = 1
        labels = [{k: v.to(device) for k, v in e["labels"][0].items()} for e in encoded]
        return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "labels": labels}

    return collate


def fine_tune(
    detector: Detector,
    dataset: CocoSubset,
    cfg: TrainConfig,
    on_step: Callable[[int, float], None] | None = None,
) -> float:
    """Train in place. Returns the mean loss of the last epoch."""
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    model, processor, device = detector.model, detector.processor, detector.device

    loader = DataLoader(
        TrainingSet(dataset),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=make_collate(processor, device),
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad_(not cfg.freeze_backbone)
    backbone = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    groups = [{"params": rest, "lr": cfg.lr}]
    if backbone:
        groups.append({"params": backbone, "lr": cfg.lr_backbone})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / max(cfg.warmup_steps, 1))
    )

    model.train()
    step, epoch_loss = 0, math.nan
    for _ in range(cfg.epochs):
        losses: list[float] = []
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device) if "pixel_mask" in batch else None
            out = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=batch["labels"])
            optimizer.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            losses.append(out.loss.item())
            if on_step:
                on_step(step, losses[-1])
            step += 1
            if cfg.max_steps and step >= cfg.max_steps:
                break
        epoch_loss = sum(losses) / max(len(losses), 1)
        if cfg.max_steps and step >= cfg.max_steps:
            break
    for param in model.parameters():
        param.requires_grad_(True)
    model.eval()
    return epoch_loss
