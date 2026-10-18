"""COCO-format datasets: a frozen hold-out set for the gate and labelled batches for retraining.

A dataset is a folder with `annotations.json` (COCO format) and `images/`. Its fingerprint is the
hash of the annotation file. The gate refuses to compare two models evaluated on different data.
"""

from __future__ import annotations

import hashlib
import json
import random
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


@dataclass(frozen=True)
class CocoSubset:
    root: Path

    @property
    def annotations_path(self) -> Path:
        return self.root / "annotations.json"

    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    def load(self) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(self.annotations_path.read_text())
        return data

    def fingerprint(self) -> str:
        digest = hashlib.sha256(self.annotations_path.read_bytes()).hexdigest()[:12]
        return f"{self.root.name}@{digest}"

    def records(self) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
        """(image info, annotations) per image, in a stable order."""
        data = self.load()
        by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in data["annotations"]:
            by_image[ann["image_id"]].append(ann)
        images = sorted(data["images"], key=lambda im: im["id"])
        return [(im, by_image[im["id"]]) for im in images]

    def open_image(self, info: dict[str, Any]) -> Image.Image:
        return Image.open(self.images_dir / info["file_name"]).convert("RGB")


def fetch_coco_annotations(cache_dir: Path) -> Path:
    """Download the COCO 2017 annotation zip once and extract the val/train instance files."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / "annotations"
    if not (out / "instances_val2017.json").exists():
        archive = cache_dir / "annotations_trainval2017.zip"
        if not archive.exists():
            urllib.request.urlretrieve(COCO_ANNOTATIONS_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extract("annotations/instances_val2017.json", cache_dir)
    return out


def build_subset(
    source: Path,
    out_dir: Path,
    n: int,
    seed: int,
    exclude: list[CocoSubset] | None = None,
) -> CocoSubset:
    """Sample `n` images (with at least one object) from a COCO annotation file, reproducibly."""
    data = json.loads(source.read_text())
    excluded = {im["id"] for subset in exclude or [] for im in subset.load()["images"]}
    with_objects = {a["image_id"] for a in data["annotations"]}
    candidates = sorted(
        im["id"] for im in data["images"] if im["id"] in with_objects and im["id"] not in excluded
    )
    if n > len(candidates):
        raise ValueError(f"Asked for {n} images, only {len(candidates)} available.")
    chosen = set(random.Random(seed).sample(candidates, n))

    subset = {
        "info": {
            "description": f"COCO subset: {n} images from {source.name}, seed {seed}",
            "source": source.name,
            "seed": seed,
        },
        "licenses": data.get("licenses", []),
        "categories": data["categories"],
        "images": [im for im in data["images"] if im["id"] in chosen],
        "annotations": [a for a in data["annotations"] if a["image_id"] in chosen],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "annotations.json").write_text(json.dumps(subset, separators=(",", ":")))
    return CocoSubset(out_dir)


def download_images(subset: CocoSubset, workers: int = 16) -> int:
    """Fetch the subset's images from their `coco_url`. Already downloaded files are skipped."""
    subset.images_dir.mkdir(parents=True, exist_ok=True)
    todo = [
        im for im in subset.load()["images"] if not (subset.images_dir / im["file_name"]).exists()
    ]

    def fetch(im: dict[str, Any]) -> None:
        urllib.request.urlretrieve(im["coco_url"], subset.images_dir / im["file_name"])

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(fetch, todo))
    return len(todo)


class TrainingSet(Dataset):
    """Yields (image, target) in the format Hugging Face DETR processors expect."""

    def __init__(self, subset: CocoSubset) -> None:
        self.subset = subset
        self.items = [(im, anns) for im, anns in subset.records() if anns]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[Image.Image, dict[str, Any]]:
        info, anns = self.items[idx]
        target = {
            "image_id": info["id"],
            "annotations": [
                {k: a[k] for k in ("bbox", "category_id", "area", "iscrowd")}
                for a in anns
                if not a.get("iscrowd", 0)
            ],
        }
        return self.subset.open_image(info), target
