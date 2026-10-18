import base64
import json

from PIL import Image

from detr_lifecycle.codec import decode_image, encode_image, image_to_bytes
from detr_lifecycle.data import CocoSubset, TrainingSet, build_subset


def test_image_roundtrip_accepts_both_base64_styles():
    img = Image.new("RGB", (32, 16), (10, 200, 30))
    raw = image_to_bytes(img, "PNG")
    assert decode_image(encode_image(raw)).size == (32, 16)
    # The 2024 client sent base64.encodebytes output, with newlines every 76 chars.
    assert decode_image(base64.encodebytes(raw).decode()).size == (32, 16)


def _coco_source(tmp_path):
    images = [{"id": i, "file_name": f"{i}.jpg", "coco_url": ""} for i in range(1, 21)]
    annotations = [
        {"id": i, "image_id": i, "category_id": 1, "bbox": [0, 0, 5, 5], "area": 25, "iscrowd": 0}
        for i in range(1, 19)  # images 19 and 20 have no objects
    ]
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"images": images, "annotations": annotations, "categories": []}))
    return path


def test_subset_is_reproducible_and_skips_empty_images(tmp_path):
    source = _coco_source(tmp_path)
    a = build_subset(source, tmp_path / "a", n=5, seed=0)
    b = build_subset(source, tmp_path / "b", n=5, seed=0)
    ids = {im["id"] for im in a.load()["images"]}
    assert ids == {im["id"] for im in b.load()["images"]}
    assert not ids & {19, 20}
    assert a.fingerprint().split("@")[1] == b.fingerprint().split("@")[1]


def test_subset_excludes_holdout_images(tmp_path):
    source = _coco_source(tmp_path)
    holdout = build_subset(source, tmp_path / "holdout", n=8, seed=0)
    batch = build_subset(source, tmp_path / "batch", n=10, seed=1, exclude=[holdout])
    held = {im["id"] for im in holdout.load()["images"]}
    assert not held & {im["id"] for im in batch.load()["images"]}


def test_training_set_drops_crowd_boxes(holdout: CocoSubset):
    data = holdout.load()
    data["annotations"][0]["iscrowd"] = 1
    crowd = holdout.root.parent / "crowd"
    (crowd / "images").mkdir(parents=True)
    for im in data["images"]:
        (crowd / "images" / im["file_name"]).write_bytes(
            (holdout.images_dir / im["file_name"]).read_bytes()
        )
    (crowd / "annotations.json").write_text(json.dumps(data))
    sizes = [len(t["annotations"]) for _, t in TrainingSet(CocoSubset(crowd))]
    assert sorted(sizes) == [0, 1, 1, 1, 1, 1]
