from PIL import Image
from transformers import DetrImageProcessor

from detr_lifecycle.train import make_collate


def _target(image_id):
    box = {"bbox": [10, 10, 20, 10], "category_id": 1, "area": 200, "iscrowd": 0}
    return {"image_id": image_id, "annotations": [box]}


def test_batch_padding_keeps_boxes_relative_to_the_real_image():
    # Regression: padding through the processor re-normalised boxes to the padded canvas,
    # which taught the pretrained model the wrong coordinates and cost 15 mAP points.
    processor = DetrImageProcessor(size={"shortest_edge": 64, "longest_edge": 128})
    collate = make_collate(processor, "cpu")
    wide, tall = Image.new("RGB", (96, 64)), Image.new("RGB", (64, 96))
    alone = collate([(wide, _target(1))])
    batched = collate([(wide, _target(1)), (tall, _target(2))])
    assert batched["labels"][0]["boxes"].tolist() == alone["labels"][0]["boxes"].tolist()
    assert batched["pixel_values"].shape[0] == 2
    # The mask marks exactly the real pixels of each image.
    assert batched["pixel_mask"][0].sum() == alone["pixel_mask"][0].sum()
