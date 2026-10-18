"""Wire format between clients and the model server: base64 images in, JSON detections out."""

from __future__ import annotations

import base64
import io
from dataclasses import asdict, dataclass

from PIL import Image


@dataclass(frozen=True)
class Detection:
    label: str
    label_id: int
    score: float
    box: tuple[float, float, float, float]  # x_min, y_min, x_max, y_max in pixels

    def to_dict(self) -> dict:
        d = asdict(self)
        d["box"] = [round(v, 1) for v in self.box]
        d["score"] = round(self.score, 4)
        return d


def encode_image(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def decode_image(b64: str) -> Image.Image:
    # b64decode ignores the newlines that base64.encodebytes adds, so older clients still work.
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def image_to_bytes(image: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()
