"""Hình học xác định: việc nào code làm được thì không giao cho VLM.

Quan trọng nhất: 'bị cắt bởi biên ảnh' (truncation) suy ra được từ toạ độ bbox,
không cần nhìn ảnh.
"""
from __future__ import annotations

import io

from PIL import Image, ImageDraw

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 theo pixel ảnh gốc


def clamp_bbox(bbox: BBox, size: tuple[int, int]) -> BBox:
    w, h = size
    x1, y1, x2, y2 = bbox
    return (max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2))


def iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def touches_border(bbox: BBox, size: tuple[int, int], margin_px: float = 4.0) -> list[str]:
    """Các cạnh ảnh mà bbox chạm hoặc vượt (trong biên margin_px)."""
    w, h = size
    x1, y1, x2, y2 = bbox
    edges = []
    if x1 <= margin_px:
        edges.append("left")
    if y1 <= margin_px:
        edges.append("top")
    if x2 >= w - margin_px:
        edges.append("right")
    if y2 >= h - margin_px:
        edges.append("bottom")
    return edges


def crop_with_padding(
    image: Image.Image,
    bbox: BBox,
    pad_ratio: float = 0.3,
    min_side: int = 224,
) -> tuple[Image.Image, BBox]:
    """Cắt quanh bbox, thêm đệm pad_ratio mỗi phía, đảm bảo cạnh tối thiểu min_side.

    Trả về (ảnh cắt, khung cắt theo toạ độ ảnh gốc).
    """
    w, h = image.size
    x1, y1, x2, y2 = clamp_bbox(bbox, image.size)
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_ratio, bh * pad_ratio
    cx1, cy1, cx2, cy2 = x1 - px, y1 - py, x2 + px, y2 + py

    def _ensure(lo: float, hi: float, limit: int) -> tuple[float, float]:
        lo, hi = max(0.0, lo), min(float(limit), hi)
        if hi - lo < min_side:
            mid = (lo + hi) / 2
            lo, hi = mid - min_side / 2, mid + min_side / 2
            if lo < 0:
                lo, hi = 0.0, min(float(limit), float(min_side))
            if hi > limit:
                lo, hi = max(0.0, limit - float(min_side)), float(limit)
        return lo, hi

    cx1, cx2 = _ensure(cx1, cx2, w)
    cy1, cy2 = _ensure(cy1, cy2, h)
    box = (int(cx1), int(cy1), int(round(cx2)), int(round(cy2)))
    return image.crop(box), (float(box[0]), float(box[1]), float(box[2]), float(box[3]))


def mark_bbox(
    crop: Image.Image,
    bbox: BBox,
    crop_box: BBox,
    color: str = "red",
    width: int = 3,
) -> Image.Image:
    """Vẽ khung đỏ quanh đối tượng mục tiêu (visual prompting) lên bản sao của crop."""
    out = crop.copy()
    draw = ImageDraw.Draw(out)
    ox, oy = crop_box[0], crop_box[1]
    rect = (bbox[0] - ox, bbox[1] - oy, bbox[2] - ox, bbox[3] - oy)
    draw.rectangle(rect, outline=color, width=width)
    return out


def to_jpeg_bytes(image: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
