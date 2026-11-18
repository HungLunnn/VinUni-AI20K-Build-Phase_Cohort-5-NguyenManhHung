"""Chuẩn hoá nhãn lớp để so sánh giữa dataset và câu trả lời của VLM."""
from __future__ import annotations

CLASS_ALIASES = {
    "motor": "motorcycle",
    "motorbike": "motorcycle",
    "moped": "motorcycle",
    "scooter": "motorcycle",
    "bike": "bicycle",
    "sedan": "car",
    "taxi": "car",
    "suv": "car",
}


def normalize_class(name: str | None) -> str:
    if not name:
        return ""
    n = str(name).strip().lower()
    return CLASS_ALIASES.get(n, n)
