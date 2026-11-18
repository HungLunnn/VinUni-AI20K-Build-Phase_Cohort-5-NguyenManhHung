"""Thí nghiệm 'probe một đối tượng': VLM chỉ trả lời về MỘT đối tượng đã biết bbox.

Đây là cách dùng VLM dự kiến cho Detection Doctor: engine xác định ra lỗi (ví dụ
một GT bị bỏ sót), VLM giải thích đối tượng đó. Dễ hơn nhiều so với bắt VLM đếm
và mô tả cả cảnh đông xe.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from .client import QwenClient, VLMError
from .geometry import BBox, crop_with_padding, mark_bbox, to_jpeg_bytes
from .labels import normalize_class
from .parsing import ParseError, extract_json

OCCLUDED_VALUES = {"yes", "no", "uncertain"}
LIGHTING_VALUES = {"bright", "dim", "dark", "unknown"}
CLASS_VALUES = {"car", "motorcycle", "bus", "truck", "bicycle", "other"}
REQUIRED = (
    "class_guess",
    "occluded",
    "occluder",
    "visible_parts",
    "hidden_parts",
    "lighting",
    "evidence",
    "uncertainty",
)
GENERIC_PHRASES = ("clearly visible", "obviously", "clearly seen", "easily visible", "plainly visible")
NULL_LIKE = {None, "", "none", "null", "n/a", "no"}


def _is_null_occluder(value) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in NULL_LIKE


def validate_probe(data: dict) -> list[str]:
    """Kiểm tra schema và các ràng buộc nhất quán. Trả về danh sách lỗi (rỗng = hợp lệ)."""
    errors: list[str] = []
    for key in REQUIRED:
        if key not in data:
            errors.append(f"thiếu trường '{key}'")

    if data.get("occluded") not in OCCLUDED_VALUES:
        errors.append(f"occluded không hợp lệ: {data.get('occluded')!r}")
    if data.get("lighting") not in LIGHTING_VALUES:
        errors.append(f"lighting không hợp lệ: {data.get('lighting')!r}")
    if normalize_class(data.get("class_guess")) not in CLASS_VALUES:
        errors.append(f"class_guess không hợp lệ: {data.get('class_guess')!r}")

    for key in ("visible_parts", "hidden_parts"):
        if key in data and not isinstance(data[key], list):
            errors.append(f"{key} phải là list")

    hidden = data.get("hidden_parts") if isinstance(data.get("hidden_parts"), list) else []
    occluder_null = _is_null_occluder(data.get("occluder"))
    if data.get("occluded") == "no" and (not occluder_null or hidden):
        errors.append("mâu thuẫn: occluded='no' nhưng có occluder hoặc hidden_parts")
    if data.get("occluded") == "yes" and occluder_null:
        errors.append("mâu thuẫn: occluded='yes' nhưng occluder rỗng")

    evidence = str(data.get("evidence", "")).lower()
    if any(p in evidence for p in GENERIC_PHRASES):
        errors.append("evidence dùng cụm chung chung bị cấm")
    if "evidence" in data and not evidence.strip():
        errors.append("evidence rỗng")
    return errors


def interpret_reply(text: str) -> dict:
    """Parse + validate một câu trả lời thô. Dùng chung cho gọi API thật và replay."""
    try:
        parsed = extract_json(text)
    except ParseError as exc:
        return {"parse_ok": False, "schema_ok": False, "parsed": None, "schema_errors": [], "error": str(exc)}
    errors = validate_probe(parsed)
    return {"parse_ok": True, "schema_ok": not errors, "parsed": parsed, "schema_errors": errors, "error": None}


def prepare_probe_image(
    image: Image.Image,
    bbox: BBox,
    pad_ratio: float = 0.3,
    mark: str = "box",
) -> tuple[bytes, BBox, Image.Image]:
    crop, crop_box = crop_with_padding(image, bbox, pad_ratio=pad_ratio)
    shown = mark_bbox(crop, bbox, crop_box) if mark == "box" else crop
    return to_jpeg_bytes(shown), crop_box, shown


def probe_object(
    client: QwenClient,
    image: Image.Image,
    bbox: BBox,
    prompt: str,
    pad_ratio: float = 0.3,
    mark: str = "box",
    temperature: float = 0.2,
    max_tokens: int = 1200,
) -> dict:
    jpeg, crop_box, _ = prepare_probe_image(image, bbox, pad_ratio, mark)
    result = {
        "crop_box": list(crop_box),
        "raw_text": "",
        "latency_s": None,
        "usage": {},
        "attempts": 0,
        "api_error": None,
    }
    try:
        reply = client.chat(prompt, jpeg, temperature=temperature, max_tokens=max_tokens)
    except VLMError as exc:
        result.update(api_error=str(exc), parse_ok=False, schema_ok=False, parsed=None, schema_errors=[], error=str(exc))
        return result
    result.update(raw_text=reply.text, latency_s=reply.latency_s, usage=reply.usage, attempts=reply.attempts)
    result.update(interpret_reply(reply.text))
    return result


def load_image(path: str | Path) -> Image.Image:
    img = Image.open(path)
    img.load()
    return img.convert("RGB")
