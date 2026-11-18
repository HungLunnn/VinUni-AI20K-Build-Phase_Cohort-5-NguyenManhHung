"""Tách JSON khỏi câu trả lời của VLM (có thể kèm ```json, lời dẫn, hoặc bị cắt)."""
from __future__ import annotations

import json


class ParseError(ValueError):
    pass


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def _first_balanced_object(text: str) -> str | None:
    """Tìm object {...} đầu tiên có ngoặc cân bằng, bỏ qua ngoặc nằm trong chuỗi."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None  # không cân bằng: thường là output bị cắt do max_tokens


def extract_json(text: str) -> dict:
    """Trả về dict. Ném ParseError nếu không có JSON object hợp lệ."""
    if not text or not text.strip():
        raise ParseError("câu trả lời rỗng")
    cleaned = _strip_fences(text)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        candidate = _first_balanced_object(cleaned)
        if candidate is None:
            raise ParseError("không tìm thấy JSON object hoàn chỉnh (có thể bị cắt do max_tokens)")
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ParseError(f"JSON không hợp lệ: {exc}") from exc
    if not isinstance(value, dict):
        raise ParseError(f"cần JSON object, nhận {type(value).__name__}")
    return value
