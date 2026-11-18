"""Kiểm tra nhất quán cho output PHÂN TÍCH CẢ CẢNH (prompt traffic_analysis_v3_x).

Các cảnh báo là GỢI Ý để người xem xem lại, không phải kết luận đúng/sai.
"""
from __future__ import annotations

VALID_POSITIONS = {
    "left foreground", "center foreground", "right foreground",
    "left background", "center background", "right background",
}
VALID_VISIBILITY = {"fully_visible", "partially_occluded", "truncated_by_boundary"}
OCCLUSION_KEYWORDS = ("hidden by", "occluded by", "blocked by")


def validate_vehicles(data: dict) -> list[str]:
    warnings: list[str] = []
    vehicles = data.get("vehicles", [])
    if not isinstance(vehicles, list):
        return ["'vehicles' không phải list."]

    signature_counts: dict[tuple, int] = {}
    for idx, v in enumerate(vehicles):
        vid = v.get("object_id", f"index_{idx}")
        vs = v.get("visibility_status")
        if vs not in VALID_VISIBILITY:
            warnings.append(f"{vid}: visibility_status không hợp lệ '{vs}'.")

        pos = v.get("position", "")
        if pos not in VALID_POSITIONS:
            warnings.append(f"{vid}: position không hợp lệ '{pos}'.")

        edges = v.get("truncated_edges", [])
        if not isinstance(edges, list):
            warnings.append(f"{vid}: truncated_edges không phải list.")
            edges = []
        if "left" in edges and not pos.startswith("left"):
            warnings.append(f"{vid}: truncated_edges có 'left' nhưng position là '{pos}'.")
        if "right" in edges and not pos.startswith("right"):
            warnings.append(f"{vid}: truncated_edges có 'right' nhưng position là '{pos}'.")
        if pos.startswith("center") and ("left" in edges or "right" in edges):
            warnings.append(f"{vid}: position 'center' nhưng truncated_edges là {edges}.")

        evidence = (v.get("evidence", "") or "").lower()
        if any(kw in evidence for kw in OCCLUSION_KEYWORDS) and v.get("occluder") is None:
            warnings.append(f"{vid}: evidence nói bị che nhưng occluder là null.")
        if any(kw in evidence for kw in OCCLUSION_KEYWORDS) and vs == "fully_visible":
            warnings.append(f"{vid}: evidence nói bị che nhưng visibility_status là fully_visible (mâu thuẫn).")

        sig = (
            v.get("class"),
            tuple(v.get("visible_parts", [])),
            tuple(v.get("hidden_parts", [])),
            tuple(v.get("truncated_edges", [])),
        )
        signature_counts[sig] = signature_counts.get(sig, 0) + 1

    for sig, count in signature_counts.items():
        if count > 1:
            warnings.append(
                f"Lặp: {count} xe có cùng class='{sig[0]}', visible/hidden parts và truncated_edges giống hệt "
                f"(nghi copy-paste thay vì phân tích từng xe)."
            )
    return warnings
