import json
import sys
from pathlib import Path
from typing import Any, Dict, List


# Lớp xe mà Detection Doctor quan tâm (theo đề án 2D detection)
VEHICLE_CLASSES = {"car", "motor", "bus", "truck", "bicycle"}


def map_weather(bdd_weather: str | None) -> str:
    """Map thời tiết BDD100K sang scene.weather cho GT."""
    if not bdd_weather:
        return "unknown"
    w = bdd_weather.lower()
    if "clear" in w:
        return "clear"
    if "rainy" in w or "overcast" in w or "cloudy" in w:
        return "cloudy"
    if "snowy" in w:
        return "snowy"
    if "foggy" in w:
        return "foggy"
    return bdd_weather


def map_timeofday(bdd_time: str | None) -> str:
    """Map thời điểm ngày BDD100K sang scene.lighting cho GT."""
    if not bdd_time:
        return "unknown"
    t = bdd_time.lower()
    if "day" in t or "daytime" in t:
        return "bright"
    if "night" in t:
        return "dark"
    if "dusk" in t or "dawn" in t:
        return "dim"
    return bdd_time


def map_visibility(occluded: bool, truncated: bool) -> str:
    """
    Map thuộc tính BDD100K sang visibility_status của GT.
    Ưu tiên truncated nếu bị cắt biên, sau đó occlusion.
    """
    if truncated:
        return "truncated_by_boundary"
    if occluded:
        return "partially_occluded"
    return "fully_visible"


def compute_position(x1: float, y1: float, x2: float, y2: float,
                     img_w: int = 1280, img_h: int = 720) -> str:
    """
    Tính vị trí粗略 từ bounding box:
    - Trái/Phải/Giữa theo tâm X
    - Trước/Sau theo tâm Y (y lớn hơn = gần camera hơn = foreground)
    """
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Phân chia ngang
    if cx < img_w * 0.33:
        h_pos = "left"
    elif cx > img_w * 0.67:
        h_pos = "right"
    else:
        h_pos = "center"

    # Phân chia dọc (y lớn hơn = thấp hơn trong ảnh = gần camera)
    if cy > img_h * 0.6:
        v_pos = "foreground"
    elif cy < img_h * 0.4:
        v_pos = "background"
    else:
        v_pos = "middle"

    return f"{h_pos} {v_pos}"


def convert_bdd_to_gt(bdd_labels: list[dict], max_images: int | None = None) -> list[dict]:
    """
    Chuyển đổi danh sách annotation BDD100K sang định dạng GT của evaluate.py.
    """
    gt_list = []
    count = 0

    for item in bdd_labels:
        if max_images is not None and count >= max_images:
            break

        image_name = item.get("name", "")
        attributes = item.get("attributes", {}) or {}
        labels = item.get("labels", []) or []

        gt_vehicles = []
        for label in labels:
            category = label.get("category", "")
            if category not in VEHICLE_CLASSES:
                continue

            box2d = label.get("box2d", {}) or {}
            x1 = box2d.get("x1", 0)
            y1 = box2d.get("y1", 0)
            x2 = box2d.get("x2", 0)
            y2 = box2d.get("y2", 0)

            label_attrs = label.get("attributes", {}) or {}
            occluded = label_attrs.get("occluded", False)
            truncated = label_attrs.get("truncated", False)

            gt_vehicles.append({
                "object_id": f"{image_name}_{category}_{label.get('id', '')}",
                "class": category,
                "position": compute_position(x1, y1, x2, y2),
                "visibility": map_visibility(occluded, truncated),
            })

        if not gt_vehicles:
            continue

        gt_list.append({
            "image": image_name,
            "vehicles": gt_vehicles,
            "scene": {
                "traffic_density": "unknown",
                "lighting": map_timeofday(attributes.get("timeofday")),
                "weather": map_weather(attributes.get("weather")),
            }
        })
        count += 1

    return gt_list


def main():
    if len(sys.argv) < 3:
        print("Usage: python bdd100k_to_gt.py <bdd_labels_json> <output_gt_json> [max_images]")
        print("Example: python bdd100k_to_gt.py bdd100k_labels_images_val.json gt_bdd_subset.json 50")
        sys.exit(1)

    bdd_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    max_images = int(sys.argv[3]) if len(sys.argv) > 3 else None

    if not bdd_path.exists():
        print(f"File not found: {bdd_path}")
        sys.exit(1)

    with open(bdd_path, "r", encoding="utf-8") as f:
        bdd_data = json.load(f)

    gt_data = convert_bdd_to_gt(bdd_data, max_images=max_images)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(gt_data)} images to {out_path}")


if __name__ == "__main__":
    main()