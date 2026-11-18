"""Chuyển nhãn BDD100K thành cases.json cho eval/study.py (lấy mẫu cân bằng che/không che).

  python tools/bdd_to_cases.py --labels bdd100k_labels_images_val.json \
      --images-dir data/bdd100k/val --n 60 --out eval/cases_bdd.json

Mỗi ca là MỘT đối tượng (bbox) với nhãn thật của dataset: class, occluded, truncated, lighting.
Nhãn từ dataset nên đặt verified=true (người đã gán nhãn), khác với bản nháp ước lượng bằng mắt.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CLASSES = {"car", "motor", "bus", "truck", "bike"}
LIGHTING = {"daytime": "bright", "night": "dark", "dawn/dusk": "dim"}


def build_cases(
    bdd: list[dict],
    images_dir: str,
    n: int,
    seed: int = 0,
    min_side: int = 40,
    classes: set[str] = CLASSES,
) -> list[dict]:
    pool_true, pool_false = [], []
    for item in bdd:
        attrs = item.get("attributes") or {}
        for label in item.get("labels") or []:
            box = label.get("box2d")
            if label.get("category") not in classes or not box:
                continue
            w, h = box["x2"] - box["x1"], box["y2"] - box["y1"]
            if min(w, h) < min_side:
                continue
            la = label.get("attributes") or {}
            if "occluded" not in la:
                continue
            case = {
                "id": f"{Path(item['name']).stem}_{label.get('id', 0)}",
                "image": f"{images_dir.rstrip('/')}/{item['name']}",
                "bbox": [box["x1"], box["y1"], box["x2"], box["y2"]],
                "gt": {
                    "class": label["category"],
                    "occluded": bool(la["occluded"]),
                    "truncated": bool(la.get("truncated", False)),
                    "lighting": LIGHTING.get(attrs.get("timeofday"), "unknown"),
                },
                "tags": [label["category"], f"weather:{attrs.get('weather', 'unknown')}"],
                "verified": True,
            }
            (pool_true if case["gt"]["occluded"] else pool_false).append(case)

    rng = random.Random(seed)
    rng.shuffle(pool_true)
    rng.shuffle(pool_false)
    half = n // 2
    chosen = pool_true[:half] + pool_false[: n - min(half, len(pool_true))]
    return chosen[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--images-dir", required=True, help="đường dẫn ảnh tương đối so với thư mục dự án")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-side", type=int, default=40)
    ap.add_argument("--out", default="eval/cases_bdd.json")
    args = ap.parse_args()

    bdd = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    cases = build_cases(bdd, args.images_dir, args.n, args.seed, args.min_side)
    Path(args.out).write_text(
        json.dumps({"note": f"Lấy mẫu từ {args.labels}, seed={args.seed}", "cases": cases}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    occ = sum(c["gt"]["occluded"] for c in cases)
    print(f"Đã ghi {len(cases)} ca ({occ} bị che, {len(cases) - occ} không che) -> {args.out}")


if __name__ == "__main__":
    main()
