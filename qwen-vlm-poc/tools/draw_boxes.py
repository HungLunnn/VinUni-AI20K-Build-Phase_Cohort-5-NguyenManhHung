"""Vẽ bbox của cases.json lên ảnh gốc để người xem xác nhận nhãn trước khi tin số liệu.

  python tools/draw_boxes.py [--cases eval/cases.json] [--out results/overlays]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="eval/cases.json")
    ap.add_argument("--out", default="results/overlays")
    args = ap.parse_args()

    cases = json.loads((ROOT / args.cases).read_text(encoding="utf-8"))["cases"]
    by_image: dict[str, list[dict]] = {}
    for c in cases:
        by_image.setdefault(c["image"], []).append(c)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for rel, group in by_image.items():
        img = Image.open(ROOT / rel).convert("RGB")
        d = ImageDraw.Draw(img)
        lw = max(3, img.size[0] // 400)
        for c in group:
            x1, y1, x2, y2 = c["bbox"]
            d.rectangle((x1, y1, x2, y2), outline="red", width=lw)
            d.text((x1 + 4, max(0, y1 - 12)), c["id"][:3], fill="yellow")
        target = out_dir / f"{Path(rel).stem}_overlay.jpg"
        img.save(target, quality=88)
        print("Đã lưu", target)


if __name__ == "__main__":
    main()
