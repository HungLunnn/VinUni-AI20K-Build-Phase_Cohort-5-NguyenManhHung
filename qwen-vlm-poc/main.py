"""Phân tích CẢ CẢNH bằng một prompt (thí nghiệm gốc, giữ lại để so sánh).

  python main.py --image images/crop_test.jpg --prompt prompts/traffic_analysis_v3_2.txt

Kết quả lưu ở results/scene_<ảnh>_<prompt>.json. Với việc nghiên cứu khả thi chính,
dùng eval/study.py (probe từng đối tượng).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm.client import QwenClient, VLMError
from vlm.geometry import to_jpeg_bytes
from vlm.parsing import ParseError, extract_json
from vlm.probe import load_image
from vlm.scene_checks import validate_vehicles


def analyze_scene(client: QwenClient, image_path: Path, prompt_path: Path, max_tokens: int = 4000) -> dict:
    prompt = prompt_path.read_text(encoding="utf-8")
    reply = client.chat(prompt, to_jpeg_bytes(load_image(image_path)), max_tokens=max_tokens)
    parsed = extract_json(reply.text)
    warnings = validate_vehicles(parsed)
    parsed["_validation"] = {
        "warnings": warnings,
        "prompt_file": str(prompt_path),
        "image_file": str(image_path),
        "latency_s": round(reply.latency_s, 2),
    }
    return parsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default="images/crop_test.jpg")
    ap.add_argument("--prompt", default="prompts/traffic_analysis_v3_2.txt")
    ap.add_argument("--out", help="mặc định: results/scene_<ảnh>_<prompt>.json")
    args = ap.parse_args()

    image_path, prompt_path = Path(args.image), Path(args.prompt)
    out = Path(args.out) if args.out else Path("results") / f"scene_{image_path.stem}_{prompt_path.stem}.json"

    try:
        result = analyze_scene(QwenClient.from_env(), image_path, prompt_path)
    except (VLMError, ParseError, FileNotFoundError) as exc:
        print(f"Lỗi: {exc}")
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for w in result["_validation"]["warnings"]:
        print("CẢNH BÁO:", w)
    print(f"Đã lưu {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
