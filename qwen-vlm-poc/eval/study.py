"""Chạy thí nghiệm khả thi: Qwen probe từng đối tượng, chấm bằng nhãn GT.

Ví dụ:
  python eval/study.py --dry-run                      # chỉ lưu crop để kiểm tra bằng mắt, không gọi API
  python eval/study.py --repeats 3 --tag pad30        # gọi API thật, lặp 3 lần/ca
  python eval/study.py --replay results/study_xxx.json  # chấm lại từ output đã lưu, không gọi API
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlm.client import QwenClient  # noqa: E402
from vlm.geometry import touches_border  # noqa: E402
from vlm.metrics import aggregate, score_call  # noqa: E402
from vlm.probe import interpret_reply, load_image, prepare_probe_image, probe_object  # noqa: E402


def run_study(
    cases: list[dict],
    prompt: str,
    client,
    base_dir: Path,
    repeats: int = 1,
    pad_ratio: float = 0.3,
    mark: str = "box",
    temperature: float = 0.2,
    max_tokens: int = 1200,
    crops_dir: Path | None = None,
    replay: dict | None = None,
) -> dict:
    """client=None -> dry-run (chỉ tạo crop). replay -> chấm lại từ raw_text đã lưu."""
    case_results = []
    for case in cases:
        image = load_image(base_dir / case["image"])
        bbox = tuple(case["bbox"])
        edges = touches_border(bbox, image.size)

        if crops_dir is not None:
            crops_dir.mkdir(parents=True, exist_ok=True)
            _, _, shown = prepare_probe_image(image, bbox, pad_ratio, mark)
            shown.save(crops_dir / f"{case['id']}.jpg", quality=90)

        calls = []
        if replay is not None:
            old = next((c for c in replay["cases"] if c["id"] == case["id"]), None)
            for i, old_call in enumerate((old or {}).get("calls", []), 1):
                call = {**old_call, **interpret_reply(old_call.get("raw_text", ""))}
                call["repeat"] = i
                calls.append(call)
        elif client is not None:
            for i in range(1, repeats + 1):
                call = probe_object(client, image, bbox, prompt, pad_ratio, mark, temperature, max_tokens)
                call["repeat"] = i
                calls.append(call)

        for call in calls:
            call["scores"] = score_call(case.get("gt", {}), call)

        case_results.append(
            {
                "id": case["id"],
                "image": case["image"],
                "bbox": list(bbox),
                "tags": case.get("tags", []),
                "verified": case.get("verified", False),
                "gt": case.get("gt", {}),
                "geometry": {
                    "image_size": list(image.size),
                    "truncated_edges": edges,
                    "truncated_by_geometry": bool(edges),
                },
                "calls": calls,
            }
        )

    return {
        "config": {
            "pad_ratio": pad_ratio,
            "mark": mark,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "repeats": repeats,
            "model": getattr(client, "model", None),
            "mode": "replay" if replay is not None else ("live" if client is not None else "dry-run"),
        },
        "cases": case_results,
        "summary": aggregate(case_results),
    }


def _fmt(r: dict) -> str:
    if r["n"] == 0:
        return "n/a"
    lo, hi = r["ci95"]
    return f"{r['k']}/{r['n']} = {r['rate']:.0%}  (CI95 {lo:.0%}–{hi:.0%})"


def render_markdown(report: dict) -> str:
    s, cfg = report["summary"], report["config"]
    out = [
        "# Kết quả thí nghiệm probe Qwen2.5-VL",
        "",
        f"- Chế độ: `{cfg['mode']}` · model: `{cfg['model']}` · pad: {cfg['pad_ratio']} · "
        f"đánh dấu: `{cfg['mark']}` · lặp: {cfg['repeats']} · temperature: {cfg['temperature']}",
        f"- Số ca: {s['n_cases']} · số lần gọi: {s['n_calls']}",
        "",
        "> Với số ca nhỏ, khoảng tin cậy (CI95) rộng: kết quả chỉ là **chỉ báo**, chưa đủ kết luận.",
        "",
        "## Tổng hợp",
        "",
        "| Chỉ số | Kết quả |",
        "| --- | --- |",
        f"| JSON parse được | {_fmt(s['parse_rate'])} |",
        f"| Đúng schema và nhất quán | {_fmt(s['schema_rate'])} |",
        f"| Đúng lớp | {_fmt(s['class_accuracy'])} |",
        f"| Occlusion đúng (trên lần có trả lời) | {_fmt(s['occluded_accuracy_answered'])} |",
        f"| Tỷ lệ trả lời 'uncertain' | {_fmt(s['occluded_abstain_rate'])} |",
        f"| Recall occlusion (GT bị che, VLM nói 'yes') | {_fmt(s['occluded_recall'])} |",
        f"| Báo che oan (GT không che, VLM nói 'yes') | {_fmt(s['occluded_false_alarm'])} |",
        f"| Đúng ánh sáng | {_fmt(s['lighting_accuracy'])} |",
        f"| Nhất quán giữa các lần lặp | {_fmt(s['consistency_across_repeats'])} |",
        f"| Luật hình học 'chạm biên' khớp nhãn truncated | {_fmt(s['truncation_rule_vs_gt'])} |",
        f"| Độ trễ mỗi lần gọi (giây) | trung bình {s['latency_s']['mean']} · trung vị {s['latency_s']['median']} |",
        "",
        "## Từng ca",
        "",
        "| Ca | Nhãn đã kiểm | GT occluded | VLM trả lời (các lần) | Khớp | Evidence (lần 1) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for c in report["cases"]:
        answers = ", ".join(str(k["scores"]["occluded_pred"]) for k in c["calls"]) or "–"
        matches = ", ".join(str(k["scores"]["occluded_match"]) for k in c["calls"]) or "–"
        first = (c["calls"][0].get("parsed") or {}).get("evidence", "") if c["calls"] else ""
        out.append(
            f"| {c['id']} | {'có' if c['verified'] else 'CHƯA'} | {c['gt'].get('occluded', '–')} | "
            f"{answers} | {matches} | {first[:90]} |"
        )
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="eval/cases.json")
    ap.add_argument("--prompt", default="prompts/object_probe_v1.txt")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--pad", type=float, default=0.3)
    ap.add_argument("--mark", choices=["box", "none"], default="box")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--dry-run", action="store_true", help="không gọi API, chỉ lưu crop")
    ap.add_argument("--replay", help="file study_*.json cũ: chấm lại, không gọi API")
    args = ap.parse_args()

    cases = json.loads((ROOT / args.cases).read_text(encoding="utf-8"))["cases"]
    prompt = (ROOT / args.prompt).read_text(encoding="utf-8")
    replay = json.loads(Path(args.replay).read_text(encoding="utf-8")) if args.replay else None
    client = None if (args.dry_run or replay) else QwenClient.from_env()
    if client is not None and not client.api_key:
        print("Thiếu FPT_API_KEY: tạo file .env từ .env.example, hoặc dùng --dry-run / --replay.")
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / args.out_dir
    report = run_study(
        cases, prompt, client, ROOT, args.repeats, args.pad, args.mark, args.temperature, args.max_tokens,
        crops_dir=out_dir / "crops" / args.tag, replay=replay,
    )
    report["config"]["prompt_file"] = args.prompt
    report["config"]["cases_file"] = args.cases

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"study_{args.tag}_{stamp}"
    base.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = render_markdown(report)
    base.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\nĐã lưu: {base.with_suffix('.json')}  và  {base.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
