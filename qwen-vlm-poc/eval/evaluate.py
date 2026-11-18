"""So output phân tích CẢ CẢNH của Qwen với GT mức "danh sách xe" (kiểm tra thô).

Khác bản cũ: KHÔNG ghép theo object_id (ID do Qwen tự đặt nên vô nghĩa). Ghép tham lam
theo thuộc tính: (lớp + vị trí) -> (lớp + phía trái/giữa/phải) -> (lớp). Đây vẫn chỉ là
heuristic thô vì output cảnh không có bbox; so sánh chính xác phải dùng eval/study.py.

Nếu GT có "complete": false thì xe Qwen thấy mà GT không có được báo là "unverified"
chứ KHÔNG tính là false positive.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlm.labels import normalize_class  # noqa: E402


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _side(position: str | None) -> str:
    return (position or "").split(" ")[0]


def _match(gt_vehicles: list[dict], q_vehicles: list[dict]):
    """Trả về (cặp [(gt, q, cách_ghép)], gt_thừa, q_thừa)."""
    passes = [
        ("class+position", lambda g, q: g.get("position") == q.get("position")),
        ("class+side", lambda g, q: _side(g.get("position")) == _side(q.get("position"))),
        ("class", lambda g, q: True),
    ]
    free_g, free_q, pairs = list(gt_vehicles), list(q_vehicles), []
    for name, extra in passes:
        for g in list(free_g):
            for q in free_q:
                if normalize_class(g.get("class")) == normalize_class(q.get("class")) and extra(g, q):
                    pairs.append((g, q, name))
                    free_g.remove(g)
                    free_q.remove(q)
                    break
    return pairs, free_g, free_q


def evaluate(qwen_output: dict, ground_truth: dict) -> dict:
    gt_vehicles = ground_truth.get("vehicles", [])
    q_vehicles = qwen_output.get("vehicles", [])
    gt_complete = ground_truth.get("complete", True)

    pairs, fn, extra_q = _match(gt_vehicles, q_vehicles)

    agreed, mismatches = [], []
    for g, q, how in pairs:
        diff = []
        if g.get("position") != q.get("position"):
            diff.append({"field": "position", "qwen": q.get("position"), "gt": g.get("position")})
        if str(q.get("visibility_status", "")).strip().lower() != str(g.get("visibility", "")).strip().lower():
            diff.append({"field": "visibility", "qwen": q.get("visibility_status"), "gt": g.get("visibility")})
        entry = {"gt_id": g.get("object_id"), "qwen_id": q.get("object_id"), "matched_by": how}
        (mismatches if diff else agreed).append({**entry, "differences": diff} if diff else entry)

    return {
        "gt_complete": gt_complete,
        "total_gt": len(gt_vehicles),
        "total_qwen": len(q_vehicles),
        "paired": len(pairs),
        "agreed": agreed,
        "mismatches": mismatches,
        "fn": fn,                                        # GT có, Qwen không có
        "fp": extra_q if gt_complete else [],            # chỉ tính FP khi GT đầy đủ
        "unverified": [] if gt_complete else extra_q,    # Qwen thấy, GT chưa đầy đủ nên chưa kết luận
    }


def print_report(r: dict) -> None:
    print("========== BÁO CÁO (ghép theo thuộc tính, heuristic thô) ==========")
    print(f"GT đầy đủ?            : {r['gt_complete']}")
    print(f"Số xe GT / Qwen       : {r['total_gt']} / {r['total_qwen']}")
    print(f"Ghép được             : {r['paired']}  (khớp hoàn toàn {len(r['agreed'])}, lệch trường {len(r['mismatches'])})")
    print(f"GT bị Qwen bỏ sót (FN): {len(r['fn'])}")
    print(f"Qwen thừa (FP)        : {len(r['fp'])}")
    print(f"Chưa kiểm chứng       : {len(r['unverified'])}  (GT chưa đầy đủ)")
    for m in r["mismatches"]:
        print(f"  {m['gt_id']} <- {m['qwen_id']} [{m['matched_by']}]")
        for d in m["differences"]:
            print(f"    {d['field']}: Qwen={d['qwen']}, GT={d['gt']}")
    for v in r["fn"]:
        print(f"  FN: {v.get('object_id')} {v.get('class')} {v.get('position')}")
    for v in r["fp"]:
        print(f"  FP: {v.get('object_id')} {v.get('class')} {v.get('position')}")
    if not r["gt_complete"]:
        print("  Lưu ý: GT chưa đầy đủ nên số FP bị ẩn; xem 'unverified'.")
    print("====================================================================")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Cách dùng: python eval/evaluate.py <qwen_output.json> <ground_truth.json>")
        print("Ví dụ:     python eval/evaluate.py results/response_v3_2.txt eval/gt_crop_test.json")
        sys.exit(1)
    for p in sys.argv[1:3]:
        if not Path(p).exists():
            print(f"Không thấy file: {p}")
            sys.exit(1)
    print_report(evaluate(load_json(sys.argv[1]), load_json(sys.argv[2])))
