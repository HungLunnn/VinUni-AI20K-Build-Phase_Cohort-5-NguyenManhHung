import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: str | Path) -> dict:
    """Load file JSON (hoặc file txt chứa JSON)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_visibility(status: str | None) -> str:
    """Chuẩn hóa chuỗi visibility để so sánh không phân biệt hoa/thường, khoảng trắng."""
    if status is None:
        return ""
    return str(status).strip().lower()


def evaluate(qwen_output: dict, ground_truth: dict) -> dict:
    """
    So sánh output Qwen với Ground Truth theo cách deterministic.
    Qwen KHÔNG được tự tính TP/FP/FN — việc này do Evaluation Engine thực hiện.

    Trả về báo cáo chi tiết: matched, fn, fp, mismatches.
    """
    qwen_vehicles = qwen_output.get("vehicles", [])
    gt_vehicles = ground_truth.get("vehicles", [])

    # Index theo object_id để tra cứu nhanh
    qwen_by_id = {v.get("object_id"): v for v in qwen_vehicles if v.get("object_id")}
    gt_by_id = {v.get("object_id"): v for v in gt_vehicles if v.get("object_id")}

    qwen_ids = set(qwen_by_id.keys())
    gt_ids = set(gt_by_id.keys())

    matched_ids = qwen_ids & gt_ids   # có ở cả hai
    fn_ids = gt_ids - qwen_ids        # có trong GT, thiếu trong Qwen  -> False Negative
    fp_ids = qwen_ids - gt_ids        # có trong Qwen, không có trong GT -> False Positive

    matched: List[dict] = []
    mismatches: List[dict] = []

    for oid in sorted(matched_ids):
        q = qwen_by_id[oid]
        g = gt_by_id[oid]
        diff: List[dict] = []

        # So sánh class
        if q.get("class") != g.get("class"):
            diff.append({
                "field": "class",
                "qwen": q.get("class"),
                "gt": g.get("class")
            })

        # So sánh position
        if q.get("position") != g.get("position"):
            diff.append({
                "field": "position",
                "qwen": q.get("position"),
                "gt": g.get("position")
            })

        # So sánh visibility_status (Qwen) vs visibility (GT)
        if _normalize_visibility(q.get("visibility_status")) != _normalize_visibility(g.get("visibility")):
            diff.append({
                "field": "visibility_status",
                "qwen": q.get("visibility_status"),
                "gt": g.get("visibility")
            })

        if diff:
            mismatches.append({
                "object_id": oid,
                "differences": diff
            })
        else:
            matched.append({"object_id": oid})

    return {
        "matched_count": len(matched),
        "fn_count": len(fn_ids),
        "fp_count": len(fp_ids),
        "mismatch_count": len(mismatches),
        "matched": matched,
        "fn": [gt_by_id[i] for i in sorted(fn_ids)],
        "fp": [qwen_by_id[i] for i in sorted(fp_ids)],
        "mismatches": mismatches,
        "total_gt": len(gt_vehicles),
        "total_qwen": len(qwen_vehicles),
    }


def print_report(report: dict) -> None:
    """In báo cáo đánh giá ra terminal."""
    print("========== EVALUATION REPORT ==========")
    print(f"Total GT vehicles : {report['total_gt']}")
    print(f"Total Qwen vehicles: {report['total_qwen']}")
    print(f"Matched           : {report['matched_count']}")
    print(f"False Negative    : {report['fn_count']}")
    print(f"False Positive    : {report['fp_count']}")
    print(f"Mismatches        : {report['mismatch_count']}")
    print()

    if report["fn"]:
        print("--- False Negative (trong GT, thiếu trong Qwen) ---")
        for v in report["fn"]:
            print(f"  {v.get('object_id')}: {v.get('class')} at {v.get('position')}")
    if report["fp"]:
        print("--- False Positive (trong Qwen, không có trong GT) ---")
        for v in report["fp"]:
            print(f"  {v.get('object_id')}: {v.get('class')} at {v.get('position')}")
    if report["mismatches"]:
        print("--- Mismatches (sai trường) ---")
        for m in report["mismatches"]:
            print(f"  {m['object_id']}:")
            for d in m["differences"]:
                print(f"    {d['field']}: Qwen={d['qwen']}, GT={d['gt']}")
    print("=======================================")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <qwen_output.json> <ground_truth.json>")
        print("Example: python evaluate.py results/response_v3_2.txt eval/gt_crop_test.json")
        sys.exit(1)

    qwen_path = Path(sys.argv[1])
    gt_path = Path(sys.argv[2])

    if not qwen_path.exists():
        print(f"File not found: {qwen_path}")
        sys.exit(1)
    if not gt_path.exists():
        print(f"File not found: {gt_path}")
        sys.exit(1)

    qwen = load_json(qwen_path)
    gt = load_json(gt_path)

    report = evaluate(qwen, gt)
    print_report(report)