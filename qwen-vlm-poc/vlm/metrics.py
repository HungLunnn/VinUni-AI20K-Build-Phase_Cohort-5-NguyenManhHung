"""Chấm điểm xác định cho thí nghiệm probe: so câu trả lời VLM với nhãn GT."""
from __future__ import annotations

import math
import statistics

from .labels import normalize_class


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Khoảng tin cậy Wilson 95% cho tỷ lệ k/n. n nhỏ thì khoảng sẽ rất rộng, đúng ý nghĩa."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci95": [round(lo, 3), round(hi, 3)]}


def score_call(gt: dict, call: dict) -> dict:
    """Điểm cho một lần gọi. Trường nào GT không có thì None (không tính)."""
    scores = {"occluded_pred": None, "occluded_match": None, "class_match": None, "lighting_match": None}
    parsed = call.get("parsed")
    if not (call.get("parse_ok") and isinstance(parsed, dict)):
        return scores

    pred_occ = parsed.get("occluded")
    scores["occluded_pred"] = pred_occ
    if "occluded" in gt and pred_occ in {"yes", "no", "uncertain"}:
        if pred_occ == "uncertain":
            scores["occluded_match"] = "abstain"
        else:
            scores["occluded_match"] = (pred_occ == "yes") == bool(gt["occluded"])
    if gt.get("class"):
        scores["class_match"] = normalize_class(parsed.get("class_guess")) == normalize_class(gt["class"])
    if gt.get("lighting"):
        scores["lighting_match"] = parsed.get("lighting") == gt["lighting"]
    return scores


def aggregate(case_results: list[dict]) -> dict:
    calls = [(c, call) for c in case_results for call in c["calls"]]
    n_calls = len(calls)
    parse_ok = sum(1 for _, k in calls if k.get("parse_ok"))
    schema_ok = sum(1 for _, k in calls if k.get("schema_ok"))

    def _count(field: str) -> tuple[int, int]:
        vals = [k["scores"][field] for _, k in calls if k["scores"].get(field) in (True, False)]
        return sum(1 for v in vals if v is True), len(vals)

    occ_vals = [(c, k["scores"]) for c, k in calls if k["scores"].get("occluded_match") is not None]
    answered = [s for _, s in occ_vals if s["occluded_match"] in (True, False)]
    abstain = [s for _, s in occ_vals if s["occluded_match"] == "abstain"]

    # recall / false alarm cho 'occluded' trên các lần trả lời (bỏ abstain)
    pos = [(c, k) for c, k in calls if c["gt"].get("occluded") is True and k["scores"]["occluded_match"] in (True, False)]
    neg = [(c, k) for c, k in calls if c["gt"].get("occluded") is False and k["scores"]["occluded_match"] in (True, False)]
    recall = _rate(sum(1 for _, k in pos if k["scores"]["occluded_match"] is True), len(pos))
    false_alarm = _rate(sum(1 for _, k in neg if k["scores"]["occluded_match"] is False), len(neg))

    # nhất quán giữa các lần lặp (cùng một ca)
    consistent = comparable = 0
    for c in case_results:
        answers = [k["scores"]["occluded_pred"] for k in c["calls"] if k["scores"]["occluded_pred"]]
        if len(answers) >= 2:
            comparable += 1
            consistent += len(set(answers)) == 1

    latencies = [k["latency_s"] for _, k in calls if k.get("latency_s") is not None]

    # luật hình học cho truncation so với nhãn GT (nếu có)
    trunc_k = trunc_n = 0
    for c in case_results:
        if "truncated" in c["gt"]:
            trunc_n += 1
            trunc_k += bool(c["geometry"]["truncated_by_geometry"]) == bool(c["gt"]["truncated"])

    return {
        "n_cases": len(case_results),
        "n_calls": n_calls,
        "parse_rate": _rate(parse_ok, n_calls),
        "schema_rate": _rate(schema_ok, n_calls),
        "class_accuracy": _rate(*_count("class_match")),
        "occluded_accuracy_answered": _rate(sum(1 for s in answered if s["occluded_match"] is True), len(answered)),
        "occluded_abstain_rate": _rate(len(abstain), len(occ_vals)),
        "occluded_recall": recall,
        "occluded_false_alarm": false_alarm,
        "lighting_accuracy": _rate(*_count("lighting_match")),
        "consistency_across_repeats": _rate(consistent, comparable),
        "latency_s": {
            "mean": round(statistics.mean(latencies), 2) if latencies else None,
            "median": round(statistics.median(latencies), 2) if latencies else None,
        },
        "truncation_rule_vs_gt": _rate(trunc_k, trunc_n),
    }
