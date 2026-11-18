import unittest

from vlm.metrics import aggregate, score_call, wilson


def call(parsed=None, parse_ok=True, schema_ok=True, latency=1.0):
    c = {"parse_ok": parse_ok, "schema_ok": schema_ok, "parsed": parsed, "latency_s": latency}
    return c


def case(gt, calls, geometry=None):
    for c in calls:
        c["scores"] = score_call(gt, c)
    return {"id": "x", "gt": gt, "calls": calls, "geometry": geometry or {"truncated_by_geometry": False}}


class MetricTests(unittest.TestCase):
    def test_wilson_small_n_is_wide(self):
        lo, hi = wilson(3, 5)
        self.assertGreater(hi - lo, 0.5)
        self.assertEqual(wilson(0, 0), (0.0, 1.0))

    def test_score_call(self):
        gt = {"class": "motorcycle", "occluded": True, "lighting": "bright"}
        s = score_call(gt, call({"class_guess": "motor", "occluded": "yes", "lighting": "bright"}))
        self.assertEqual((s["class_match"], s["occluded_match"], s["lighting_match"]), (True, True, True))
        s = score_call(gt, call({"class_guess": "car", "occluded": "no", "lighting": "dark"}))
        self.assertEqual((s["class_match"], s["occluded_match"], s["lighting_match"]), (False, False, False))
        self.assertEqual(score_call(gt, call({"occluded": "uncertain"}))["occluded_match"], "abstain")
        self.assertIsNone(score_call(gt, call(None, parse_ok=False))["occluded_match"])

    def test_missing_gt_fields_not_scored(self):
        s = score_call({"class": "car"}, call({"class_guess": "car", "occluded": "yes", "lighting": "dark"}))
        self.assertIsNone(s["occluded_match"])
        self.assertIsNone(s["lighting_match"])

    def test_aggregate(self):
        yes = {"class_guess": "car", "occluded": "yes", "lighting": "bright"}
        no = {"class_guess": "car", "occluded": "no", "lighting": "bright"}
        cases = [
            case({"class": "car", "occluded": True}, [call(yes), call(no)]),       # đúng 1, sai 1, không nhất quán
            case({"class": "car", "occluded": False}, [call(no), call(no)]),       # đúng cả hai, nhất quán
            case({"class": "car", "occluded": True, "truncated": True}, [call(None, parse_ok=False, schema_ok=False)],
                 geometry={"truncated_by_geometry": True}),
        ]
        a = aggregate(cases)
        self.assertEqual((a["n_cases"], a["n_calls"]), (3, 5))
        self.assertEqual(a["parse_rate"]["k"], 4)
        self.assertEqual((a["occluded_accuracy_answered"]["k"], a["occluded_accuracy_answered"]["n"]), (3, 4))
        self.assertEqual((a["occluded_recall"]["k"], a["occluded_recall"]["n"]), (1, 2))
        self.assertEqual((a["occluded_false_alarm"]["k"], a["occluded_false_alarm"]["n"]), (0, 2))
        self.assertEqual((a["consistency_across_repeats"]["k"], a["consistency_across_repeats"]["n"]), (1, 2))
        self.assertEqual((a["truncation_rule_vs_gt"]["k"], a["truncation_rule_vs_gt"]["n"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
