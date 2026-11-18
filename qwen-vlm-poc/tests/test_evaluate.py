import unittest

from eval.evaluate import evaluate


def veh(i, cls, pos, vis="fully_visible", key="visibility_status"):
    return {"object_id": i, "class": cls, "position": pos, key: vis}


class EvaluateTests(unittest.TestCase):
    def test_not_matched_by_arbitrary_id(self):
        """Bản cũ ghép theo object_id: GT V3 (car) bị ghép với Qwen V3 (motorcycle). Bản mới không."""
        gt = {"vehicles": [veh("V1", "motorcycle", "left foreground", "fully_visible", "visibility"),
                           veh("V3", "car", "right foreground", "truncated_by_boundary", "visibility")]}
        q = {"vehicles": [veh("V1", "motorcycle", "left foreground"),
                          veh("V3", "motorcycle", "right background"),
                          veh("V7", "car", "right foreground")]}
        r = evaluate(q, gt)
        pairs = {(m["gt_id"], m["qwen_id"]) for m in r["agreed"] + r["mismatches"]}
        self.assertIn(("V3", "V7"), pairs)          # car <-> car
        self.assertNotIn(("V3", "V3"), pairs)
        self.assertEqual(len(r["fn"]), 0)

    def test_incomplete_gt_hides_fp(self):
        gt = {"complete": False, "vehicles": [veh("V1", "car", "left foreground", "fully_visible", "visibility")]}
        q = {"vehicles": [veh("A", "car", "left foreground"), veh("B", "motorcycle", "center background")]}
        r = evaluate(q, gt)
        self.assertEqual(len(r["fp"]), 0)
        self.assertEqual(len(r["unverified"]), 1)

    def test_complete_gt_counts_fp_and_fn(self):
        gt = {"vehicles": [veh("V1", "bus", "center foreground", "fully_visible", "visibility")]}
        q = {"vehicles": [veh("A", "car", "left foreground")]}
        r = evaluate(q, gt)
        self.assertEqual((len(r["fn"]), len(r["fp"])), (1, 1))

    def test_class_alias(self):
        gt = {"vehicles": [veh("V1", "motor", "left foreground", "fully_visible", "visibility")]}
        q = {"vehicles": [veh("A", "motorcycle", "left foreground")]}
        self.assertEqual(len(evaluate(q, gt)["agreed"]), 1)


if __name__ == "__main__":
    unittest.main()
