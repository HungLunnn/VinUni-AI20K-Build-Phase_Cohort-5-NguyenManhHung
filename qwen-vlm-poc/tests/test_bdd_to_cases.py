import json
import unittest
from pathlib import Path

from tools.bdd_to_cases import build_cases

ROOT = Path(__file__).resolve().parents[1]


class BddToCasesTests(unittest.TestCase):
    def test_sample_labels_convert(self):
        bdd = json.loads((ROOT / "eval" / "sample_bdd_labels.json").read_text(encoding="utf-8"))
        cases = build_cases(bdd, "data/val", n=10, min_side=40)
        self.assertGreaterEqual(len(cases), 3)
        c = cases[0]
        self.assertTrue(c["image"].startswith("data/val/"))
        self.assertEqual(len(c["bbox"]), 4)
        self.assertIn(c["gt"]["lighting"], {"bright", "dim", "dark", "unknown"})
        self.assertTrue(all(x["verified"] for x in cases))
        self.assertTrue(any(x["gt"]["occluded"] for x in cases))

    def test_balanced_and_seeded(self):
        labels = [{"name": f"{i}.jpg", "attributes": {"timeofday": "night"}, "labels": [
            {"id": 1, "category": "car", "box2d": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
             "attributes": {"occluded": i % 2 == 0, "truncated": False}}]} for i in range(40)]
        a = build_cases(labels, "d", n=10, seed=1)
        b = build_cases(labels, "d", n=10, seed=1)
        self.assertEqual([x["id"] for x in a], [x["id"] for x in b])
        self.assertEqual(sum(x["gt"]["occluded"] for x in a), 5)
        self.assertEqual(a[0]["gt"]["lighting"], "dark")

    def test_small_boxes_filtered(self):
        labels = [{"name": "a.jpg", "labels": [{"id": 1, "category": "car",
                   "box2d": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "attributes": {"occluded": True}}]}]
        self.assertEqual(build_cases(labels, "d", n=5, min_side=40), [])


if __name__ == "__main__":
    unittest.main()
