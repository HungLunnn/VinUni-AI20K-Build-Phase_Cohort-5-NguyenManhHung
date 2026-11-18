import json
import tempfile
import unittest
from pathlib import Path

from eval.study import render_markdown, run_study
from tests.fakes import GOOD_NO, GOOD_YES, FakeClient

ROOT = Path(__file__).resolve().parents[1]


def load_cases():
    return json.loads((ROOT / "eval" / "cases.json").read_text(encoding="utf-8"))["cases"]


class StudyOfflineTests(unittest.TestCase):
    def test_cases_file_is_well_formed(self):
        cases = load_cases()
        self.assertGreaterEqual(len(cases), 5)
        ids = [c["id"] for c in cases]
        self.assertEqual(len(ids), len(set(ids)))
        for c in cases:
            self.assertTrue((ROOT / c["image"]).exists(), c["image"])
            x1, y1, x2, y2 = c["bbox"]
            self.assertTrue(x2 > x1 and y2 > y1)

    def test_geometry_truncation_matches_draft_gt(self):
        report = run_study(load_cases(), "p", None, ROOT)
        s = report["summary"]["truncation_rule_vs_gt"]
        self.assertEqual(s["k"], s["n"])  # luật chạm-biên khớp mọi nhãn truncated trong bản nháp

    def test_dry_run_saves_crops_and_makes_no_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_study(load_cases(), "p", None, ROOT, crops_dir=Path(tmp))
            self.assertEqual(report["config"]["mode"], "dry-run")
            self.assertEqual(report["summary"]["n_calls"], 0)
            self.assertEqual(len(list(Path(tmp).glob("*.jpg"))), len(load_cases()))

    def test_live_with_fake_client_and_replay_agree(self):
        cases = load_cases()
        client = FakeClient([GOOD_YES, GOOD_NO, "not json"])
        live = run_study(cases, "p", client, ROOT, repeats=2)
        self.assertEqual(live["summary"]["n_calls"], 2 * len(cases))
        self.assertEqual(client.calls, 2 * len(cases))
        self.assertLess(live["summary"]["parse_rate"]["rate"], 1.0)  # có câu 'not json'

        replayed = run_study(cases, "p", None, ROOT, replay=json.loads(json.dumps(live)))
        self.assertEqual(replayed["config"]["mode"], "replay")
        self.assertEqual(replayed["summary"]["parse_rate"], live["summary"]["parse_rate"])
        self.assertEqual(replayed["summary"]["occluded_accuracy_answered"], live["summary"]["occluded_accuracy_answered"])

    def test_markdown_renders(self):
        report = run_study(load_cases(), "p", FakeClient([GOOD_NO]), ROOT)
        md = render_markdown(report)
        self.assertIn("Tổng hợp", md)
        self.assertIn("c01_taxi", md)
        self.assertIn("CHƯA", md)  # nhãn chưa được kiểm


if __name__ == "__main__":
    unittest.main()
