import json
import unittest

from PIL import Image

from tests.fakes import GOOD_NO, GOOD_YES, FakeClient
from vlm.client import VLMError
from vlm.probe import interpret_reply, probe_object, validate_probe


def with_(base: str, **changes) -> dict:
    d = json.loads(base)
    d.update(changes)
    return d


class ValidateTests(unittest.TestCase):
    def test_good(self):
        self.assertEqual(validate_probe(json.loads(GOOD_NO)), [])
        self.assertEqual(validate_probe(json.loads(GOOD_YES)), [])

    def test_contradictions(self):
        self.assertTrue(validate_probe(with_(GOOD_NO, occluder="red bike")))
        self.assertTrue(validate_probe(with_(GOOD_NO, hidden_parts=["rear wheel"])))
        self.assertTrue(validate_probe(with_(GOOD_YES, occluder=None)))

    def test_bad_enums_and_generic_evidence(self):
        self.assertTrue(validate_probe(with_(GOOD_NO, occluded="maybe")))
        self.assertTrue(validate_probe(with_(GOOD_NO, lighting="sunny")))
        self.assertTrue(validate_probe(with_(GOOD_NO, class_guess="spaceship")))
        self.assertTrue(validate_probe(with_(GOOD_NO, evidence="The car is clearly visible.")))

    def test_missing_field(self):
        d = json.loads(GOOD_NO)
        d.pop("evidence")
        self.assertTrue(any("evidence" in e for e in validate_probe(d)))

    def test_class_alias_accepted(self):
        self.assertEqual(validate_probe(with_(GOOD_NO, class_guess="motor")), [])


class InterpretTests(unittest.TestCase):
    def test_fenced_good(self):
        r = interpret_reply("```json\n" + GOOD_YES + "\n```")
        self.assertTrue(r["parse_ok"] and r["schema_ok"])

    def test_parse_fail(self):
        r = interpret_reply("sorry I cannot")
        self.assertFalse(r["parse_ok"])
        self.assertIsNone(r["parsed"])

    def test_parse_ok_schema_bad(self):
        r = interpret_reply('{"occluded": "yes"}')
        self.assertTrue(r["parse_ok"])
        self.assertFalse(r["schema_ok"])
        self.assertTrue(r["schema_errors"])


class ProbeObjectTests(unittest.TestCase):
    def setUp(self):
        self.img = Image.new("RGB", (640, 480), "gray")

    def test_live_path(self):
        out = probe_object(FakeClient([GOOD_YES]), self.img, (100, 100, 300, 300), "prompt")
        self.assertTrue(out["schema_ok"])
        self.assertEqual(out["parsed"]["occluded"], "yes")
        self.assertEqual(len(out["crop_box"]), 4)

    def test_api_error_is_captured_not_raised(self):
        class Boom:
            model = "x"

            def chat(self, *a, **k):
                raise VLMError("HTTP 503")

        out = probe_object(Boom(), self.img, (100, 100, 300, 300), "prompt")
        self.assertFalse(out["parse_ok"])
        self.assertEqual(out["api_error"], "HTTP 503")


if __name__ == "__main__":
    unittest.main()
