import unittest

from vlm.parsing import ParseError, extract_json


class ParsingTests(unittest.TestCase):
    def test_plain_and_fenced(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_prose_around_json(self):
        self.assertEqual(extract_json('Here you go:\n{"a": {"b": 2}}\nHope it helps'), {"a": {"b": 2}})

    def test_braces_inside_strings(self):
        self.assertEqual(extract_json('x {"a": "has } brace"} y'), {"a": "has } brace"})

    def test_truncated_output_raises(self):
        with self.assertRaises(ParseError):
            extract_json('{"vehicles": [{"object_id": "V1", "class": "car"')

    def test_empty_and_non_object(self):
        for bad in ["", "   ", "[1, 2]", "no json at all"]:
            with self.assertRaises(ParseError):
                extract_json(bad)


if __name__ == "__main__":
    unittest.main()
