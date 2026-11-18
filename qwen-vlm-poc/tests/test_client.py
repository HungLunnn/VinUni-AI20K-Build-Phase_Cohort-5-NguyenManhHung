import unittest
from unittest import mock

import requests

from vlm.client import QwenClient, VLMError


class FakeResp:
    def __init__(self, status, body=None, text=""):
        self.status_code, self._body, self.text = status, body or {}, text

    def json(self):
        return self._body


def ok_body(text="{}"):
    return {"choices": [{"message": {"content": text}}], "usage": {"total_tokens": 5}}


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def post(self, *a, **k):
        item = self.responses[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


class ClientTests(unittest.TestCase):
    def make(self, responses, **kw):
        return QwenClient(api_key="k", session=FakeSession(responses), backoff_s=0, **kw)

    def test_missing_key(self):
        with self.assertRaises(VLMError):
            QwenClient(api_key="", session=FakeSession([])).chat("p", b"x")

    def test_success_and_wrapped_data(self):
        c = self.make([FakeResp(200, {"data": ok_body("hello")})])
        self.assertEqual(c.chat("p", b"x").text, "hello")

    def test_retry_then_success(self):
        c = self.make([FakeResp(429), requests.exceptions.Timeout("t"), FakeResp(200, ok_body("done"))])
        reply = c.chat("p", b"x")
        self.assertEqual((reply.text, reply.attempts), ("done", 3))

    def test_client_error_not_retried(self):
        sess = FakeSession([FakeResp(401, text="bad key")])
        c = QwenClient(api_key="k", session=sess, backoff_s=0)
        with self.assertRaises(VLMError):
            c.chat("p", b"x")
        self.assertEqual(sess.calls, 1)

    def test_gives_up_after_max_retries(self):
        c = self.make([FakeResp(503)] * 3, max_retries=3)
        with self.assertRaises(VLMError):
            c.chat("p", b"x")

    def test_no_choices(self):
        c = self.make([FakeResp(200, {"choices": []})])
        with self.assertRaises(VLMError):
            c.chat("p", b"x")


if __name__ == "__main__":
    unittest.main()
