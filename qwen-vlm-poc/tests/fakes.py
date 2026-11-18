from vlm.client import VLMReply

GOOD_NO = (
    '{"class_guess":"car","occluded":"no","occluder":null,"visible_parts":["front wheel"],'
    '"hidden_parts":[],"lighting":"bright","evidence":"White car, full body inside the red box.","uncertainty":""}'
)
GOOD_YES = (
    '{"class_guess":"car","occluded":"yes","occluder":"white taxi","visible_parts":["rear"],'
    '"hidden_parts":["left side"],"lighting":"bright","evidence":"Left side hidden by the white taxi.","uncertainty":""}'
)


class FakeClient:
    """Trả lần lượt các chuỗi trong `texts` (lặp vòng nếu hết)."""

    model = "fake"

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def chat(self, prompt, image_bytes, mime="image/jpeg", temperature=0.2, max_tokens=1200):
        text = self.texts[self.calls % len(self.texts)]
        self.calls += 1
        return VLMReply(text=text, latency_s=0.5, usage={"total_tokens": 10}, attempts=1)
