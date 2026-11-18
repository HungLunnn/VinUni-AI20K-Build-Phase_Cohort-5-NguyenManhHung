"""Client mỏng gọi Qwen2.5-VL qua FPT Cloud (OpenAI-compatible).

Không chạy gì khi import. Không đọc ảnh/prompt cố định: người gọi truyền vào.
Tách riêng để backend của nhóm có thể import và để test thay bằng client giả.
"""
from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field

import requests

DEFAULT_BASE_URL = "https://mkp-api.fptcloud.com/v1"
DEFAULT_MODEL = "Qwen2.5-VL-7B-Instruct"
RETRY_STATUS = {429, 500, 502, 503, 504}


class VLMError(RuntimeError):
    """Lỗi gọi API (thiếu key, HTTP lỗi không thử lại được, hết lượt thử lại)."""


@dataclass
class VLMReply:
    text: str
    latency_s: float
    usage: dict = field(default_factory=dict)
    attempts: int = 1


class QwenClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        backoff_s: float = 2.0,
        session=None,
    ):
        self.api_key = api_key if api_key is not None else os.getenv("FPT_API_KEY")
        self.base_url = (base_url or os.getenv("FPT_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.getenv("FPT_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.session = session or requests

    @classmethod
    def from_env(cls, **kwargs) -> "QwenClient":
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:  # python-dotenv là tuỳ chọn; biến môi trường vẫn dùng được
            pass
        return cls(**kwargs)

    def chat(
        self,
        prompt: str,
        image_bytes: bytes,
        mime: str = "image/jpeg",
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> VLMReply:
        if not self.api_key:
            raise VLMError("FPT_API_KEY chưa được đặt (xem .env.example)")

        data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error = "chưa gọi"
        for attempt in range(1, self.max_retries + 1):
            started = time.perf_counter()
            try:
                resp = self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                latency = time.perf_counter() - started
                if resp.status_code in RETRY_STATUS:
                    last_error = f"HTTP {resp.status_code}"
                elif resp.status_code >= 400:
                    raise VLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                else:
                    return self._to_reply(resp.json(), latency, attempt)
            if attempt < self.max_retries:
                time.sleep(self.backoff_s * (2 ** (attempt - 1)))

        raise VLMError(f"Hết {self.max_retries} lần thử: {last_error}")

    @staticmethod
    def _to_reply(result: dict, latency: float, attempt: int) -> VLMReply:
        # FPT đôi khi bọc kết quả trong "data"; hỗ trợ cả hai dạng.
        body = result.get("data", result)
        choices = body.get("choices") or []
        if not choices:
            raise VLMError(f"Response không có choices: {str(result)[:300]}")
        text = choices[0]["message"]["content"]
        if isinstance(text, list):  # một số server trả content dạng list các phần
            text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
        return VLMReply(
            text=text or "",
            latency_s=latency,
            usage=body.get("usage", {}) or {},
            attempts=attempt,
        )
