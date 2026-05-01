"""MiMo API client — thin wrapper around the Xiaomi MiMo API platform."""

import os
import time
import json
import requests
from typing import Any, Optional
from dataclasses import dataclass, field


BASE_URL = os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
API_KEY = os.getenv("MIMO_API_KEY", "")


@dataclass
class Message:
    """A single message in a conversation."""
    role: str          # "system" | "user" | "assistant" | "tool"
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list] = None

    def to_dict(self) -> dict:
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d


@dataclass
class UsageStats:
    """Token usage for a single API call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __repr__(self) -> str:
        return (f"UsageStats(prompt={self.prompt_tokens}, "
                f"completion={self.completion_tokens}, total={self.total_tokens})")


@dataclass
class MiMoResponse:
    """Parsed response from MiMo API."""
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: UsageStats = field(default_factory=UsageStats)
    model: str = ""
    latency_ms: float = 0.0


class MiMoClient:
    """
    Thin, clean client for the Xiaomi MiMo API.

    Supports:
    - Chat completions (text + tool calling)
    - Streaming
    - Configurable base URL + API key
    - Built-in retry + timeout
    - Token usage tracking
    """

    DEFAULT_MODEL = "MiMo-Text-24B"
    SUPPORTED_MODELS = [
        "MiMo-Text-24B",
        "MiMo-Max",
        "MiMo-Reasoning",
        "MiMo-Vision-72B",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.api_key = api_key or API_KEY
        self.base_url = base_url or BASE_URL
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "MiMo-Orchestrator/0.1.0",
        })

    # ------------------------------------------------------------------
    # Core API methods
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[Message | dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[list[dict]] = None,
        stream: bool = False,
    ) -> MiMoResponse:
        """
        Send a chat completion request to MiMo.

        Args:
            messages:      List of Message objects or dicts
            model:         Override default model
            temperature:   Sampling temperature (0.0–2.0)
            max_tokens:   Max tokens in response
            tools:         OpenAI-style tool definitions
            stream:        Whether to stream the response

        Returns:
            MiMoResponse with content, tool_calls, usage, latency
        """
        if not self.api_key:
            raise ValueError(
                "MiMo API key not set. "
                "Set MIMO_API_KEY env var or pass api_key= to MiMoClient()."
            )

        payload = self._build_payload(
            messages=messages,
            model=model or self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            stream=stream,
        )

        url = f"{self.base_url}/chat/completions"
        last_err: Exception | None = None

        for attempt in range(self.max_retries):
            t0 = time.monotonic()
            try:
                resp = self._session.post(url, json=payload, timeout=self.timeout)
                latency = (time.monotonic() - t0) * 1000

                if resp.status_code == 429:
                    # Rate limited — exponential backoff
                    wait = 2 ** attempt
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()

                if stream:
                    # For streaming, collect all chunks first (simplified)
                    text_parts = []
                    tool_parts = []
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        if line.startswith(b"data: "):
                            chunk = json.loads(line[6:])
                            if chunk["choices"][0]["delta"].get("content"):
                                text_parts.append(chunk["choices"][0]["delta"]["content"])
                    content = "".join(text_parts)
                    return MiMoResponse(content=content, latency_ms=latency)

                return self._parse_response(data, latency)

            except requests.RequestException as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    time.sleep(1.5 ** attempt)
                    continue
                raise

        raise RuntimeError(f"MiMo API failed after {self.max_retries} retries: {last_err}")

    def chat_streaming(
        self,
        messages: list[Message | dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[list[dict]] = None,
    ):
        """
        Generator that yields streaming chunks from MiMo.
        Yields: dict with "content", "tool_call", "done" keys.
        """
        if not self.api_key:
            raise ValueError("MiMo API key not set.")

        payload = self._build_payload(
            messages=messages,
            model=model or self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            stream=True,
        )

        url = f"{self.base_url}/chat/completions"
        with self._session.post(url, json=payload, timeout=self.timeout, stream=True) as r:
            r.raise_for_status()
            buffer = ""
            for line in r.iter_lines():
                if not line or line.startswith(b": ping"):
                    continue
                if line.startswith(b"data: "):
                    chunk_data = json.loads(line[6:])
                    if chunk_data["choices"][0]["finish_reason"] == "stop":
                        yield {"done": True}
                        return
                    delta = chunk_data["choices"][0].get("delta", {})
                    if delta.get("content"):
                        yield {"content": delta["content"]}
                    if delta.get("tool_calls"):
                        for tc in delta["tool_calls"]:
                            yield {"tool_call": tc}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        messages: list[Message | dict],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[list[dict]],
        stream: bool,
    ) -> dict:
        """Build the request payload."""
        msg_list = []
        for m in messages:
            if isinstance(m, Message):
                msg_list.append(m.to_dict())
            elif isinstance(m, dict):
                msg_list.append(m)
            else:
                msg_list.append({"role": "user", "content": str(m)})

        payload: dict[str, Any] = {
            "model": model,
            "messages": msg_list,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
        return payload

    def _parse_response(self, data: dict, latency: float) -> MiMoResponse:
        """Parse a non-streaming MiMo API response."""
        choice = data["choices"][0]
        msg = choice["message"]
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        finish = choice.get("finish_reason", "stop")

        usage_data = data.get("usage", {})
        usage = UsageStats(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return MiMoResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish,
            usage=usage,
            model=data.get("model", self.model),
            latency_ms=latency,
        )

    def count_tokens(self, text: str) -> int:
        """
        Rough token estimate: ~4 chars per token for Chinese+English mixed text.
        For accurate counting, use tiktoken or the MiMo tokenizer.
        """
        return len(text) // 4 + len(text.split())

    def __repr__(self) -> str:
        return f"MiMoClient(model={self.model!r}, base_url={self.base_url!r})"
