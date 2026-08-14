"""Minimal OpenAI-compatible chat-completions client.

Used for *analysis* (failure-mode classification and synthesis), not for the
model under evaluation. Works against any OpenAI-compatible endpoint (vLLM,
SGLang, OpenAI, Anthropic-compatible shims, ...).
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class LLMClient:
    """Thin wrapper over an OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout_seconds

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return the assistant text for a chat request."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]

    def complete_json(self, messages: list[dict[str, str]]) -> Any:
        """Ask for JSON and parse it, tolerating code fences and prose padding."""
        raw = self.complete(messages)
        return _extract_json(raw)


def _extract_json(text: str) -> Any:
    """Parse JSON from a model response, stripping ```json fences and padding."""
    text = text.strip()
    if text.startswith("```"):
        # Drop opening fence line, keep everything until a closing fence.
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise
