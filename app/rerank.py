from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("config/config.local.json")
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class RerankError(Exception):
    pass


def _require_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"rerank.{field_name} must be a non-empty string")
    return value.strip()


def _require_int(value: object, field_name: str, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"rerank.{field_name} must be an integer >= {minimum}")
    return value


def _require_number(value: object, field_name: str, minimum: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"rerank.{field_name} must be a number")
    number = float(value)
    if number < minimum:
        raise ValueError(f"rerank.{field_name} must be >= {minimum}")
    return number


@dataclass(frozen=True, slots=True)
class RerankConfig:
    enabled: bool = False
    url: str = field(default="", repr=False)
    model: str = field(default="", repr=False)
    timeout_seconds: float = 30.0
    retries: int = 2
    batch_size: int = 64
    api_key_env: str = field(default="", repr=False)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RerankConfig:
        enabled = raw.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("rerank.enabled must be a boolean")
        if not enabled:
            return cls(enabled=False)
        url = _require_str(raw.get("url"), "url")
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError("rerank.url must start with http:// or https://")
        api_key_env = raw.get("api_key_env", "")
        if not isinstance(api_key_env, str):
            raise ValueError("rerank.api_key_env must be a string")
        return cls(
            enabled=True,
            url=url,
            model=_require_str(raw.get("model"), "model"),
            timeout_seconds=_require_number(
                raw.get("timeout_seconds", 30.0), "timeout_seconds", 0.1
            ),
            retries=_require_int(raw.get("retries", 2), "retries", 1),
            batch_size=_require_int(raw.get("batch_size", 64), "batch_size", 1),
            api_key_env=api_key_env.strip(),
        )


def load_rerank_config(path: Path | None = None) -> RerankConfig:
    target = CONFIG_PATH if path is None else path
    if not target.is_file():
        return RerankConfig(enabled=False)
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError(f"cannot read rerank config {target}") from None
    if not isinstance(document, dict):
        raise ValueError(f"rerank config {target} must be a JSON object")
    raw = document.get("rerank")
    if raw is None:
        return RerankConfig(enabled=False)
    if not isinstance(raw, dict):
        raise ValueError(f"rerank config {target} is missing a rerank object")
    return RerankConfig.from_mapping(raw)


class RerankClient:
    def __init__(self, config: RerankConfig) -> None:
        if not config.enabled:
            raise ValueError("rerank is not enabled")
        self.config = config
        self.model = config.model
        self._api_key: str | None = None
        if config.api_key_env:
            value = os.environ.get(config.api_key_env)
            if not value:
                raise ValueError(f"{config.api_key_env} is not set")
            self._api_key = value

    def close(self) -> None:
        pass

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        scores: list[float] = [0.0] * len(documents)
        batch = max(1, self.config.batch_size)
        for start in range(0, len(documents), batch):
            chunk = documents[start : start + batch]
            values = self._request(query, chunk)
            if len(values) != len(chunk):
                raise RerankError("rerank service returned an invalid result")
            for offset, value in enumerate(values):
                scores[start + offset] = value
        return scores

    def _request(self, query: str, documents: Sequence[str]) -> list[float]:
        payload = json.dumps(
            {
                "model": self.model,
                "query": query,
                "documents": list(documents),
            }
        ).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.config.retries):
            try:
                request = urllib.request.Request(
                    self.config.url,
                    data=payload,
                    headers=self._headers(),
                    method="POST",
                )
                with urllib.request.urlopen(
                    request, timeout=self.config.timeout_seconds
                ) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return self._parse(body, len(documents))
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in _RETRYABLE_STATUS:
                    break
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                last_error = exc
            if attempt + 1 < self.config.retries:
                time.sleep(0.2 * (attempt + 1))
        raise RerankError(f"rerank request failed: {last_error}")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    @staticmethod
    def _parse(body: Any, expected: int) -> list[float]:
        if not isinstance(body, dict):
            raise ValueError("rerank response must be an object")
        rows = body.get("results", body.get("data"))
        if not isinstance(rows, list) or len(rows) != expected:
            raise ValueError("rerank response has an unexpected results length")
        scores: list[float] = [0.0] * expected
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("rerank result must be an object")
            index = row.get("index")
            score = row.get("relevance_score", row.get("score"))
            if not isinstance(index, int) or not isinstance(score, (int, float)):
                raise ValueError("rerank result is missing index or score")
            if not 0 <= index < expected:
                raise ValueError("rerank result index is out of range")
            scores[index] = float(score)
        return scores
