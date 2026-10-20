from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("config/config.local.json")
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class EmbeddingError(Exception):
    pass


def _require_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"embedding.{field} must be a non-empty string")
    return value.strip()


def _require_template(value: object, field: str, placeholder: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"embedding.{field} must be a non-empty string")
    if placeholder not in value:
        raise ValueError(f"embedding.{field} must contain {placeholder}")
    return value


def _require_int(value: object, field: str, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"embedding.{field} must be an integer >= {minimum}")
    return value


def _require_number(value: object, field: str, minimum: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"embedding.{field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"embedding.{field} must be >= {minimum}")
    return number


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    enabled: bool = False
    url: str = field(default="", repr=False)
    model: str = field(default="", repr=False)
    dimension: int = 0
    query_template: str = field(default="{query}", repr=False)
    document_template: str = field(default="{document}", repr=False)
    batch_size: int = 10
    batch_max_chars: int = 16000
    max_input_chars: int = 12000
    timeout_seconds: float = 60.0
    retries: int = 3
    api_key_env: str = field(default="", repr=False)
    tiebreak_weight: float = 0.0

    @property
    def space(self) -> str:
        material = "\n".join(
            [self.model, str(self.dimension), self.query_template, self.document_template]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> EmbeddingConfig:
        enabled = raw.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("embedding.enabled must be a boolean")
        if not enabled:
            return cls(enabled=False)
        url = _require_str(raw.get("url"), "url")
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError("embedding.url must start with http:// or https://")
        query_template = _require_template(
            raw.get("query_template"), "query_template", "{query}"
        )
        document_template = _require_template(
            raw.get("document_template"), "document_template", "{document}"
        )
        api_key_env = raw.get("api_key_env", "")
        if not isinstance(api_key_env, str):
            raise ValueError("embedding.api_key_env must be a string")
        return cls(
            enabled=True,
            url=url,
            model=_require_str(raw.get("model"), "model"),
            dimension=_require_int(raw.get("dimension"), "dimension", 1),
            query_template=query_template,
            document_template=document_template,
            batch_size=_require_int(raw.get("batch_size", 10), "batch_size", 1),
            batch_max_chars=_require_int(
                raw.get("batch_max_chars", 16000), "batch_max_chars", 1
            ),
            max_input_chars=_require_int(
                raw.get("max_input_chars", 12000), "max_input_chars", 1
            ),
            timeout_seconds=_require_number(
                raw.get("timeout_seconds", 60.0), "timeout_seconds", 0.1
            ),
            retries=_require_int(raw.get("retries", 3), "retries", 1),
            api_key_env=api_key_env.strip(),
            tiebreak_weight=_require_number(
                raw.get("tiebreak_weight", 0.0), "tiebreak_weight", 0.0
            ),
        )


def load_embedding_config(path: Path | None = None) -> EmbeddingConfig:
    target = CONFIG_PATH if path is None else path
    if not target.is_file():
        raise ValueError(f"embedding config {target} is required")
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError(f"cannot read embedding config {target}") from None
    if not isinstance(document, dict):
        raise ValueError(f"embedding config {target} must be a JSON object")
    raw = document.get("embedding")
    if not isinstance(raw, dict):
        raise ValueError(f"embedding config {target} is missing an embedding object")
    return EmbeddingConfig.from_mapping(raw)


class EmbeddingClient:
    def __init__(self, config: EmbeddingConfig) -> None:
        if not config.enabled:
            raise ValueError("embedding is not enabled")
        self.config = config
        self.model = config.model
        self.space = config.space
        self.dimension = config.dimension
        self.max_input_chars = config.max_input_chars
        self.tiebreak_weight = config.tiebreak_weight
        self._api_key: str | None = None
        if config.api_key_env:
            value = os.environ.get(config.api_key_env)
            if not value:
                raise ValueError(f"{config.api_key_env} is not set")
            self._api_key = value

    def close(self) -> None:
        pass

    def embed_query(self, text: str) -> list[float]:
        prepared = self._apply_template(self.config.query_template, "query", text)
        return self._embed([prepared])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        prepared = [
            self._apply_template(self.config.document_template, "document", text)
            for text in texts
        ]
        batches: list[list[str]] = []
        current: list[str] = []
        current_chars = 0
        for text in prepared:
            if current and (
                len(current) >= self.config.batch_size
                or current_chars + len(text) > self.config.batch_max_chars
            ):
                batches.append(current)
                current = []
                current_chars = 0
            current.append(text)
            current_chars += len(text)
        if current:
            batches.append(current)
        vectors: list[list[float]] = []
        for batch in batches:
            vectors.extend(self._embed(batch))
        return vectors

    def _apply_template(self, template: str, key: str, text: str) -> str:
        return template.replace("{" + key + "}", text[: self.config.max_input_chars])

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {
            "model": self.config.model,
            "input": list(texts),
            "encoding_format": "float",
        }
        return self._parse(self._post(payload), len(texts))

    def _post(self, payload: Mapping[str, Any]) -> object:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        for attempt in range(self.config.retries):
            request = urllib.request.Request(
                self.config.url,
                data=data,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.config.timeout_seconds
                ) as response:
                    raw = response.read()
                break
            except urllib.error.HTTPError as exc:
                if exc.code in _RETRYABLE_STATUS and attempt + 1 < self.config.retries:
                    time.sleep(1 if attempt == 0 else 2)
                    continue
                raise EmbeddingError(f"embedding server returned HTTP {exc.code}") from None
            except (TimeoutError, urllib.error.URLError, OSError):
                if attempt + 1 < self.config.retries:
                    time.sleep(1 if attempt == 0 else 2)
                    continue
                raise EmbeddingError("embedding server is unavailable") from None
        else:
            raise EmbeddingError("embedding server is unavailable")
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise EmbeddingError("embedding server returned invalid JSON") from None

    def _parse(self, body: object, expected: int) -> list[list[float]]:
        if not isinstance(body, dict):
            raise EmbeddingError("embedding server returned an invalid response")
        data = body.get("data")
        if not isinstance(data, list) or len(data) != expected:
            raise EmbeddingError("embedding server returned an unexpected number of vectors")
        vectors: dict[int, list[float]] = {}
        for fallback_index, item in enumerate(data):
            if not isinstance(item, dict):
                raise EmbeddingError("embedding server returned an invalid vector")
            index = item.get("index", fallback_index)
            raw_vector = item.get("embedding")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index >= expected
                or index in vectors
                or not isinstance(raw_vector, list)
            ):
                raise EmbeddingError("embedding server returned invalid vector metadata")
            vector: list[float] = []
            for value in raw_vector:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise EmbeddingError("embedding server returned a non-numeric vector")
                number = float(value)
                if not math.isfinite(number):
                    raise EmbeddingError("embedding server returned a non-finite vector")
                vector.append(number)
            if len(vector) != self.config.dimension:
                raise EmbeddingError("embedding server returned an unexpected dimension")
            norm = math.sqrt(sum(value * value for value in vector))
            if not math.isfinite(norm) or norm <= 1e-12:
                raise EmbeddingError("embedding server returned a zero-length vector")
            vectors[index] = [value / norm for value in vector]
        if len(vectors) != expected:
            raise EmbeddingError("embedding server returned incomplete vectors")
        return [vectors[index] for index in range(expected)]


def vector_to_blob(vector: Sequence[float]) -> bytes:
    if not vector:
        raise ValueError("vector must not be empty")
    return struct.pack(f"<{len(vector)}f", *vector)


def blob_to_vector(blob: bytes, dimension: int) -> list[float]:
    if dimension <= 0 or len(blob) != dimension * 4:
        raise ValueError("stored vector has an invalid dimension")
    return list(struct.unpack(f"<{dimension}f", blob))
