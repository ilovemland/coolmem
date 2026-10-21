from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, replace

from app.embedding import EmbeddingClient, EmbeddingError
from app.errors import InvalidInput, Unavailable
from app.policy import (
    MAX_CONTENT_CHARS,
    MAX_IDENTIFIER_CHARS,
    MAX_MESSAGES_PER_ADD,
    NEARBY_SPAN,
    Policy,
)
from app.storage.repository import MessageRepository, NewMessage
from app.text import ms_to_iso, utc_now
from app.timefmt import format_time

_ROLES = frozenset({"user", "assistant"})


FEEDBACK_PREFIX = "streaming online qa feedback:"


def _content_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_feedback(content: str) -> bool:
    return content.lstrip().lower().startswith(FEEDBACK_PREFIX)


@dataclass(frozen=True, slots=True)
class IncomingMessage:
    role: str
    content: str
    timestamp: int | None = None


@dataclass(frozen=True, slots=True)
class AddResult:
    inserted: int
    skipped: int


class AddService:
    def __init__(
        self,
        repository: MessageRepository,
        policy: Policy,
        embedding: EmbeddingClient,
    ) -> None:
        if embedding is None:
            raise ValueError("embedding is required")
        self._repo = repository
        self._policy = policy
        self._embedding = embedding

    def add(
        self,
        *,
        request_id: str,
        user_id: str,
        session_id: str,
        messages: Sequence[IncomingMessage],
    ) -> AddResult:
        request_id = self._required(
            request_id,
            field="request_id",
            max_chars=MAX_IDENTIFIER_CHARS,
        )
        user_id = self._required(user_id, field="user_id", max_chars=MAX_IDENTIFIER_CHARS)
        session_id = self._required(
            session_id,
            field="session_id",
            max_chars=MAX_IDENTIFIER_CHARS,
        )
        if not messages:
            raise InvalidInput("messages must not be empty", details={"field": "messages"})
        if len(messages) > MAX_MESSAGES_PER_ADD:
            raise InvalidInput(
                f"at most {MAX_MESSAGES_PER_ADD} messages per request",
                details={"field": "messages", "count": len(messages)},
            )

        now = utc_now()
        rows = [
            replace(self._prepare(index, message, now), is_feedback=_is_feedback(message.content))
            for index, message in enumerate(messages)
        ]
        self._attach_context(rows)
        texts = [row.render() for row in rows]
        digests = [_content_digest(text) for text in texts]
        space = self._embedding.space
        dimension = self._embedding.dimension
        cached = self._repo.cached_vectors(space, digests, dimension)
        missing = [index for index, digest in enumerate(digests) if digest not in cached]
        if missing:
            try:
                fresh = self._embedding.embed_documents([texts[index] for index in missing])
            except (EmbeddingError, ValueError) as exc:
                raise Unavailable("embedding service is unavailable") from exc
            if len(fresh) != len(missing):
                raise Unavailable("embedding service returned an invalid result")
            self._repo.store_vectors(
                space,
                [(digests[index], vector) for index, vector in zip(missing, fresh, strict=True)],
            )
            for index, vector in zip(missing, fresh, strict=True):
                cached[digests[index]] = vector
        vectors = [cached[digest] for digest in digests]
        result = self._repo.insert_many(
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            rows=rows,
            vectors=vectors,
            space=self._embedding.space,
            model=self._embedding.model,
        )
        return AddResult(inserted=len(result.ids), skipped=result.skipped)

    def _attach_context(self, rows: list[NewMessage]) -> None:
        window = max(0, int(NEARBY_SPAN))
        if window == 0:
            return
        texts = [row.render() for row in rows]
        for index, row in enumerate(rows):
            lo, hi = max(0, index - window), min(len(rows), index + window + 1)
            neighbours = [texts[j] for j in range(lo, hi) if j != index]
            rows[index] = replace(row, nearby_text="\n".join(neighbours))

    def _prepare(self, position: int, message: IncomingMessage, now: str) -> NewMessage:
        role = self._required(
            message.role, field=f"messages[{position}].role", max_chars=MAX_IDENTIFIER_CHARS
        )
        if role not in _ROLES:
            raise InvalidInput(
                f"messages[{position}].role must be one of {sorted(_ROLES)}",
                details={"field": f"messages[{position}].role"},
            )
        content = message.content
        if not content.strip():
            raise InvalidInput(
                f"messages[{position}].content must not be blank",
                details={"field": f"messages[{position}].content"},
            )
        if len(content) > MAX_CONTENT_CHARS:
            raise InvalidInput(
                f"messages[{position}].content must be at most "
                f"{MAX_CONTENT_CHARS} characters",
                details={"field": f"messages[{position}].content", "length": len(content)},
            )
        if message.timestamp is not None and message.timestamp < 0:
            raise InvalidInput(
                f"messages[{position}].timestamp must be a non-negative Unix millisecond",
                details={"field": f"messages[{position}].timestamp"},
            )
        return NewMessage(
            position=position,
            role=role,
            content=content,
            timestamp=message.timestamp,
            created_at=(ms_to_iso(message.timestamp) if message.timestamp is not None else now),
            time_text=format_time(message.timestamp),
        )

    @staticmethod
    def _required(value: str, *, field: str, max_chars: int) -> str:
        if not value.strip():
            raise InvalidInput(f"{field} must not be empty", details={"field": field})
        if len(value) > max_chars:
            raise InvalidInput(
                f"{field} must be at most {max_chars} characters",
                details={"field": field, "length": len(value)},
            )
        return value
