from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from app.embedding import EmbeddingClient, EmbeddingError
from app.errors import InvalidInput, Unavailable
from app.policy import Policy
from app.storage.repository import MessageRepository, NewMessage
from app.text import ms_to_iso, utc_now
from app.timefmt import format_time

_ROLES = frozenset({"user", "assistant"})


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
            max_chars=self._policy.max_request_id_chars,
        )
        user_id = self._required(user_id, field="user_id", max_chars=self._policy.max_user_id_chars)
        session_id = self._required(
            session_id,
            field="session_id",
            max_chars=self._policy.max_session_id_chars,
        )
        if not messages:
            raise InvalidInput("messages must not be empty", details={"field": "messages"})
        if len(messages) > self._policy.max_messages_per_add:
            raise InvalidInput(
                f"at most {self._policy.max_messages_per_add} messages per request",
                details={"field": "messages", "count": len(messages)},
            )

        now = utc_now()
        rows = [self._prepare(index, message, now) for index, message in enumerate(messages)]
        self._attach_neighbours(rows)
        try:
            vectors = self._embedding.embed_documents([row.render() for row in rows])
        except (EmbeddingError, ValueError) as exc:
            raise Unavailable("embedding service is unavailable") from exc
        if len(vectors) != len(rows):
            raise Unavailable("embedding service returned an invalid result")
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

    def _attach_neighbours(self, rows: list[NewMessage]) -> None:
        window = max(0, int(self._policy.neighbor_window))
        if window == 0:
            return
        texts = [row.render() for row in rows]
        for index, row in enumerate(rows):
            lo, hi = max(0, index - window), min(len(rows), index + window + 1)
            neighbours = [texts[j] for j in range(lo, hi) if j != index]
            rows[index] = replace(row, neighbor_text="\n".join(neighbours))

    def _prepare(self, position: int, message: IncomingMessage, now: str) -> NewMessage:
        role = self._required(
            message.role, field=f"messages[{position}].role", max_chars=self._policy.max_role_chars
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
        if len(content) > self._policy.max_content_chars:
            raise InvalidInput(
                f"messages[{position}].content must be at most "
                f"{self._policy.max_content_chars} characters",
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
