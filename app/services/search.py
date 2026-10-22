from __future__ import annotations

from dataclasses import dataclass

from app.embedding import EmbeddingError
from app.errors import InvalidInput, Unavailable
from app.policy import MAX_QUERY_CHARS, MAX_TOP_K, MAX_IDENTIFIER_CHARS, Policy
from app.retrieval import Message
from app.storage.repository import MessageRepository


@dataclass(frozen=True, slots=True)
class SearchResult:
    message: Message
    score: float


class SearchService:
    def __init__(self, repository: MessageRepository, policy: Policy) -> None:
        self._repo = repository
        self._policy = policy

    def search(self, *, user_id: str, query: str, top_k: int) -> list[SearchResult]:
        user_id = self._identifier(
            user_id, field="user_id", max_chars=MAX_IDENTIFIER_CHARS
        )
        if not query.strip():
            raise InvalidInput("query must not be empty", details={"field": "query"})
        if len(query) > MAX_QUERY_CHARS:
            raise InvalidInput(
                f"query must be at most {MAX_QUERY_CHARS} characters",
                details={"field": "query", "length": len(query)},
            )
        if top_k <= 0:
            raise InvalidInput("top_k must be positive", details={"field": "top_k"})
        limit = min(top_k, MAX_TOP_K)

        try:
            hits = self._repo.search(user_id=user_id, query=query, limit=limit)
        except EmbeddingError as exc:
            raise Unavailable("embedding service is unavailable") from exc
        if not hits:
            return []

        best = hits[0].score
        results = [
            SearchResult(
                message=hit.message,
                score=round(hit.score / best, 6) if best > 0 else 0.0,
            )
            for hit in hits
        ]
        return results

    @staticmethod
    def _identifier(value: str, *, field: str, max_chars: int) -> str:
        if not value.strip():
            raise InvalidInput(f"{field} must not be empty", details={"field": field})
        if len(value) > max_chars:
            raise InvalidInput(
                f"{field} must be at most {max_chars} characters",
                details={"field": field, "length": len(value)},
            )
        return value
