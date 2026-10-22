from __future__ import annotations

from dataclasses import dataclass

from app.embedding import EmbeddingError
from app.errors import InvalidInput, Unavailable
from app.policy import MAX_QUERY_CHARS, MAX_TOP_K, MAX_IDENTIFIER_CHARS, Policy
from app.retrieval import Message
from app.storage.repository import MessageRepository


PROFILE_PREFIX = "[User profile] "


@dataclass(frozen=True, slots=True)
class SearchResult:
    message: Message
    score: float


def _profile_result(text: str, user_id: str) -> SearchResult:
    return SearchResult(
        message=Message(
            id=-1,
            request_id="profile",
            session_id=user_id,
            role="system",
            content=PROFILE_PREFIX + text,
            created_at="",
        ),
        score=1.0,
    )


class SearchService:
    def __init__(self, repository: MessageRepository, policy: Policy, profile=None) -> None:
        self._repo = repository
        self._policy = policy
        self._profile = profile

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
        profile_text = self._profile.get(user_id) if self._profile is not None else None
        if profile_text and len(results) > 1:
            results = [_profile_result(profile_text, user_id), *results[: len(results) - 1]]
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
