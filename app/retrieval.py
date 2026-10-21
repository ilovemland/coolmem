from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.policy import NEARBY_WEIGHT, SEARCH_CANDIDATES, SUBSTRING_DEPTH_RATIO
from app.storage.database import Database
from app.text import cjk_bigrams, cjk_words, fts_query, terms, trigram_query

_COLUMNS = (
    "messages.id, messages.request_id, messages.session_id, messages.role, "
    "messages.content, messages.created_at"
)



@dataclass(frozen=True, slots=True)
class Message:
    id: int
    request_id: str
    session_id: str
    role: str
    content: str
    created_at: str


@dataclass(frozen=True, slots=True)
class ScoredHit:
    message: Message
    raw: float


@dataclass(frozen=True, slots=True)
class SearchHit:
    message: Message
    score: float


class Retriever(Protocol):
    name: str
    tiebreak_weight: float
    depth_ratio: float | None

    def retrieve(self, *, user_id: str, query: str, limit: int) -> list[ScoredHit]: ...


class QueryEmbedder(Protocol):
    space: str
    tiebreak_weight: float

    def embed_query(self, text: str) -> list[float]: ...


class VectorSearcher(Protocol):
    def vector_search(
        self,
        *,
        user_id: str,
        space: str,
        query_vector: Sequence[float],
        limit: int,
    ) -> list[ScoredHit]: ...


def _to_message(row: sqlite3.Row) -> Message:
    return Message(
        id=int(row["id"]),
        request_id=str(row["request_id"]),
        session_id=str(row["session_id"]),
        role=str(row["role"]),
        content=str(row["content"]),
        created_at=str(row["created_at"]),
    )


class LexicalRetriever:
    def __init__(self, database: Database, policy: Policy) -> None:
        self._db = database
        self._policy = policy
        self.name = "lexical"
        self.tiebreak_weight = 0.02
        self.depth_ratio = None

    def retrieve(self, *, user_id: str, query: str, limit: int) -> list[ScoredHit]:
        candidates = max(limit, SEARCH_CANDIDATES)
        relaxed = fts_query(query, operator="OR")
        if not relaxed:
            return []
        rows = self._run_fts(user_id, relaxed, candidates)
        strict = fts_query(query, operator="AND")
        if not strict or strict == relaxed:
            return rows[:limit]
        strict_rows = self._run_fts(user_id, strict, candidates)
        strict_ids = {hit.message.id for hit in strict_rows}
        filled = [*strict_rows, *(hit for hit in rows if hit.message.id not in strict_ids)]
        return filled[:limit]

    def _run_fts(self, user_id: str, match: str, limit: int) -> list[ScoredHit]:
        weight = NEARBY_WEIGHT
        rows = (
            self._db.connection()
            .execute(
                f"""
            SELECT {_COLUMNS}, bm25(messages_fts, 1.0, ?, 0.2) AS rank_score
              FROM messages_fts JOIN messages ON messages.id = messages_fts.rowid
             WHERE messages_fts MATCH ? AND messages.user_id = ?
             ORDER BY bm25(messages_fts, 1.0, ?, 0.2), messages.id
             LIMIT ?
            """,
                (weight, match, user_id, weight, limit),
            )
            .fetchall()
        )
        return [ScoredHit(message=_to_message(row), raw=float(row["rank_score"])) for row in rows]


class SubstringRetriever:
    def __init__(self, database: Database, policy: Policy) -> None:
        self._db = database
        self._policy = policy
        self.name = "substring"
        self.tiebreak_weight = 0.0
        self.depth_ratio = SUBSTRING_DEPTH_RATIO

    def retrieve(self, *, user_id: str, query: str, limit: int) -> list[ScoredHit]:
        match = trigram_query(query)
        if not match:
            return []
        rows = (
            self._db.connection()
            .execute(
                f"""
            SELECT {_COLUMNS}, bm25(messages_trgm) AS rank_score
              FROM messages_trgm JOIN messages ON messages.id = messages_trgm.rowid
             WHERE messages_trgm MATCH ? AND messages.user_id = ?
             ORDER BY bm25(messages_trgm), messages.id
             LIMIT ?
            """,
                (match, user_id, limit),
            )
            .fetchall()
        )
        return [ScoredHit(message=_to_message(row), raw=float(row["rank_score"])) for row in rows]


class DenseRetriever:
    def __init__(self, searcher: VectorSearcher, embedder: QueryEmbedder) -> None:
        self._searcher = searcher
        self._embedder = embedder
        self.name = "dense"
        self.tiebreak_weight = float(embedder.tiebreak_weight)
        self.max_input_chars = getattr(embedder, "max_input_chars", None)
        self.depth_ratio = None

    def retrieve(self, *, user_id: str, query: str, limit: int) -> list[ScoredHit]:
        if limit <= 0:
            return []
        vector = self._embedder.embed_query(query)
        return self._searcher.vector_search(
            user_id=user_id,
            space=self._embedder.space,
            query_vector=vector,
            limit=limit,
        )


_WORD_RE = re.compile(r"\w+", re.UNICODE)
