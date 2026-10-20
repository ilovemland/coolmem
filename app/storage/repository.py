from __future__ import annotations

import heapq
import sqlite3
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace

from app.embedding import blob_to_vector, vector_to_blob
from app.errors import PayloadConflict
from app.gating import DENSE, HYBRID, LITERAL, core_query, profile_query
from app.normalize import expanded_text
from app.query import clip_query
from app.policy import Policy
from app.retrieval import (
    DenseRetriever,
    LexicalRetriever,
    Message,
    QueryEmbedder,
    RetrievalResult,
    Retriever,
    ScoredHit,
    SearchHit,
    SubstringRetriever,
    fuse,
)
from app.storage.database import Database
from app.text import utc_now


@dataclass(frozen=True, slots=True)
class NewMessage:
    position: int
    role: str
    content: str
    timestamp: int | None
    created_at: str
    neighbor_text: str = ""
    time_text: str = ""

    def render(self) -> str:
        return f"{self.time_text} {self.content}".strip()


@dataclass(frozen=True, slots=True)
class InsertResult:
    ids: list[int]
    skipped: int


class MessageRepository:
    def __init__(
        self,
        database: Database,
        policy: Policy,
        retrievers: Sequence[Retriever] | None = None,
        embedding: QueryEmbedder | None = None,
    ) -> None:
        self._db = database
        self._policy = policy
        if retrievers is None:
            if embedding is None:
                raise ValueError("embedding is required")
            self._retrievers = [
                LexicalRetriever(database, policy),
                SubstringRetriever(database, policy),
                DenseRetriever(self, embedding),
            ]
        else:
            self._retrievers = list(retrievers)

    def insert_many(
        self,
        *,
        request_id: str,
        user_id: str,
        session_id: str,
        rows: Sequence[NewMessage],
        vectors: Sequence[Sequence[float]] | None = None,
        space: str = "",
        model: str = "",
    ) -> InsertResult:
        if vectors is not None:
            if len(vectors) != len(rows):
                raise ValueError("vectors must match rows")
            if not space or not model:
                raise ValueError("space and model are required with vectors")
            if rows:
                dimension = len(vectors[0])
                if dimension <= 0 or any(len(vector) != dimension for vector in vectors):
                    raise ValueError("vectors must have a consistent non-zero dimension")
        now = utc_now()
        with self._db.transaction() as conn:
            existing = conn.execute(
                """
                SELECT position, role, content, source_content, timestamp, session_id
                  FROM messages
                 WHERE user_id = ? AND request_id = ?
                 ORDER BY position
                """,
                (user_id, request_id),
            ).fetchall()
            if existing:
                mismatch = self._payload_mismatch(existing, session_id, rows)
                if mismatch is None:
                    return InsertResult(ids=[], skipped=len(rows))
                raise PayloadConflict(
                    f"payload conflict on request_id={request_id!r} user_id={user_id!r}: "
                    f"mismatch={mismatch} existing_messages={len(existing)} "
                    f"incoming_messages={len(rows)}",
                    details={
                        "request_id": request_id,
                        "user_id": user_id,
                        "mismatch": mismatch,
                        "existing_messages": len(existing),
                        "incoming_messages": len(rows),
                    },
                )

            ids: list[int] = []
            skipped = 0
            for index, row in enumerate(rows):
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO messages
                        (request_id, position, user_id, session_id, role,
                         content, source_content, search_text, neighbor_text, timestamp, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request_id,
                        row.position,
                        user_id,
                        session_id,
                        row.role,
                        row.render(),
                        row.content,
                        expanded_text(row.render()),
                        row.neighbor_text,
                        row.timestamp,
                        row.created_at,
                    ),
                )
                if cursor.rowcount == 1:
                    message_id = int(cursor.lastrowid)
                    ids.append(message_id)
                    if vectors is not None:
                        vector = vectors[index]
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO message_vectors
                                (message_id, space, model, dimension, vector, created_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                message_id,
                                space,
                                model,
                                len(vector),
                                vector_to_blob(vector),
                                now,
                            ),
                        )
                else:
                    skipped += 1
            return InsertResult(ids=ids, skipped=skipped)

    @staticmethod
    def _payload_mismatch(
        existing: Sequence[sqlite3.Row],
        session_id: str,
        rows: Sequence[NewMessage],
    ) -> str | None:
        if len(existing) != len(rows):
            return "message_count"
        if str(existing[0]["session_id"]) != session_id:
            return "session_id"
        for stored, incoming in zip(existing, rows, strict=True):
            position = int(stored["position"])
            if position != incoming.position:
                return f"position@{position}"
            if str(stored["role"]) != incoming.role:
                return f"role@{position}"
            source = stored["source_content"]
            if source is None:
                source = stored["content"]
            if str(source) != incoming.content:
                return f"content@{position}"
            if stored["timestamp"] != incoming.timestamp:
                return f"timestamp@{position}"
        return None

    def insert_vectors(
        self,
        *,
        ids: Sequence[int],
        vectors: Sequence[Sequence[float]],
        space: str,
        model: str,
    ) -> None:
        if not ids or len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same non-zero length")
        dimension = len(vectors[0])
        if dimension <= 0 or any(len(vector) != dimension for vector in vectors):
            raise ValueError("vectors must have a consistent non-zero dimension")
        now = utc_now()
        with self._db.transaction() as conn:
            for message_id, vector in zip(ids, vectors, strict=True):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO message_vectors
                        (message_id, space, model, dimension, vector, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (message_id, space, model, dimension, vector_to_blob(vector), now),
                )

    def vector_search(
        self,
        *,
        user_id: str,
        space: str,
        query_vector: Sequence[float],
        limit: int,
    ) -> list[ScoredHit]:
        if limit <= 0 or not query_vector:
            return []
        dimension = len(query_vector)
        rows = self._db.connection().execute(
            """
            SELECT messages.id, messages.request_id, messages.session_id, messages.role,
                   messages.content, messages.created_at,
                   message_vectors.vector, message_vectors.dimension
              FROM message_vectors
              JOIN messages ON messages.id = message_vectors.message_id
             WHERE messages.user_id = ? AND message_vectors.space = ?
            """,
            (user_id, space),
        )
        heap: list[tuple[float, int, Message]] = []
        for row in rows:
            if int(row["dimension"]) != dimension:
                continue
            blob = row["vector"]
            if not isinstance(blob, (bytes, bytearray)):
                continue
            try:
                vector = blob_to_vector(bytes(blob), dimension)
            except ValueError:
                continue
            score = sum(a * b for a, b in zip(query_vector, vector, strict=True))
            message = Message(
                id=int(row["id"]),
                request_id=str(row["request_id"]),
                session_id=str(row["session_id"]),
                role=str(row["role"]),
                content=str(row["content"]),
                created_at=str(row["created_at"]),
            )
            entry = (score, message.id, message)
            if len(heap) < limit:
                heapq.heappush(heap, entry)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, entry)
        ordered = sorted(heap, key=lambda item: (-item[0], item[1]))
        return [ScoredHit(message=item[2], raw=item[0]) for item in ordered]

    def count(self) -> int:
        row = self._db.connection().execute("SELECT COUNT(*) AS n FROM messages").fetchone()
        return int(row["n"])

    def search(self, *, user_id: str, query: str, limit: int) -> list[SearchHit]:
        query = clip_query(query, self._query_limit())
        hits = self._search_once(user_id=user_id, query=query, limit=limit)
        if not self._policy.instruction_decomposition:
            return self._dilate(user_id, hits)
        core = core_query(query, self._policy.gate)
        if core is None or core == query:
            return self._dilate(user_id, hits)
        core_hits = self._search_once(user_id=user_id, query=core, limit=limit)
        if not hits or not core_hits:
            return self._dilate(user_id, hits or core_hits)
        return self._dilate(user_id, self._merge(hits, core_hits, limit))

    def _dilate(self, user_id: str, hits: Sequence[SearchHit]) -> list[SearchHit]:
        turns = max(0, int(self._policy.context_window_turns))
        if turns == 0 or not hits or self._db is None:
            return list(hits)
        grouped: dict[str, list[SearchHit]] = defaultdict(list)
        for hit in hits:
            grouped[hit.message.session_id].append(hit)
        windows: dict[str, list[tuple[int, str]]] = {}
        indexes: dict[str, dict[int, int]] = {}
        for session_id in grouped:
            rows = self._db.connection().execute(
                "SELECT id, content FROM messages "
                "WHERE user_id = ? AND session_id = ? ORDER BY id",
                (user_id, session_id),
            ).fetchall()
            windows[session_id] = [
                (int(row["id"]), str(row["content"])) for row in rows
            ]
            indexes[session_id] = {
                message_id: position
                for position, (message_id, _) in enumerate(windows[session_id])
            }
        dilated = []
        for hit in hits:
            session_id = hit.message.session_id
            session_rows = windows[session_id]
            center = indexes[session_id].get(hit.message.id)
            if center is None:
                dilated.append(hit)
                continue
            lo, hi = max(0, center - turns), min(len(session_rows), center + turns + 1)
            content = "\n".join(text for _, text in session_rows[lo:hi])
            dilated.append(
                SearchHit(
                    message=replace(hit.message, content=content),
                    score=hit.score,
                )
            )
        return dilated

    def _merge(
        self,
        full: Sequence[SearchHit],
        core: Sequence[SearchHit],
        limit: int,
    ) -> list[SearchHit]:
        scores: dict[int, float] = {}
        pool: dict[int, SearchHit] = {}
        for rank, hit in enumerate(full, start=1):
            mid = hit.message.id
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (15 + rank)
            pool[mid] = hit
        for rank, hit in enumerate(core, start=1):
            mid = hit.message.id
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (15 + rank)
            pool[mid] = hit
        ordered = sorted(pool, key=lambda mid: (-scores[mid], mid))
        return [
            SearchHit(message=pool[mid].message, score=scores[mid])
            for mid in ordered[:limit]
        ]

    def _search_once(self, *, user_id: str, query: str, limit: int) -> list[SearchHit]:
        if not any(retriever.name == "dense" for retriever in self._retrievers):
            raise RuntimeError("dense retriever is required")
        candidates = max(limit, self._policy.search_candidates)
        if not self._policy.lane_gating:
            results = [
                self._retrieve(retriever, user_id, query, limit, candidates)
                for retriever in self._retrievers
            ]
            return fuse(results, query=query, policy=self._policy, limit=limit)
        profile = profile_query(query, self._policy.gate)
        by_name = {retriever.name: retriever for retriever in self._retrievers}
        dense = by_name.get("dense")
        if dense is None:
            raise RuntimeError("dense retriever is required")
        lexical = by_name.get("lexical")
        substring = by_name.get("substring")
        if profile.lane == LITERAL:
            primary = [r for r in (lexical, substring) if r is not None]
            weights = {
                "lexical": self._policy.literal_lexical_weight,
                "substring": self._policy.literal_substring_weight,
            }
            fallback = [dense]
            fallback_weights = {"dense": self._policy.dense_channel_weight}
        elif profile.lane == HYBRID:
            primary = list(self._retrievers)
            weights = {
                "dense": self._policy.dense_channel_weight,
                "lexical": self._policy.hybrid_lexical_weight,
                "substring": self._policy.hybrid_substring_weight,
            }
            fallback = []
            fallback_weights = {}
        else:
            primary = [dense]
            weights = {"dense": self._policy.dense_channel_weight}
            fallback = []
            fallback_weights = {}
        results = [
            self._retrieve(retriever, user_id, query, limit, candidates)
            for retriever in primary
        ]
        hits = fuse(
            results, query=query, policy=self._policy, limit=limit, weights=weights
        )
        if hits or not fallback:
            return hits
        fallback_results = [
            self._retrieve(retriever, user_id, query, limit, candidates)
            for retriever in fallback
            if retriever not in primary
        ]
        if not fallback_results:
            return hits
        return fuse(
            fallback_results,
            query=query,
            policy=self._policy,
            limit=limit,
            weights=fallback_weights,
        )

    def _retrieve(
        self,
        retriever: Retriever,
        user_id: str,
        query: str,
        limit: int,
        candidates: int,
    ) -> RetrievalResult:
        return RetrievalResult(
            name=retriever.name,
            hits=retriever.retrieve(
                user_id=user_id,
                query=query,
                limit=self._candidate_limit(retriever, limit, candidates),
            ),
            tiebreak_weight=retriever.tiebreak_weight,
        )

    def _query_limit(self) -> int:
        limit = max(1, int(self._policy.search_query_chars))
        for retriever in self._retrievers:
            if retriever.name != "dense":
                continue
            value = getattr(retriever, "max_input_chars", None)
            if isinstance(value, int) and value > 0:
                limit = min(limit, value)
        return limit

    @staticmethod
    def _candidate_limit(retriever: Retriever, limit: int, candidates: int) -> int:
        if retriever.depth_ratio is None:
            return candidates
        return max(1, int(limit * retriever.depth_ratio))
