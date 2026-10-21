from __future__ import annotations

import hashlib
import heapq
import math

import numpy
import sqlite3
from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace

from app.embedding import blob_to_vector, vector_to_blob
from app.errors import PayloadConflict
from app.rerank import RerankClient as Reranker, RerankError
from app.gating import core_query, query_segments
from app.normalize import expanded_text
from app.query import clip_query
from app.policy import (
    RERANK_DEPTH,
    RERANK_DOC_CHARS,
    SEARCH_CANDIDATES,
    SEARCH_QUERY_CHARS,
)
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


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class VectorCache:
    def __init__(self, capacity: int = 4) -> None:
        self._items: OrderedDict[tuple[str, str, int], tuple[list[Message], numpy.ndarray]] = OrderedDict()
        self._capacity = max(1, capacity)

    def get(self, key: tuple[str, str, int]):
        item = self._items.get(key)
        if item is not None:
            self._items.move_to_end(key)
        return item

    def put(self, key: tuple[str, str, int], value) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self._capacity:
            self._items.popitem(last=False)

    def forget(self, user_id: str) -> None:
        for key in [key for key in self._items if key[0] == user_id]:
            del self._items[key]


@dataclass(frozen=True, slots=True)
class NewMessage:
    position: int
    role: str
    content: str
    timestamp: int | None
    created_at: str
    nearby_text: str = ""
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
        reranker: Reranker | None = None,
    ) -> None:
        self._db = database
        self._policy = policy
        self._reranker = reranker
        self._vectors = VectorCache()
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
                         content, source_content, search_text, nearby_text, timestamp, created_at)
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
                        row.nearby_text,
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
            self._vectors.forget(user_id)
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
        entry = self._vector_entry(user_id, space, len(query_vector))
        if entry is None:
            return []
        messages, matrix, identities = entry
        scores = matrix @ numpy.asarray(query_vector, dtype=numpy.float64)
        width = min(limit, len(messages))
        if width <= 0:
            return []
        order = numpy.lexsort((identities, -scores))[:width]
        return [ScoredHit(message=messages[int(index)], raw=float(scores[index])) for index in order]

    def _vector_entry(
        self, user_id: str, space: str, dimension: int
    ) -> tuple[list[Message], numpy.ndarray] | None:
        key = (user_id, space, dimension)
        cached = self._vectors.get(key)
        if cached is not None:
            return cached
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
        ).fetchall()
        messages: list[Message] = []
        vectors: list[numpy.ndarray] = []
        for row in rows:
            if int(row["dimension"]) != dimension:
                continue
            blob = row["vector"]
            if not isinstance(blob, (bytes, bytearray)) or len(blob) != dimension * 4:
                continue
            vectors.append(numpy.frombuffer(bytes(blob), dtype=numpy.float32))
            messages.append(
                Message(
                    id=int(row["id"]),
                    request_id=str(row["request_id"]),
                    session_id=str(row["session_id"]),
                    role=str(row["role"]),
                    content=str(row["content"]),
                    created_at=str(row["created_at"]),
                )
            )
        if not vectors:
            return None
        entry = (
            messages,
            numpy.stack(vectors).astype(numpy.float64),
            numpy.array([message.id for message in messages], dtype=numpy.int64),
        )
        self._vectors.put(key, entry)
        return entry

    def cached_vectors(
        self, space: str, hashes: Sequence[str], dimension: int
    ) -> dict[str, list[float]]:
        if not hashes or self._db is None or dimension <= 0:
            return {}
        found: dict[str, list[float]] = {}
        connection = self._db.connection()
        marks = ",".join("?" for _ in hashes)
        rows = connection.execute(
            f"SELECT content_hash, vector, dimension FROM embedding_cache "
            f"WHERE space = ? AND content_hash IN ({marks})",
            (space, *hashes),
        ).fetchall()
        for row in rows:
            if int(row["dimension"]) != dimension:
                continue
            blob = row["vector"]
            if not isinstance(blob, (bytes, bytearray)) or len(blob) != dimension * 4:
                continue
            found[str(row["content_hash"])] = list(
                numpy.frombuffer(bytes(blob), dtype=numpy.float32)
            )
        return found

    def store_vectors(
        self,
        space: str,
        entries: Sequence[tuple[str, Sequence[float]]],
    ) -> None:
        if not entries or self._db is None:
            return
        now = utc_now()
        with self._db.transaction() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embedding_cache "
                "(space, content_hash, dimension, vector, created_at) VALUES (?, ?, ?, ?, ?)",
                [
                    (space, digest, len(vector), vector_to_blob(vector), now)
                    for digest, vector in entries
                ],
            )

    def cached_rerank(
        self, model: str, query_hash: str, document_hashes: Sequence[str]
    ) -> dict[str, float]:
        if not document_hashes or self._db is None:
            return {}
        marks = ",".join("?" for _ in document_hashes)
        rows = self._db.connection().execute(
            f"SELECT document_hash, score FROM rerank_cache "
            f"WHERE model = ? AND query_hash = ? AND document_hash IN ({marks})",
            (model, query_hash, *document_hashes),
        ).fetchall()
        return {str(row["document_hash"]): float(row["score"]) for row in rows}

    def store_rerank(
        self,
        model: str,
        query_hash: str,
        entries: Sequence[tuple[str, float]],
    ) -> None:
        if not entries or self._db is None:
            return
        now = utc_now()
        with self._db.transaction() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO rerank_cache "
                "(model, query_hash, document_hash, score, created_at) VALUES (?, ?, ?, ?, ?)",
                [(model, query_hash, digest, score, now) for digest, score in entries],
            )

    def count(self) -> int:
        row = self._db.connection().execute("SELECT COUNT(*) AS n FROM messages").fetchone()
        return int(row["n"])

    def search(self, *, user_id: str, query: str, limit: int) -> list[SearchHit]:
        query = clip_query(query, SEARCH_QUERY_CHARS)
        pool = limit
        if self._policy.rerank_enabled and limit >= max(1, RERANK_DEPTH // 4):
            pool = max(limit, RERANK_DEPTH)
        hits = self._search_once(user_id=user_id, query=query, limit=pool)
        if self._policy.segment_search:
            for segment in query_segments(query)[1:]:
                segment_hits = self._search_once(user_id=user_id, query=segment, limit=pool)
                if hits and segment_hits:
                    hits = self._merge(hits, segment_hits, pool)
                elif segment_hits:
                    hits = segment_hits
        hits = self._rerank(query, hits, core_query(query))
        return hits[:limit]

    def _rerank(
        self, query: str, hits: Sequence[SearchHit], core: str | None = None
    ) -> list[SearchHit]:
        if not self._policy.rerank_enabled or self._reranker is None or not hits:
            return list(hits)
        depth = min(len(hits), RERANK_DEPTH)
        head = list(hits[:depth])
        tail = list(hits[depth:])
        documents = [hit.message.content[:RERANK_DOC_CHARS] for hit in head]
        target = core or query
        query_hash = _digest(target)
        document_hashes = [_digest(document) for document in documents]
        model = f"{getattr(self._reranker, 'model', '')}|{RERANK_DOC_CHARS}|{RERANK_DEPTH}"
        cached = self.cached_rerank(model, query_hash, document_hashes)
        missing = [index for index, digest in enumerate(document_hashes) if digest not in cached]
        if missing:
            try:
                fresh = self._reranker.rerank(target, [documents[index] for index in missing])
            except RerankError:
                return list(hits)
            if len(fresh) != len(missing):
                return list(hits)
            self.store_rerank(
                model,
                query_hash,
                [(document_hashes[index], value) for index, value in zip(missing, fresh, strict=True)],
            )
            for index, value in zip(missing, fresh, strict=True):
                cached[document_hashes[index]] = float(value)
        raw = [cached[digest] for digest in document_hashes]
        weights = [1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value)))) for value in raw]
        order = sorted(
            range(len(head)), key=lambda index: (-weights[index], head[index].message.id)
        )
        head = [
            SearchHit(message=head[index].message, score=weights[index])
            for index in order
        ]
        return head + tail

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
        candidates = max(limit, SEARCH_CANDIDATES)
        results = [
            self._retrieve(retriever, user_id, query, limit, candidates)
            for retriever in self._retrievers
        ]
        return fuse(results, query=query, limit=limit)

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

    @staticmethod
    def _candidate_limit(retriever: Retriever, limit: int, candidates: int) -> int:
        if retriever.depth_ratio is None:
            return candidates
        return max(1, int(limit * retriever.depth_ratio))
