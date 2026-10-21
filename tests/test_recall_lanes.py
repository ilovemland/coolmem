from __future__ import annotations

import pathlib
import tempfile
import unittest
from datetime import UTC, datetime

from app.embedding import EmbeddingError, load_embedding_config
from app.errors import Unavailable
from app.gating import core_query, query_segments
from app.policy import Policy
from app.query import clip_query
from app.retrieval import DenseRetriever, Message, ScoredHit, SearchHit
from app.services.add import AddService, IncomingMessage
from app.services.search import SearchService
from app.storage.database import Database
from app.storage.repository import MessageRepository, NewMessage
from app.timefmt import format_time


def message(mid: int, content: str = "text") -> Message:
    return Message(
        id=mid,
        request_id=f"r{mid}",
        session_id="s",
        role="user",
        content=content,
        created_at="2026-01-01T00:00:00Z",
    )


class FakeRetriever:
    def __init__(self, name: str, hits: list[ScoredHit], depth_ratio: float | None = None) -> None:
        self.name = name
        self.tiebreak_weight = 0.0
        self.depth_ratio = depth_ratio
        self.hits = hits
        self.limits: list[int] = []
        self.queries: list[str] = []

    def retrieve(self, *, user_id: str, query: str, limit: int) -> list[ScoredHit]:
        self.limits.append(limit)
        self.queries.append(query)
        return self.hits[:limit]




class TimeTextTest(unittest.TestCase):
    def test_format(self) -> None:
        ms = int(datetime(2022, 1, 21, 19, 31, tzinfo=UTC).timestamp() * 1000)
        self.assertEqual(
            format_time(ms),
            "2022-01-21 19:31 2022年1月21日 星期五 January 2022 Friday 07:31 PM",
        )
        self.assertEqual(format_time(None), "")

    def test_content_and_source_content(self) -> None:
        ms = int(datetime(2022, 1, 21, 19, 31, tzinfo=UTC).timestamp() * 1000)
        with tempfile.TemporaryDirectory() as temporary:
            database = Database(pathlib.Path(temporary) / "test.sqlite3")
            database.migrate()
            repo = MessageRepository(database, Policy(), retrievers=[])
            row = NewMessage(
                position=0,
                role="user",
                content="hello",
                timestamp=ms,
                created_at="2026-01-01T00:00:00Z",
                time_text=format_time(ms),
            )
            result = repo.insert_many(request_id="r", user_id="u", session_id="s", rows=[row])
            stored = database.connection().execute(
                "select content, source_content from messages where id = ?", (result.ids[0],)
            ).fetchone()
            self.assertEqual(
                stored["content"],
                "2022-01-21 19:31 2022年1月21日 星期五 January 2022 Friday 07:31 PM hello",
            )
            self.assertEqual(stored["source_content"], "hello")
            replay = repo.insert_many(request_id="r", user_id="u", session_id="s", rows=[row])
            self.assertEqual(replay.skipped, 1)
            database.close()


class ClipQueryTest(unittest.TestCase):
    def test_short_query_unchanged(self) -> None:
        self.assertEqual(clip_query("hello world", 100), "hello world")

    def test_tail_question_is_kept(self) -> None:
        query = "Document title\n" + ("filler text line\n" * 3000) + "What is the answer?"
        clipped = clip_query(query, 1000)
        self.assertLessEqual(len(clipped), 1000)
        self.assertIn("What is the answer?", clipped)
        self.assertIn("Document title", clipped)

    def test_policy_allows_long_queries(self) -> None:
        from app.policy import MAX_QUERY_CHARS

        self.assertGreaterEqual(MAX_QUERY_CHARS, 200_000)

    def test_search_service_accepts_long_query(self) -> None:
        class EmptyRepository:
            def search(self, *, user_id: str, query: str, limit: int):
                return []

        service = SearchService(EmptyRepository(), Policy())
        self.assertEqual(service.search(user_id="u", query="x" * 186_000, top_k=5), [])

    def test_repository_clips_before_retrieval(self) -> None:
        lexical = FakeRetriever("lexical", [])
        dense = FakeRetriever("dense", [])
        repo = MessageRepository(None, Policy(), retrievers=[lexical, dense])
        query = "Header\n" + ("filler text line\n" * 3000) + "What is the answer?"
        repo.search(user_id="u", query=query, limit=5)
        self.assertEqual(lexical.queries[-1], dense.queries[-1])
        from app.policy import SEARCH_QUERY_CHARS

        self.assertLessEqual(len(lexical.queries[-1]), SEARCH_QUERY_CHARS)
        self.assertIn("What is the answer?", lexical.queries[-1])


class EmbeddingRequiredTest(unittest.TestCase):
    def test_missing_embedding_config_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = pathlib.Path(temporary) / "missing.json"
            with self.assertRaises(ValueError):
                load_embedding_config(missing)

    def test_repository_requires_embedding(self) -> None:
        with self.assertRaises(ValueError):
            MessageRepository(None, Policy())

    def test_add_service_requires_embedding(self) -> None:
        repo = MessageRepository(None, Policy(), retrievers=[])
        with self.assertRaises(ValueError):
            AddService(repo, Policy(), embedding=None)

    def test_dense_failure_is_not_swallowed(self) -> None:
        class FailingEmbedder:
            space = "x"
            tiebreak_weight = 0.0

            def embed_query(self, text: str) -> list[float]:
                raise EmbeddingError("down")

        dense = DenseRetriever(object(), FailingEmbedder())
        with self.assertRaises(EmbeddingError):
            dense.retrieve(user_id="u", query="q", limit=10)

    def test_search_maps_embedding_failure_to_unavailable(self) -> None:
        class FailingRepository:
            def search(self, *, user_id: str, query: str, limit: int):
                raise EmbeddingError("down")

        service = SearchService(FailingRepository(), Policy())
        with self.assertRaises(Unavailable):
            service.search(user_id="u", query="q", top_k=5)

    def test_add_maps_embedding_failure_to_unavailable(self) -> None:
        class FailingEmbedding:
            space = "x"
            model = "m"
            dimension = 4

            def embed_documents(self, texts):
                raise EmbeddingError("down")

        repo = MessageRepository(None, Policy(), retrievers=[])
        service = AddService(repo, Policy(), embedding=FailingEmbedding())
        with self.assertRaises(Unavailable):
            service.add(
                request_id="r",
                user_id="u",
                session_id="s",
                messages=[IncomingMessage(role="user", content="hello")],
            )



class MergeTest(unittest.TestCase):
    def test_merge_prefers_double_hits(self) -> None:
        repo = MessageRepository(None, Policy(), retrievers=[])
        full = [SearchHit(message(1), 0.5), SearchHit(message(2), 0.4)]
        core = [SearchHit(message(2), 0.3), SearchHit(message(3), 0.2)]
        merged = repo._merge(full, core, 10)
        self.assertEqual([hit.message.id for hit in merged], [2, 1, 3])


