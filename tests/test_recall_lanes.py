from __future__ import annotations

import pathlib
import tempfile
import unittest
from datetime import UTC, datetime

from app.embedding import EmbeddingError, load_embedding_config
from app.errors import Unavailable
from app.gating import DENSE, HYBRID, LITERAL, core_query, profile_query
from app.policy import Policy
from app.query import clip_query
from app.retrieval import DenseRetriever, Message, RetrievalResult, ScoredHit, SearchHit, fuse
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


class GatingTest(unittest.TestCase):
    def test_quality_first_defaults(self) -> None:
        policy = Policy()
        self.assertFalse(policy.lane_gating)
        self.assertTrue(policy.instruction_decomposition)
        self.assertEqual(policy.context_window_turns, 0)
        self.assertEqual(policy.context_expand_chars, 0)
        self.assertTrue(policy.rerank_enabled)

    def test_lanes(self) -> None:
        cases = {
            "How long did it take for Jon to open his studio?": DENSE,
            "What was Taylor David's job title as of December 18, 2025?": HYBRID,
            "What specific bugs were logged besides the 'TypeError' in autocomplete.js?": HYBRID,
            "DB task index 137": LITERAL,
            "Reach 24 using the numbers [0, 8, 9, 9].": LITERAL,
            "Who was the head of Saarland in May, 1955?": LITERAL,
            "Was technical debt management discussed on March 29th, 2024?": LITERAL,
            "How do I lose 10 kg?": DENSE,
            "Which is better in 2024?": DENSE,
        }
        for query, lane in cases.items():
            with self.subTest(query=query):
                self.assertEqual(profile_query(query).lane, lane)

    def test_core_query(self) -> None:
        books = (
            "childrens fantasy novels about geology\n\n"
            "Use only the provided document corpus. Return all document titles "
            "as a JSON array."
        )
        self.assertEqual(core_query(books), "childrens fantasy novels about geology")
        choice = "Where was the Foreman?\n\nA. A place.\nB. Another place.\n\nPlease provide the option."
        self.assertIsNone(core_query(choice))
        self.assertIsNone(core_query("How long did it take for Jon to open his studio?"))
        chinese = "美国小说\n\n仅使用提供的语料，返回所有标题。"
        self.assertEqual(core_query(chinese), "美国小说")

    def test_long_code_prompt_stays_dense(self) -> None:
        query = (
            "I am planning a long trip with several stops and my passport number is Z3847592. "
            "Please walk me through preparing the visa documentation, the supporting letters, "
            "the photographs, the appointment booking, the insurance paperwork, the transit "
            "rules for each country, the packing checklist, the currency plan, the local "
            "transport options, the emergency contacts, and the arrival checklist for a "
            "multi-city itinerary that lasts several weeks."
        )
        self.assertEqual(profile_query(query).lane, DENSE)


class FusionTest(unittest.TestCase):
    def test_channel_weight_changes_order(self) -> None:
        dense = RetrievalResult(
            name="dense", hits=[ScoredHit(message(1), 0.9)], tiebreak_weight=0.0
        )
        lexical = RetrievalResult(
            name="lexical", hits=[ScoredHit(message(2), 10.0)], tiebreak_weight=0.0
        )
        policy = Policy()
        equal = fuse([dense, lexical], query="alpha beta", policy=policy, limit=2)
        weighted = fuse(
            [dense, lexical],
            query="alpha beta",
            policy=policy,
            limit=2,
            weights={"dense": 1.0, "lexical": 8.0},
        )
        self.assertEqual([hit.message.id for hit in equal], [1, 2])
        self.assertEqual([hit.message.id for hit in weighted], [2, 1])


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
        self.assertGreaterEqual(Policy().max_query_chars, 200_000)

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
        self.assertLessEqual(len(lexical.queries[-1]), Policy().search_query_chars)
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


class DilateTest(unittest.TestCase):
    def test_window_contains_neighbours(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Database(pathlib.Path(temporary) / "test.sqlite3")
            database.migrate()
            repo = MessageRepository(database, Policy(), retrievers=[])
            rows = [
                NewMessage(
                    position=index,
                    role="user",
                    content=f"m{index}",
                    timestamp=None,
                    created_at="2026-01-01T00:00:00Z",
                )
                for index in range(3)
            ]
            result = repo.insert_many(request_id="r", user_id="u", session_id="s", rows=rows)
            middle = Message(
                id=result.ids[1],
                request_id="r",
                session_id="s",
                role="user",
                content="m1",
                created_at="2026-01-01T00:00:00Z",
            )
            windowed = MessageRepository(database, Policy(context_window_turns=2), retrievers=[])
            dilated = windowed._dilate("u", [SearchHit(middle, 1.0)])
            self.assertEqual(dilated[0].message.content, "m0\nm1\nm2")
            off = MessageRepository(database, Policy(context_window_turns=0), retrievers=[])
            kept = off._dilate("u", [SearchHit(middle, 1.0)])
            self.assertEqual(kept[0].message.content, "m1")


class MergeTest(unittest.TestCase):
    def test_merge_prefers_double_hits(self) -> None:
        repo = MessageRepository(None, Policy(), retrievers=[])
        full = [SearchHit(message(1), 0.5), SearchHit(message(2), 0.4)]
        core = [SearchHit(message(2), 0.3), SearchHit(message(3), 0.2)]
        merged = repo._merge(full, core, 10)
        self.assertEqual([hit.message.id for hit in merged], [2, 1, 3])


class RoutingTest(unittest.TestCase):
    def build(self, dense_hits: list[ScoredHit] | None = None):
        dense = FakeRetriever("dense", dense_hits if dense_hits is not None else [ScoredHit(message(1), 0.9)])
        lexical = FakeRetriever("lexical", [ScoredHit(message(2), 5.0)])
        substring = FakeRetriever("substring", [ScoredHit(message(3), 5.0)], depth_ratio=0.5)
        repo = MessageRepository(None, Policy(lane_gating=True), retrievers=[lexical, substring, dense])
        return repo, dense, lexical, substring

    def test_dense_lane_runs_only_dense(self) -> None:
        repo, dense, lexical, substring = self.build()
        hits = repo.search(user_id="u", query="How long did it take for Jon to open his studio?", limit=10)
        self.assertEqual([hit.message.id for hit in hits], [1])
        self.assertEqual(dense.limits, [200])
        self.assertEqual(lexical.limits, [])
        self.assertEqual(substring.limits, [])

    def test_literal_lane_skips_dense(self) -> None:
        repo, dense, lexical, substring = self.build()
        hits = repo.search(user_id="u", query="DB task index 137", limit=10)
        self.assertEqual([hit.message.id for hit in hits], [2, 3])
        self.assertEqual(dense.limits, [])
        self.assertEqual(lexical.limits, [200])
        self.assertEqual(substring.limits, [5])

    def test_hybrid_lane_runs_all(self) -> None:
        repo, dense, lexical, substring = self.build()
        repo.search(user_id="u", query="What was Taylor David's job title as of December 18, 2025?", limit=10)
        self.assertEqual(dense.limits, [200])
        self.assertEqual(lexical.limits, [200])
        self.assertEqual(substring.limits, [5])

    def test_dense_empty_does_not_fall_back(self) -> None:
        repo, dense, lexical, substring = self.build(dense_hits=[])
        hits = repo.search(user_id="u", query="How long did it take for Jon to open his studio?", limit=10)
        self.assertEqual(hits, [])
        self.assertEqual(dense.limits, [200])
        self.assertEqual(lexical.limits, [])
        self.assertEqual(substring.limits, [])


if __name__ == "__main__":
    unittest.main()
