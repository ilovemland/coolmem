from __future__ import annotations

import time
import unittest

from app.policy import Policy
from app.rerank import RerankClient, RerankConfig, RerankError
from app.retrieval import Message, SearchHit
from app.storage.repository import MessageRepository


def message(message_id: int, content: str) -> Message:
    return Message(
        id=message_id,
        request_id="r",
        session_id="s",
        role="user",
        content=content,
        created_at="2026-01-01T00:00:00Z",
    )


class FakeReranker:
    def __init__(self, scores: list[float] | None = None, error: Exception | None = None) -> None:
        self.scores = scores or []
        self.error = error
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        if self.error is not None:
            raise self.error
        return self.scores


class ParseTest(unittest.TestCase):
    def test_parse_maps_scores_by_index(self) -> None:
        body = {"results": [{"index": 1, "relevance_score": 0.25}, {"index": 0, "relevance_score": 0.75}]}
        self.assertEqual(RerankClient._parse(body, 2), [0.75, 0.25])

    def test_parse_rejects_short_results(self) -> None:
        with self.assertRaises(ValueError):
            RerankClient._parse({"results": [{"index": 0, "relevance_score": 1.0}]}, 2)

    def test_parse_rejects_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            RerankClient._parse({"results": [{"index": 5, "relevance_score": 1.0}]}, 1)


class CircuitTest(unittest.TestCase):
    def test_trips_after_threshold(self) -> None:
        config = RerankConfig(
            enabled=True,
            url="http://127.0.0.1:9/v1/rerank",
            model="x",
            timeout_seconds=0.2,
            retries=1,
            failure_threshold=2,
            cooldown_seconds=60.0,
        )
        client = RerankClient(config)
        for _ in range(2):
            with self.assertRaises(RerankError):
                client.rerank("q", ["d"])
        started = time.monotonic()
        with self.assertRaises(RerankError):
            client.rerank("q", ["d"])
        self.assertLess(time.monotonic() - started, 0.05)


class RepositoryRerankTest(unittest.TestCase):
    def build(self, reranker) -> MessageRepository:
        return MessageRepository(None, Policy(rerank_enabled=True), retrievers=[], reranker=reranker)

    def test_reorders_by_rerank_score(self) -> None:
        reranker = FakeReranker(scores=[-5.0, 3.0, 0.0])
        repo = self.build(reranker)
        hits = [SearchHit(message(1, "a"), 0.3), SearchHit(message(2, "b"), 0.2), SearchHit(message(3, "c"), 0.1)]
        ordered = repo._rerank("q", hits)
        self.assertEqual([hit.message.id for hit in ordered], [2, 3, 1])

    def test_uses_core_query(self) -> None:
        reranker = FakeReranker(scores=[0.0])
        repo = self.build(reranker)
        repo._rerank("full question", [SearchHit(message(1, "a"), 0.3)], "core")
        self.assertEqual(reranker.calls[0][0], "core")

    def test_degrades_on_error(self) -> None:
        reranker = FakeReranker(error=RerankError("down"))
        repo = self.build(reranker)
        hits = [SearchHit(message(1, "a"), 0.3), SearchHit(message(2, "b"), 0.2)]
        self.assertEqual([hit.message.id for hit in repo._rerank("q", hits)], [1, 2])

    def test_documents_are_truncated(self) -> None:
        from app.policy import RERANK_DOC_CHARS

        reranker = FakeReranker(scores=[0.0])
        repo = self.build(reranker)
        repo._rerank("q", [SearchHit(message(1, "x" * (RERANK_DOC_CHARS + 50)), 0.3)])
        self.assertEqual(len(reranker.calls[0][1][0]), RERANK_DOC_CHARS)


if __name__ == "__main__":
    unittest.main()
