from __future__ import annotations

from dataclasses import dataclass

from app.embedding import EmbeddingClient, load_embedding_config
from app.policy import Policy
from app.rerank import RerankClient, load_rerank_config
from app.services.add import AddService
from app.services.search import SearchService
from app.settings import Settings
from app.storage.database import Database
from app.storage.repository import MessageRepository


@dataclass(slots=True)
class Container:
    settings: Settings
    policy: Policy
    database: Database
    repository: MessageRepository
    add: AddService
    search: SearchService
    embedding: EmbeddingClient
    reranker: RerankClient | None = None

    @classmethod
    def build(cls, settings: Settings, policy: Policy) -> Container:
        embedding_config = load_embedding_config()
        if not embedding_config.enabled:
            raise ValueError("embedding.enabled must be true")
        embedding = EmbeddingClient(embedding_config)
        database = Database(settings.resolved_db_path)
        reranker = None
        if policy.rerank_enabled:
            rerank_config = load_rerank_config()
            if rerank_config.enabled:
                reranker = RerankClient(rerank_config)
        repository = MessageRepository(
            database, policy, embedding=embedding, reranker=reranker
        )
        return cls(
            settings=settings,
            policy=policy,
            database=database,
            repository=repository,
            add=AddService(repository, policy, embedding=embedding),
            search=SearchService(repository, policy),
            embedding=embedding,
            reranker=reranker,
        )

    def close(self) -> None:
        self.embedding.close()
        if self.reranker is not None:
            self.reranker.close()
        self.database.close()
