from __future__ import annotations

from dataclasses import dataclass

from app.embedding import EmbeddingClient, load_embedding_config
from app.policy import Policy
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

    @classmethod
    def build(cls, settings: Settings, policy: Policy) -> Container:
        embedding_config = load_embedding_config()
        if not embedding_config.enabled:
            raise ValueError("embedding.enabled must be true")
        embedding = EmbeddingClient(embedding_config)
        database = Database(settings.resolved_db_path)
        repository = MessageRepository(database, policy, embedding=embedding)
        return cls(
            settings=settings,
            policy=policy,
            database=database,
            repository=repository,
            add=AddService(repository, policy, embedding=embedding),
            search=SearchService(repository, policy),
            embedding=embedding,
        )

    def close(self) -> None:
        self.embedding.close()
        self.database.close()
