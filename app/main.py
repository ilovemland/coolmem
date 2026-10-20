from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app import __version__
from app.api.middleware import (
    BearerAuthMiddleware,
    BodySizeLimitMiddleware,
    RequestContextMiddleware,
)
from app.api.routers import health, memory
from app.container import Container
from app.errors import install_error_handlers
from app.policy import DEFAULT_POLICY, Policy
from app.settings import Settings

logger = logging.getLogger("uvicorn.error")


def create_app(
    settings: Settings | None = None,
    policy: Policy | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    policy = policy or DEFAULT_POLICY
    container = Container.build(settings, policy)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        version = container.database.migrate()
        logger.info(
            "startup service=coolmem version=%s db=%s schema=%d",
            __version__,
            container.database.path,
            version,
        )
        try:
            yield
        finally:
            container.close()
            logger.info("shutdown service=coolmem")

    app = FastAPI(
        title="coolmem",
        version=__version__,
        summary="Append-only memory store with hybrid full-text retrieval",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.container = container

    install_error_handlers(app)

    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)
    app.add_middleware(BearerAuthMiddleware, token=settings.bearer_token)
    app.add_middleware(RequestContextMiddleware)

    app.include_router(health.router)
    app.include_router(memory.router)
    return app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        settings = Settings.from_env()
        app = create_app(settings)
    except ValueError as exc:
        raise SystemExit(f"configuration error: {exc}") from None
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
        log_config=None,
        access_log=False,
    )


if __name__ == "__main__":
    main()
