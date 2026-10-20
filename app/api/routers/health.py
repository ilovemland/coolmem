from __future__ import annotations

import sqlite3

from fastapi import APIRouter

from app.api.deps import ContainerDep
from app.api.schemas import HealthOut
from app.errors import Unavailable

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthOut)
def health(container: ContainerDep) -> HealthOut:
    try:
        container.repository.count()
    except sqlite3.Error as exc:
        raise Unavailable(
            "database is not readable", details={"reason": type(exc).__name__}
        ) from exc
    return HealthOut(status="ok")
