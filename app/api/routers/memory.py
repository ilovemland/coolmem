from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.api.deps import AddServiceDep, SearchServiceDep
from app.api.schemas import (
    AddRequest,
    AddResponse,
    ErrorResponse,
    SearchItemOut,
    SearchRequest,
    SearchResponse,
)
from app.services.add import IncomingMessage

router = APIRouter(tags=["memory"])

_ERRORS: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "Idempotency conflict"},
    401: {"model": ErrorResponse, "description": "Missing or invalid bearer token"},
    413: {"model": ErrorResponse, "description": "Body too large"},
    422: {"model": ErrorResponse, "description": "Validation or business-rule error"},
}


@router.post("/add", response_model=AddResponse, responses=_ERRORS)
def add_messages(
    payload: AddRequest,
    service: AddServiceDep,
) -> AddResponse:
    service.add(
        request_id=payload.request_id,
        user_id=payload.user_id,
        session_id=payload.session_id,
        messages=[
            IncomingMessage(
                role=message.role,
                content=message.content,
                timestamp=message.timestamp,
            )
            for message in payload.messages
        ],
    )
    return AddResponse(
        success=True,
        request_id=payload.request_id,
        user_id=payload.user_id,
        session_id=payload.session_id,
    )


@router.post("/search", response_model=SearchResponse, responses=_ERRORS)
def search_messages(
    payload: SearchRequest,
    service: SearchServiceDep,
) -> SearchResponse:
    results = service.search(
        user_id=payload.user_id,
        query=payload.query,
        top_k=payload.top_k,
    )
    return SearchResponse(data=[SearchItemOut.from_domain(result) for result in results])
