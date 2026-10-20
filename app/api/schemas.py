from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.services.search import SearchResult as SearchResultDomain


class MessageIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str
    timestamp: int | None = Field(default=None, ge=0)


class AddRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    messages: list[MessageIn] = Field(min_length=1)
    user_id: str
    session_id: str


class AddResponse(BaseModel):
    success: bool
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    options: list[str] | None = None
    user_id: str
    top_k: int


class SearchItemOut(BaseModel):
    id: str
    content: str
    score: float
    created_at: str

    @classmethod
    def from_domain(cls, result: SearchResultDomain) -> SearchItemOut:
        return cls(
            id=f"mem_{result.message.id}",
            content=result.message.content,
            score=result.score,
            created_at=result.message.created_at,
        )


class SearchResponse(BaseModel):
    data: list[SearchItemOut]


class HealthOut(BaseModel):
    status: str


class ErrorBody(BaseModel):
    code: str
    message: str
    trace_id: str | None = None
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
