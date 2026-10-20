from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("uvicorn.error")


class AppError(Exception):
    status_code: int = 500
    code: str = "internal_error"
    default_message: str = "internal error"

    def __init__(
        self,
        message: str | None = None,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.message = message or self.default_message
        super().__init__(self.message)
        self.details: dict[str, Any] = dict(details or {})


class InvalidInput(AppError):
    status_code = 422
    code = "invalid_input"
    default_message = "request rejected by a business rule"


class PayloadConflict(AppError):
    status_code = 400
    code = "request_id_payload_conflict"
    default_message = "request_id already used with a different payload"


class Unavailable(AppError):
    status_code = 503
    code = "service_unavailable"
    default_message = "service dependency is unavailable"


def error_payload(
    code: str,
    message: str,
    *,
    trace_id: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"code": code, "message": message}
    if trace_id:
        body["trace_id"] = trace_id
    if details:
        body["details"] = dict(details)
    return {"error": body}


def trace_id_of(request: Request) -> str | None:
    return getattr(request.state, "trace_id", None)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        trace_id = trace_id_of(request)
        if exc.status_code >= 500:
            logger.warning(
                "app_error trace_id=%s code=%s status=%d message=%s",
                trace_id,
                exc.code,
                exc.status_code,
                exc.message,
            )
        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload(exc.code, exc.message, trace_id=trace_id, details=exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {
                "loc": list(error.get("loc", ())),
                "msg": error.get("msg", ""),
                "type": error.get("type", ""),
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=error_payload(
                "validation_error",
                "request failed schema validation",
                trace_id=trace_id_of(request),
                details={"errors": details},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload(
                "http_error",
                str(exc.detail),
                trace_id=trace_id_of(request),
            ),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        trace_id = trace_id_of(request)
        logger.exception(
            "unhandled_error trace_id=%s method=%s path=%s",
            trace_id,
            request.method,
            request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content=error_payload(
                "internal_error",
                "internal server error",
                trace_id=trace_id,
            ),
        )
