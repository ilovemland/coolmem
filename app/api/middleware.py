from __future__ import annotations

import hmac
import logging
import re
import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.errors import error_payload

logger = logging.getLogger("uvicorn.error")

_TRACE_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,64}")
_HEALTH_PATHS = frozenset({"/health", "/health/"})


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get("x-request-id", "")
        trace_id = incoming if _TRACE_ID_RE.fullmatch(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["trace_id"] = trace_id

        status_code = 500

        async def send_with_trace_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                MutableHeaders(scope=message)["x-request-id"] = trace_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_trace_id)
        finally:
            logger.info(
                "%s %s %d",
                scope.get("method", "-"),
                scope.get("path", "-"),
                status_code,
            )


class BearerAuthMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        token: str,
        exempt_paths: frozenset[str] = _HEALTH_PATHS,
    ) -> None:
        self.app = app
        self.token = token.encode("utf-8")
        self.exempt_paths = exempt_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") in self.exempt_paths:
            await self.app(scope, receive, send)
            return

        header = Headers(scope=scope).get("authorization", "")
        scheme, _, value = header.strip().partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(
            value.strip().encode("utf-8"), self.token
        ):
            await _send_error(
                scope,
                receive,
                send,
                401,
                "unauthorized",
                "bearer token is missing or invalid",
                headers={"WWW-Authenticate": "Bearer"},
            )
            return

        await self.app(scope, receive, send)


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        raw = Headers(scope=scope).get("content-length")
        if raw is not None:
            try:
                length = int(raw)
            except ValueError:
                await _send_error(
                    scope,
                    receive,
                    send,
                    400,
                    "invalid_content_length",
                    "Content-Length must be an integer",
                )
                return
            if length > self.max_bytes:
                await _send_error(
                    scope,
                    receive,
                    send,
                    413,
                    "payload_too_large",
                    f"request body exceeds {self.max_bytes} bytes",
                    details={"limit_bytes": self.max_bytes},
                )
                return

        await self.app(scope, receive, send)


async def _send_error(
    scope: Scope,
    receive: Receive,
    send: Send,
    status_code: int,
    code: str,
    message: str,
    *,
    details: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
) -> None:
    trace_id = scope.get("state", {}).get("trace_id")
    response: Response = JSONResponse(
        error_payload(code, message, trace_id=trace_id, details=details),
        status_code=status_code,
        headers=headers,
    )
    await response(scope, receive, send)
