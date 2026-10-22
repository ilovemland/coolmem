from __future__ import annotations

import hmac
import json
import logging
import os
import re
import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.errors import error_payload

logger = logging.getLogger("uvicorn.error")

_TRACE_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,64}")
_HEALTH_PATHS = frozenset({"/health", "/health/"})
_LOG_BODY = os.environ.get("COOLMEM_LOG_BODY", "1") != "0"
_BODY_CAP = 64 * 1024 * 1024
_TEXT_PREVIEW = int(os.environ.get("COOLMEM_LOG_TEXT_CHARS", "0"))
_MSG_PREVIEW = int(os.environ.get("COOLMEM_LOG_MSG_COUNT", "0"))


def _summarize(raw: bytes) -> str:
    if not raw:
        return "body=-"
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return f"body=<unparsed {len(raw)}B>"
    if not isinstance(payload, dict):
        return f"body=<non-object {len(raw)}B>"
    parts = [f"bytes={len(raw)}"]
    for key in ("request_id", "user_id", "session_id"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            parts.append(f"{key}={value}")
    if isinstance(payload.get("query"), str):
        query = payload["query"]
        if _TEXT_PREVIEW > 0:
            query = query[:_TEXT_PREVIEW]
        parts.append(f"query={query!r}")
    if isinstance(payload.get("options"), list):
        parts.append(f"options={len(payload['options'])}")
    if isinstance(payload.get("top_k"), int):
        parts.append(f"top_k={payload['top_k']}")
    messages = payload.get("messages")
    if isinstance(messages, list):
        parts.append(f"messages={len(messages)}")
        shown = messages if _MSG_PREVIEW <= 0 else messages[:_MSG_PREVIEW]
        for index, message in enumerate(shown):
            if not isinstance(message, dict):
                parts.append(f"m{index}=<non-object>")
                continue
            role = message.get("role", "?")
            content = str(message.get("content", ""))
            if _TEXT_PREVIEW > 0:
                content = content[:_TEXT_PREVIEW]
            stamp = message.get("timestamp")
            label = f"m{index}[{role}]" + (f"@{stamp}" if stamp is not None else "")
            parts.append(f"{label}={content!r}")
        if _MSG_PREVIEW > 0 and len(messages) > _MSG_PREVIEW:
            parts.append(f"...+{len(messages) - _MSG_PREVIEW}")
    return " ".join(parts)


async def _read_body(receive):
    chunks: list[bytes] = []
    size = 0
    while True:
        message = await receive()
        if message["type"] != "http.request":
            return b"".join(chunks), message
        chunk = message.get("body", b"")
        size += len(chunk)
        if size > _BODY_CAP:
            return b"".join(chunks), message
        chunks.append(chunk)
        if not message.get("more_body"):
            return b"".join(chunks), None


def _replay(body: bytes, tail):
    sent = False

    async def receive() -> Message:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        if tail is not None:
            return tail
        return {"type": "http.disconnect"}

    return receive


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
        summary = ""
        if _LOG_BODY and scope.get("method") == "POST":
            body, tail = await _read_body(receive)
            receive = _replay(body, tail)
            summary = _summarize(body)

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
                "%s %s %d trace=%s %s",
                scope.get("method", "-"),
                scope.get("path", "-"),
                status_code,
                trace_id,
                summary,
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
