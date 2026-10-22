from __future__ import annotations

import json
import logging
import os
import re
import threading
import urllib.error
import urllib.request

from app.storage.database import Database

logger = logging.getLogger("uvicorn.error")

PROFILE_CHARS = int(os.environ.get("COOLMEM_PROFILE_CHARS", "2000"))
PROFILE_EVERY = int(os.environ.get("COOLMEM_PROFILE_EVERY", "10"))
PROFILE_WINDOW = int(os.environ.get("COOLMEM_PROFILE_WINDOW", "200"))
PROFILE_TIMEOUT = float(os.environ.get("COOLMEM_PROFILE_TIMEOUT", "30"))
PROFILE_URL = os.environ.get("COOLMEM_PROFILE_URL", "https://api.xingdingwangluo.cn/v1/chat/completions")
PROFILE_MODEL = os.environ.get("COOLMEM_PROFILE_MODEL", "gpt-6-luna")
PROFILE_KEY_ENV = os.environ.get("COOLMEM_PROFILE_KEY_ENV", "")
PROFILE_KEY_FILE = os.environ.get("COOLMEM_PROFILE_KEY_FILE", "luna.txt")

INSTRUCTIONS = """You maintain a user profile for a memory retrieval system.

Input: the previous profile (may be empty) and new dialogue lines, oldest first.
Output: an updated profile, plain markdown, at most {limit} characters.

Sections, keep every section present:
## Identity
## Current state (latest value first, annotate when it changed)
## Stable preferences
## Standing instructions
## Key relationships
## Timeline anchors

Rules:
- New statements override older ones. Never keep two conflicting current values; move the old one into a parenthetical with its date.
- Record only what the dialogue states. No guesses, no generic filler.
- Keep absolute dates when they appear; convert relative ones only when the message timestamp makes it unambiguous.
- Write in English. Prefer short bullets over prose.
"""


class ProfileService:
    def __init__(self, database: Database) -> None:
        self._db = database
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self._pending: set[str] = set()

    def get(self, user_id: str) -> str | None:
        row = (
            self._db.connection()
            .execute("SELECT content FROM profiles WHERE user_id = ?", (user_id,))
            .fetchone()
        )
        return str(row["content"]) if row and row["content"] else None

    def maybe_refresh(self, user_id: str) -> None:
        if PROFILE_EVERY <= 0 or not _api_key():
            return
        requests = (
            self._db.connection()
            .execute(
                "SELECT COUNT(DISTINCT request_id) AS n FROM messages WHERE user_id = ?",
                (user_id,),
            )
            .fetchone()["n"]
        )
        if int(requests) % PROFILE_EVERY != 0:
            return
        with self._guard:
            if user_id in self._pending:
                return
            self._pending.add(user_id)
        threading.Thread(target=self._run, args=(user_id,), daemon=True).start()

    def _run(self, user_id: str) -> None:
        try:
            self._refresh(user_id)
        except Exception as exc:
            logger.warning("profile refresh failed user=%s err=%s", user_id, exc)
        finally:
            with self._guard:
                self._pending.discard(user_id)

    def _refresh(self, user_id: str) -> None:
        rows = (
            self._db.connection()
            .execute(
                """
                SELECT role, content FROM messages
                 WHERE user_id = ? AND is_feedback = 0
                 ORDER BY id DESC LIMIT ?
                """,
                (user_id, PROFILE_WINDOW),
            )
            .fetchall()
        )
        if not rows:
            return
        total = (
            self._db.connection()
            .execute("SELECT COUNT(*) AS n FROM messages WHERE user_id = ?", (user_id,))
            .fetchone()["n"]
        )
        previous = self.get(user_id) or "(none yet)"
        lines = [f"{row['role']}: {row['content']}" for row in reversed(rows)]
        prompt = (
            INSTRUCTIONS.format(limit=PROFILE_CHARS)
            + "\n\nPrevious profile:\n"
            + previous
            + "\n\nNew dialogue:\n"
            + "\n".join(lines)
        )
        content = self._call_llm(prompt)
        if not content:
            return
        content = content.strip()[: PROFILE_CHARS * 2]
        if not content:
            return
        now = rows and self._db.connection().execute("SELECT datetime('now') AS t").fetchone()["t"]
        self._db.connection().execute(
            """
            INSERT INTO profiles (user_id, content, message_count, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                content = excluded.content,
                message_count = excluded.message_count,
                updated_at = excluded.updated_at
            """,
            (user_id, content, int(total), str(now)),
        )
        self._db.connection().commit()
        logger.info(
            "profile updated user=%s messages=%d chars=%d", user_id, int(total), len(content)
        )

    def _call_llm(self, prompt: str) -> str | None:
        key = _api_key()
        if not key:
            return None
        body = {
            "model": PROFILE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": PROFILE_CHARS,
        }
        request = urllib.request.Request(
            PROFILE_URL,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=PROFILE_TIMEOUT) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            logger.warning("profile llm unavailable: %s", exc)
            return None
        choices = payload.get("choices") or []
        if not choices:
            return None
        return str(choices[0].get("message", {}).get("content", ""))


_KEY_RE = re.compile(r"sk-[A-Za-z0-9]{16,}")


def _api_key() -> str:
    value = os.environ.get(PROFILE_KEY_ENV, "") if PROFILE_KEY_ENV else ""
    if value:
        return value
    try:
        with open(PROFILE_KEY_FILE, encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return ""
    found = _KEY_RE.findall(text)
    return found[0] if found else ""
