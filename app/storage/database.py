from __future__ import annotations

import contextlib
import itertools
import logging
import sqlite3
import threading
from collections.abc import Callable, Iterator

from app.policy import NEARBY_SPAN
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger("uvicorn.error")

_SCHEMA_VERSION = 15

_DDL_V1: tuple[str, ...] = (
    """
    CREATE TABLE messages (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id  TEXT    NOT NULL,
        position    INTEGER NOT NULL,
        user_id     TEXT    NOT NULL,
        session_id  TEXT    NOT NULL,
        role        TEXT    NOT NULL,
        content     TEXT    NOT NULL,
        timestamp   INTEGER,
        created_at  TEXT    NOT NULL,
        UNIQUE (request_id, position)
    )
    """,
    "CREATE INDEX messages_user_idx ON messages (user_id, id)",
    """
    CREATE VIRTUAL TABLE messages_fts USING fts5 (
        content, role,
        tokenize = 'unicode61 remove_diacritics 2'
    )
    """,
    """
    CREATE VIRTUAL TABLE messages_trgm USING fts5 (
        content,
        tokenize = 'trigram'
    )
    """,
    """
    CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts (rowid, content, role)
            VALUES (new.id, new.content, new.role);
        INSERT INTO messages_trgm (rowid, content)
            VALUES (new.id, new.content);
    END
    """,
    """
    CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
        DELETE FROM messages_fts WHERE rowid = old.id;
        DELETE FROM messages_trgm WHERE rowid = old.id;
    END
    """,
)


def _apply_v1(conn: sqlite3.Connection) -> None:
    for statement in _DDL_V1:
        conn.execute(statement)


_DDL_V2: tuple[str, ...] = (
    "ALTER TABLE messages ADD COLUMN search_text TEXT",
    "DROP TRIGGER messages_ai",
    "DROP TRIGGER messages_ad",
    "DROP TABLE messages_fts",
    """
    CREATE VIRTUAL TABLE messages_fts USING fts5 (
        search_text, role,
        tokenize = 'unicode61 remove_diacritics 2'
    )
    """,
    """
    CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts (rowid, search_text, role)
            VALUES (new.id, COALESCE(new.search_text, new.content), new.role);
        INSERT INTO messages_trgm (rowid, content)
            VALUES (new.id, new.content);
    END
    """,
    """
    CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
        DELETE FROM messages_fts WHERE rowid = old.id;
        DELETE FROM messages_trgm WHERE rowid = old.id;
    END
    """,
)


def _apply_v2(conn: sqlite3.Connection) -> None:
    from app.normalize import expanded_text

    for statement in _DDL_V2:
        conn.execute(statement)
    pending = conn.execute("SELECT id, content FROM messages ORDER BY id").fetchall()
    for row in pending:
        conn.execute(
            "UPDATE messages SET search_text = ? WHERE id = ?",
            (expanded_text(row["content"]), row["id"]),
        )
    conn.execute(
        "INSERT INTO messages_fts (rowid, search_text, role) "
        "SELECT id, search_text, role FROM messages"
    )


_DDL_V3: tuple[str, ...] = (
    "ALTER TABLE messages ADD COLUMN neighbor_text TEXT",
    "DROP TRIGGER messages_ai",
    "DROP TRIGGER messages_ad",
    "DROP TABLE messages_fts",
    """
    CREATE VIRTUAL TABLE messages_fts USING fts5 (
        search_text, neighbor_text, role,
        tokenize = 'unicode61 remove_diacritics 2'
    )
    """,
    """
    CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts (rowid, search_text, neighbor_text, role)
            VALUES (new.id, COALESCE(new.search_text, new.content),
                    COALESCE(new.neighbor_text, ''), new.role);
        INSERT INTO messages_trgm (rowid, content)
            VALUES (new.id, new.content);
    END
    """,
    """
    CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
        DELETE FROM messages_fts WHERE rowid = old.id;
        DELETE FROM messages_trgm WHERE rowid = old.id;
    END
    """,
)


def _apply_v3(conn: sqlite3.Connection) -> None:

    window = max(0, NEARBY_SPAN)
    for statement in _DDL_V3:
        conn.execute(statement)
    rows = conn.execute(
        "SELECT id, user_id, request_id, position, content FROM messages ORDER BY user_id, request_id, position"
    ).fetchall()
    by_key: dict[tuple[str, str], list[sqlite3.Row]] = {}
    for row in rows:
        by_key.setdefault((str(row["user_id"]), str(row["request_id"])), []).append(row)
    for group in by_key.values():
        for index, row in enumerate(group):
            lo, hi = max(0, index - window), min(len(group), index + window + 1)
            neighbours = [str(group[j]["content"]) for j in range(lo, hi) if j != index]
            conn.execute(
                "UPDATE messages SET neighbor_text = ? WHERE id = ?",
                ("\n".join(neighbours), row["id"]),
            )
    conn.execute(
        "INSERT INTO messages_fts (rowid, search_text, neighbor_text, role) "
        "SELECT id, search_text, COALESCE(neighbor_text, ''), role FROM messages"
    )


def _apply_v4(conn: sqlite3.Connection) -> None:
    from app.normalize import expanded_text

    rows = conn.execute("SELECT id, content FROM messages").fetchall()
    for row in rows:
        conn.execute(
            "UPDATE messages SET search_text = ? WHERE id = ?",
            (expanded_text(row["content"]), row["id"]),
        )
    conn.execute("DELETE FROM messages_fts")
    conn.execute(
        "INSERT INTO messages_fts (rowid, search_text, neighbor_text, role) "
        "SELECT id, search_text, COALESCE(neighbor_text, ''), role FROM messages"
    )


_DDL_V5: tuple[str, ...] = (
    "DROP TRIGGER messages_ai",
    "DROP TRIGGER messages_ad",
    "DROP TABLE messages_fts",
    """
    CREATE VIRTUAL TABLE messages_fts USING fts5 (
        search_text, neighbor_text, role,
        tokenize = 'porter unicode61 remove_diacritics 2'
    )
    """,
    """
    CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts (rowid, search_text, neighbor_text, role)
            VALUES (new.id, COALESCE(new.search_text, new.content),
                    COALESCE(new.neighbor_text, ''), new.role);
        INSERT INTO messages_trgm (rowid, content)
            VALUES (new.id, new.content);
    END
    """,
    """
    CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
        DELETE FROM messages_fts WHERE rowid = old.id;
        DELETE FROM messages_trgm WHERE rowid = old.id;
    END
    """,
)


def _apply_v5(conn: sqlite3.Connection) -> None:
    from app.normalize import expanded_text

    for statement in _DDL_V5:
        conn.execute(statement)
    rows = conn.execute("SELECT id, content FROM messages").fetchall()
    for row in rows:
        conn.execute(
            "UPDATE messages SET search_text = ? WHERE id = ?",
            (expanded_text(row["content"]), row["id"]),
        )
    conn.execute(
        "INSERT INTO messages_fts (rowid, search_text, neighbor_text, role) "
        "SELECT id, search_text, COALESCE(neighbor_text, ''), role FROM messages"
    )


_DDL_V6: tuple[str, ...] = (
    """
    CREATE TABLE messages_v6 (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id    TEXT    NOT NULL,
        position      INTEGER NOT NULL,
        user_id       TEXT    NOT NULL,
        session_id    TEXT    NOT NULL,
        role          TEXT    NOT NULL,
        content       TEXT    NOT NULL,
        timestamp     INTEGER,
        created_at    TEXT    NOT NULL,
        search_text   TEXT,
        neighbor_text TEXT,
        UNIQUE (user_id, request_id, position)
    )
    """,
    """
    INSERT INTO messages_v6
        (id, request_id, position, user_id, session_id, role, content,
         timestamp, created_at, search_text, neighbor_text)
        SELECT id, request_id, position, user_id, session_id, role, content,
               timestamp, created_at, search_text, neighbor_text
        FROM messages
    """,
    "DROP TABLE messages",
    "ALTER TABLE messages_v6 RENAME TO messages",
    "CREATE INDEX messages_user_idx ON messages (user_id, id)",
    "DELETE FROM sqlite_sequence WHERE name = 'messages'",
    """
    INSERT INTO sqlite_sequence (name, seq)
        SELECT 'messages', COALESCE(MAX(id), 0) FROM messages
    """,
    """
    CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts (rowid, search_text, neighbor_text, role)
            VALUES (new.id, COALESCE(new.search_text, new.content),
                    COALESCE(new.neighbor_text, ''), new.role);
        INSERT INTO messages_trgm (rowid, content)
            VALUES (new.id, new.content);
    END
    """,
    """
    CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
        DELETE FROM messages_fts WHERE rowid = old.id;
        DELETE FROM messages_trgm WHERE rowid = old.id;
    END
    """,
)


def _apply_v6(conn: sqlite3.Connection) -> None:
    for statement in _DDL_V6:
        conn.execute(statement)


_DDL_V7: tuple[str, ...] = (
    """
    CREATE TABLE message_vectors (
        message_id INTEGER NOT NULL,
        space      TEXT    NOT NULL,
        model      TEXT    NOT NULL,
        dimension  INTEGER NOT NULL,
        vector     BLOB    NOT NULL,
        created_at TEXT    NOT NULL,
        PRIMARY KEY (message_id, space),
        FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX message_vectors_space_idx ON message_vectors (space)",
)


def _apply_v7(conn: sqlite3.Connection) -> None:
    for statement in _DDL_V7:
        conn.execute(statement)


_DDL_V8: tuple[str, ...] = (
    "ALTER TABLE messages ADD COLUMN source_content TEXT",
    "UPDATE messages SET source_content = content WHERE source_content IS NULL",
)


def _apply_v8(conn: sqlite3.Connection) -> None:
    for statement in _DDL_V8:
        conn.execute(statement)


_DDL_V9: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS messages_session_idx ON messages (user_id, session_id, id)",
)


def _apply_v9(conn: sqlite3.Connection) -> None:
    for statement in _DDL_V9:
        conn.execute(statement)


_DDL_V10: tuple[str, ...] = (
    "ALTER TABLE messages RENAME COLUMN neighbor_text TO nearby_text",
    "DROP TRIGGER IF EXISTS messages_ai",
    "DROP TRIGGER IF EXISTS messages_ad",
    "DROP TABLE IF EXISTS messages_fts",
    """
    CREATE VIRTUAL TABLE messages_fts USING fts5 (
        search_text, nearby_text, role,
        tokenize = 'unicode61 remove_diacritics 2'
    )
    """,
    """
    CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts (rowid, search_text, nearby_text, role)
            VALUES (new.id, COALESCE(new.search_text, new.content),
                    COALESCE(new.nearby_text, ''), new.role);
        INSERT INTO messages_trgm (rowid, content)
            VALUES (new.id, new.content);
    END
    """,
    """
    CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
        DELETE FROM messages_fts WHERE rowid = old.id;
        DELETE FROM messages_trgm WHERE rowid = old.id;
    END
    """,
    "INSERT INTO messages_fts (rowid, search_text, nearby_text, role) "
    "SELECT id, search_text, COALESCE(nearby_text, ''), role FROM messages",
)


def _apply_v10(conn: sqlite3.Connection) -> None:
    for statement in _DDL_V10:
        conn.execute(statement)


_DDL_V11: tuple[str, ...] = (
    """
    CREATE TABLE embedding_cache (
        space      TEXT    NOT NULL,
        content_hash TEXT  NOT NULL,
        dimension  INTEGER NOT NULL,
        vector     BLOB    NOT NULL,
        created_at TEXT    NOT NULL,
        PRIMARY KEY (space, content_hash)
    )
    """,
)


def _apply_v11(conn: sqlite3.Connection) -> None:
    for statement in _DDL_V11:
        conn.execute(statement)


_DDL_V12: tuple[str, ...] = (
    """
    CREATE TABLE rerank_cache (
        model        TEXT    NOT NULL,
        query_hash   TEXT    NOT NULL,
        document_hash TEXT   NOT NULL,
        score        REAL    NOT NULL,
        created_at   TEXT    NOT NULL,
        PRIMARY KEY (model, query_hash, document_hash)
    )
    """,
)


def _apply_v12(conn: sqlite3.Connection) -> None:
    for statement in _DDL_V12:
        conn.execute(statement)


_DDL_V13: tuple[str, ...] = (
    "ALTER TABLE messages ADD COLUMN is_feedback INTEGER NOT NULL DEFAULT 0",
    "UPDATE messages SET is_feedback = 1 WHERE content LIKE 'Streaming online QA feedback:%'",
    "CREATE INDEX messages_feedback_idx ON messages (user_id, is_feedback)",
)


def _apply_v13(conn: sqlite3.Connection) -> None:
    for statement in _DDL_V13:
        conn.execute(statement)


_DDL_V14: tuple[str, ...] = (
    """
    CREATE TABLE profiles (
        user_id       TEXT    PRIMARY KEY,
        content       TEXT    NOT NULL,
        message_count INTEGER NOT NULL,
        updated_at    TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS messages_request_idx ON messages (user_id, request_id)",
)


def _apply_v14(conn: sqlite3.Connection) -> None:
    for statement in _DDL_V14:
        conn.execute(statement)


_DDL_V15: tuple[str, ...] = (
    "ALTER TABLE profiles ADD COLUMN last_message_id INTEGER NOT NULL DEFAULT 0",
)


def _apply_v15(conn: sqlite3.Connection) -> None:
    for statement in _DDL_V15:
        conn.execute(statement)


_MIGRATIONS: tuple[Callable[[sqlite3.Connection], None], ...] = (
    _apply_v1,
    _apply_v2,
    _apply_v3,
    _apply_v4,
    _apply_v5,
    _apply_v6,
    _apply_v7,
    _apply_v8,
    _apply_v9,
    _apply_v10,
    _apply_v11,
    _apply_v12,
    _apply_v13,
    _apply_v14,
    _apply_v15,
)


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._connections_lock = threading.Lock()
        self._savepoint_ids = itertools.count(1)

    def connection(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "connection", None)
        if conn is not None:
            return conn
        conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL").fetchone()
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        with self._connections_lock:
            self._connections.append(conn)
        self._local.connection = conn
        return conn

    def migrate(self) -> int:
        with self.transaction() as conn:
            current = int(conn.execute("PRAGMA user_version").fetchone()[0])
            for version, apply in enumerate(_MIGRATIONS, start=1):
                if version <= current:
                    continue
                apply(conn)
                conn.execute(f"PRAGMA user_version = {version}")
                current = version
                logger.info("sqlite_migrate version=%d path=%s", version, self.path)
        if current != _SCHEMA_VERSION:
            raise RuntimeError(
                f"schema version {current} does not match the code's {_SCHEMA_VERSION}"
            )
        return current

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self.connection()
        if conn.in_transaction:
            name = f"sp_{next(self._savepoint_ids)}"
            conn.execute(f"SAVEPOINT {name}")
            try:
                yield conn
            except BaseException:
                conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
                conn.execute(f"RELEASE SAVEPOINT {name}")
                raise
            else:
                conn.execute(f"RELEASE SAVEPOINT {name}")
            return

        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    def close(self) -> None:
        with self._connections_lock:
            connections, self._connections = self._connections, []
        for conn in connections:
            with contextlib.suppress(sqlite3.Error):
                conn.close()
        self._local = threading.local()
