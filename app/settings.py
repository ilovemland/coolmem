from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

ENV_PREFIX = "COOLMEM_"


def _text(env: Mapping[str, str], name: str, default: str) -> str:
    return env.get(f"{ENV_PREFIX}{name}", default).strip()


def _integer(env: Mapping[str, str], name: str, default: int) -> int:
    raw = _text(env, name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{ENV_PREFIX}{name} must be an integer, got {raw!r}") from exc


def _read_token(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"cannot read token file {path}: {exc}") from None
    except UnicodeDecodeError:
        raise ValueError(f"token file {path} is not valid UTF-8") from None


@dataclass(frozen=True, slots=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 8005
    data_dir: Path = Path("data")
    db_path: Path | None = None
    max_body_bytes: int = 4_000_000
    bearer_token: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if len(self.bearer_token) < 16 or any(
            char.isspace() for char in self.bearer_token
        ):
            raise ValueError(
                "COOLMEM_BEARER_TOKEN must be at least 16 characters and contain no whitespace"
            )

    @property
    def resolved_db_path(self) -> Path:
        return self.db_path or (self.data_dir / "coolmem.sqlite3")

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        token_file: Path | None = None,
    ) -> Settings:
        source = os.environ if env is None else env
        token = _text(source, "BEARER_TOKEN", "")
        if not token and (env is None or token_file is not None):
            candidates = (Path("env"), Path(".env")) if token_file is None else (token_file,)
            for candidate in candidates:
                if candidate.is_file():
                    token = _read_token(candidate)
                    break
        return cls(
            host=_text(source, "HOST", "0.0.0.0"),
            port=_integer(source, "PORT", 8005),
            bearer_token=token,
        )
