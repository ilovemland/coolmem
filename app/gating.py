from __future__ import annotations

import re

from app.policy import CORE_QUERY_MAX_CHARS, CORE_QUERY_MIN_CHARS

OPTION_LINE = re.compile(r"(?m)(?:^|\s)([A-F])[.)]\s+\S")
CJK = re.compile(r"[\u3400-\u9fff]")
BLOCK_BREAK = re.compile(r"\n\s*\n")


def _blocks(query: str) -> list[str]:
    return [block.strip() for block in BLOCK_BREAK.split(query) if block.strip()]


def _usable(text: str) -> bool:
    minimum = 4 if CJK.search(text) else CORE_QUERY_MIN_CHARS
    return minimum <= len(text) <= CORE_QUERY_MAX_CHARS


def query_segments(query: str, max_segments: int = 3) -> list[str]:
    if len(OPTION_LINE.findall(query)) >= 2:
        return [query]
    blocks = _blocks(query)
    if len(blocks) < 2:
        return [query]
    segments = [query]
    for block in (blocks[0], blocks[-1]):
        if _usable(block) and block not in segments:
            segments.append(block)
    return segments[:max_segments]


def core_query(query: str) -> str | None:
    if len(OPTION_LINE.findall(query)) >= 2:
        return None
    blocks = _blocks(query)
    if len(blocks) < 2:
        return None
    return blocks[0] if _usable(blocks[0]) else None
