from __future__ import annotations

import re

_QUESTION_MARK = re.compile(r"[?？]\s*$")
_QUESTION_START = re.compile(r"^\s*(?:question|q|problem|task)\s*[:：]", re.I)
_QUESTION_WORD = re.compile(
    r"^\s*(?:what|who|when|where|which|why|how|is|are|was|were|do|does|did|"
    r"can|could|please|请|问)\b",
    re.I,
)
_BOUNDARY = re.compile(r"[.!?。！？](?:\s|$)|\n")
_WORD_END = re.compile(r"\s+\S*$")
_WORD_START = re.compile(r"^\s*\S+\s+")
_MARKER = "\n...\n"


def _line_score(line: str, index: int, last: int) -> int:
    score = 0
    if _QUESTION_MARK.search(line):
        score += 4
    if _QUESTION_START.search(line):
        score += 3
    if _QUESTION_WORD.search(line):
        score += 2
    if index == last:
        score += 1
    return score


def _line_anchors(query: str, budget: int) -> str:
    lines = [(index, line.strip()) for index, line in enumerate(query.splitlines()) if line.strip()]
    if not lines:
        return ""
    last = len(query.splitlines()) - 1
    scored = [
        (index, line, _line_score(line, index, last))
        for index, line in lines
    ]
    picked = [item for item in scored if item[2] >= 2]
    if not picked:
        picked = [scored[0], scored[-1]] if len(scored) > 1 else [scored[0]]
    picked.sort(key=lambda item: (-item[2], item[0]))
    anchors: list[str] = []
    used = 0
    for _, line, score in picked:
        if used >= budget:
            break
        if len(line) > budget - used:
            if score >= 4:
                line = _tail(line, budget - used)
            else:
                line = _head(line, budget - used)
        if not line or line in anchors:
            continue
        anchors.append(line)
        used += len(line) + 1
    return "\n".join(anchors)


def _head(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    raw = text[:budget]
    boundary = None
    for match in _BOUNDARY.finditer(raw):
        if match.end() >= budget * 0.6:
            boundary = match.end()
    if boundary is not None:
        return raw[:boundary].rstrip()
    space = _WORD_END.search(raw)
    if space and space.start() >= budget * 0.6:
        return raw[: space.start()].rstrip()
    return raw.rstrip()


def _tail(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    raw = text[-budget:]
    boundary = None
    for match in _BOUNDARY.finditer(raw):
        if match.start() <= budget * 0.4:
            boundary = match.start()
            break
    if boundary is not None:
        return raw[boundary:].lstrip()
    space = _WORD_START.search(raw)
    if space and space.start() <= budget * 0.4:
        return raw[space.start():].lstrip()
    return raw.lstrip()


def clip_query(query: str, limit: int, anchor_chars: int = 2_000) -> str:
    if limit <= 0 or len(query) <= limit:
        return query
    anchor_budget = min(max(200, limit // 3), anchor_chars)
    anchors = _line_anchors(query, anchor_budget)
    remaining = limit - len(anchors) - 2 * len(_MARKER)
    if remaining < 200:
        return anchors[:limit] if anchors else _head(query, limit)
    head_budget = int(remaining * 0.4)
    tail_budget = remaining - head_budget
    parts = [
        anchors,
        _head(query, head_budget),
        _tail(query, tail_budget),
    ]
    result = _MARKER.join(part for part in parts if part)
    if len(result) > limit:
        result = result[:limit]
    return result
