from __future__ import annotations

import contextlib
import io
import re
import threading
from datetime import UTC, datetime

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_FRAGMENT_RE = re.compile(r"[^\s\"'()]+", re.UNICODE)

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "me",
        "might",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "ours",
        "she",
        "should",
        "so",
        "some",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
    }
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def ms_to_iso(timestamp_ms: int) -> str:
    return (
        datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


_CJK_RUN_RE = re.compile(r"[\u3400-\u9fff]+")


def cjk_bigrams(text: str) -> list[str]:
    grams: list[str] = []
    for run in _CJK_RUN_RE.findall(text):
        if len(run) == 1:
            grams.append(run)
        else:
            grams.extend(run[i : i + 2] for i in range(len(run) - 1))
    return grams


def cjk_trigrams(text: str) -> list[str]:
    grams: list[str] = []
    for run in _CJK_RUN_RE.findall(text):
        grams.extend(run[i : i + 3] for i in range(len(run) - 2))
    return grams


_CUTTER = None
_CUTTER_LOCK = threading.Lock()


def _cutter():
    global _CUTTER
    if _CUTTER is None:
        with _CUTTER_LOCK:
            if _CUTTER is None:
                with contextlib.redirect_stdout(io.StringIO()):
                    from cutword import Cutter

                    _CUTTER = Cutter()
    return _CUTTER


def cjk_words(text: str) -> list[str]:
    words: list[str] = []
    for run in _CJK_RUN_RE.findall(text):
        for token in _cutter().cutword(run):
            token = token.strip()
            if token and _WORD_RE.fullmatch(token):
                words.append(token)
    return words


def terms(text: str, *, max_terms: int = 32) -> list[str]:
    words = [word for word in _WORD_RE.findall(text) if word.lower() not in _STOPWORDS]
    return words[:max_terms]


def fts_query(text: str, *, max_terms: int = 32, operator: str = "AND") -> str:
    parts: list[str] = []
    for word in terms(text, max_terms=max_terms):
        if _CJK_RUN_RE.search(word):
            clauses: list[str] = []
            for piece in [*cjk_words(word), *cjk_bigrams(word)]:
                if piece not in clauses:
                    clauses.append(piece)
            clauses += [piece for piece in _CJK_RUN_RE.split(word) if piece]
            quoted = [f'"{clause}"' for clause in clauses]
            clause = f'( {" OR ".join(quoted)} )' if len(quoted) > 1 else quoted[0]
            parts.append(clause)
        else:
            parts.append(f'"{word}"')
    return f" {operator} ".join(parts)


def trigram_query(text: str, *, min_chars: int = 3, max_fragments: int = 16) -> str:
    fragments: list[str] = []
    for fragment in _FRAGMENT_RE.findall(text):
        if fragment.lower() in _STOPWORDS:
            continue
        pieces: list[str] = []
        if _CJK_RUN_RE.search(fragment):
            pieces = [g for g in cjk_trigrams(fragment) if len(g) >= min_chars]
            pieces += [p for p in _CJK_RUN_RE.split(fragment) if len(p) >= min_chars]
            if not pieces and len(fragment) >= min_chars:
                pieces = [fragment]
        elif len(fragment) >= min_chars:
            pieces = [fragment]
        for piece in pieces:
            if piece not in fragments:
                fragments.append(piece)
    return " OR ".join(f'"{fragment}"' for fragment in fragments[:max_fragments])
