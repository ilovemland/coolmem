from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

WORD = re.compile(r"[A-Za-z]+|[\u3400-\u9fff]")
QUOTED = re.compile(r"[\"'“”‘’](.{2,80}?)[\"'“”‘’]", re.S)
DATE = re.compile(
    r"\b(?:19|20)\d{2}\b"
    r"|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b",
    re.I,
)
EXACT_DATE = re.compile(
    r"\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b"
    r"|\b\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2}\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+"
    r"\d{1,2}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{2}\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s+"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?,?\s+"
    r"(?:19|20)\d{2}\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?,?\s+"
    r"(?:19|20)\d{2}\b",
    re.I,
)
CODE = re.compile(
    r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"|(?:https?://|/[\w.-]+/)[^\s]+"
    r"|\b(?=[A-Za-z0-9_-]*[A-Za-z])(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{4,}\b"
    r"|\b[A-Z]{2,}[_-][A-Z0-9_-]{2,}\b"
    r"|\b\w+\.(?:py|json|ya?ml|toml|sh|sql|md|txt|csv|log)\b"
    r"|\bsk-[A-Za-z0-9]{8,}\b"
)
NAME = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
NUMBER = re.compile(r"(?<![\w.])\d{2,}(?![\w.])")
CJK = re.compile(r"[\u3400-\u9fff]")
OPTION_LINE = re.compile(r"(?m)(?:^|\s)([A-F])[.)]\s+\S")
INSTRUCTION = re.compile(
    r"\b(?:use only|return all|return one|return the|output|please provide|respond with|"
    r"format|json array|do not explain|no explanation|enclose|following|answer the)\b"
    r"|(?:仅使用|返回|输出|格式|不要解释|请提供|以下)",
    re.I,
)
QSTART = re.compile(
    r"^\s*(?:what|who|when|where|which|why|how|is|are|was|were|did|does|do|"
    r"can|could|would|should|will|has|have|had|tell|list|give|find|show)\b",
    re.I,
)
AUX = re.compile(
    r"\b(?:is|are|was|were|did|does|do|can|could|would|should|has|have|had)\b",
    re.I,
)
RANGE = re.compile(
    r"\b(?:as of|between|after|before|since|until)\b"
    r"|\bby\s+(?:the\s+)?(?:\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
    re.I,
)

DENSE: Final = "dense"
HYBRID: Final = "hybrid"
LITERAL: Final = "literal"


@dataclass(frozen=True, slots=True)
class GateConfig:
    literal_max_words: int = 16
    quoted_literal_max_words: int = 14
    keyword_max_words: int = 6
    hybrid_max_words: int = 40
    named_hybrid_max_words: int = 6
    closed_lookup_max_words: int = 8
    core_query_min_chars: int = 8
    core_query_max_chars: int = 400


DEFAULT_GATE = GateConfig()


@dataclass(frozen=True, slots=True)
class QueryProfile:
    lane: str
    words: int
    signals: tuple[str, ...]


def core_query(query: str, config: GateConfig = DEFAULT_GATE) -> str | None:
    if len(OPTION_LINE.findall(query)) >= 2:
        return None
    blocks = [block.strip() for block in re.split(r"\n\s*\n", query) if block.strip()]
    if len(blocks) < 2:
        return None
    first = blocks[0]
    minimum = 4 if CJK.search(first) else config.core_query_min_chars
    if len(first) < minimum or len(first) > config.core_query_max_chars:
        return None
    if not INSTRUCTION.search("\n\n".join(blocks[1:])):
        return None
    return first


def profile_query(query: str, config: GateConfig = DEFAULT_GATE) -> QueryProfile:
    text = query.strip()
    words = len(WORD.findall(text))
    quoted = bool(QUOTED.search(text))
    code = bool(CODE.search(text))
    date = bool(DATE.search(text))
    exact_date = bool(EXACT_DATE.search(text))
    number = bool(NUMBER.search(text))
    named = bool(NAME.search(text))
    cjk = bool(CJK.search(text))
    question = bool(QSTART.search(text) or "?" in text or "？" in text or AUX.search(text))
    ranged = bool(RANGE.search(text))
    signals = tuple(
        name
        for name, active in (
            ("quoted", quoted),
            ("code", code),
            ("date", date),
            ("number", number),
            ("named", named),
            ("cjk", cjk),
            ("question", question),
            ("range", ranged),
        )
        if active
    )
    literal = (code or (quoted and words <= config.quoted_literal_max_words)) and not question
    literal = literal and words <= config.literal_max_words
    keyword = words <= config.keyword_max_words and not question
    keyword = keyword and (date or number or cjk or named)
    closed = exact_date and not named and question
    closed = closed and words <= config.closed_lookup_max_words
    if literal or keyword or closed:
        return QueryProfile(LITERAL, words, signals)
    hybrid = (quoted or code) and words <= config.hybrid_max_words
    hybrid = hybrid or (date and ranged)
    hybrid = hybrid or (named and words <= config.named_hybrid_max_words)
    if hybrid:
        return QueryProfile(HYBRID, words, signals)
    return QueryProfile(DENSE, words, signals)
