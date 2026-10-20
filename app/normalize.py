from __future__ import annotations

import re

MONTHS: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

MONTH_NAMES: dict[int, str] = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

NUMBER_WORDS: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

_NUMBER_NAMES: dict[int, str] = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
}

MULTIPLIERS: dict[str, int] = {"twice": 2, "thrice": 3, "dozen": 12}

_NON_COUNT_BEFORE = frozenset({"no", "any", "every", "each", "this", "that", "which", "the"})
_NON_COUNT_AFTER = frozenset({"of", "day", "another", "other"})

_TAIL_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "but",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "so",
        "than",
        "that",
        "the",
        "then",
        "to",
        "was",
        "were",
        "when",
        "with",
    }
)

_MONTH_PATTERN = "|".join(sorted(MONTHS, key=len, reverse=True))
_DAY = r"(\d{1,2})(?:st|nd|rd|th)?"

_RE_MDY = re.compile(
    rf"\b({_MONTH_PATTERN})\.?\s+{_DAY},?\s+((?:19|20)\d{{2}})\b",
    re.IGNORECASE,
)
_RE_DMY = re.compile(
    rf"\b{_DAY}\s+(?:of\s+)?({_MONTH_PATTERN})\.?,?\s+((?:19|20)\d{{2}})\b",
    re.IGNORECASE,
)
_RE_ISO = re.compile(r"(?<!\d)((?:19|20)\d{2})-(\d{2})-(\d{2})(?!\d)")
_RE_MY = re.compile(rf"\b({_MONTH_PATTERN})\.?,?\s+((?:19|20)\d{{2}})\b", re.IGNORECASE)
_RE_ISO_SLASH = re.compile(r"(?<!\d)((?:19|20)\d{2})/(\d{1,2})/(\d{1,2})(?!\d)")
_RE_TIME = re.compile(r"(?<!\d)\d{1,2}:\d{2}(?::\d{2})?")

_DATE_PATTERNS = (_RE_MDY, _RE_DMY, _RE_ISO, _RE_ISO_SLASH, _RE_MY, _RE_TIME)

_RE_WORD = re.compile(r"[A-Za-z][A-Za-z']*|\d+")

MAX_EXPANSIONS = 64


def _iso(year: int, month: int, day: int) -> str | None:
    if not (1 <= month <= 12) or not (1 <= day <= 31):
        return None
    if month == 2:
        if day > 29:
            return None
    elif month in (4, 6, 9, 11) and day > 30:
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def date_expansions(text: str) -> list[str]:
    found: list[str] = []

    def add(values: list[str]) -> None:
        for value in values:
            if value not in found:
                found.append(value)

    for match in _RE_MDY.finditer(text):
        month = MONTHS[match.group(1).lower()]
        day, year = int(match.group(2)), int(match.group(3))
        iso = _iso(year, month, day)
        if iso:
            add([iso, iso[:7], f"{MONTH_NAMES[month]} {day}, {year}"])

    for match in _RE_DMY.finditer(text):
        month = MONTHS[match.group(2).lower()]
        day, year = int(match.group(1)), int(match.group(3))
        iso = _iso(year, month, day)
        if iso:
            add([iso, iso[:7], f"{day} {MONTH_NAMES[month]} {year}"])

    for match in _RE_ISO.finditer(text):
        year, month, day = (int(match.group(i)) for i in (1, 2, 3))
        iso = _iso(year, month, day)
        if iso:
            add([iso, iso[:7], f"{MONTH_NAMES[month]} {day}, {year}"])

    for match in _RE_ISO_SLASH.finditer(text):
        year, month, day = (int(match.group(i)) for i in (1, 2, 3))
        iso = _iso(year, month, day)
        if iso:
            add([iso, iso[:7]])

    for match in _RE_MY.finditer(text):
        month = MONTHS[match.group(1).lower()]
        add([f"{match.group(2)}-{month:02d}"])

    return found[:MAX_EXPANSIONS]


def mask_dates(text: str) -> str:
    chars = list(text)
    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            for index in range(match.start(), match.end()):
                chars[index] = " "
    return "".join(chars)


def _is_count(text: str, start: int, end: int) -> bool:
    before = _RE_WORD.findall(text[:start].lower())[-1:]
    after_match = _RE_WORD.search(text, end)
    after = after_match.group(0).lower() if after_match else ""
    word = text[start:end].lower()
    if word == "one":
        if before and before[0] in _NON_COUNT_BEFORE:
            return False
        if after in _NON_COUNT_AFTER or after.endswith("'s"):
            return False
    return not (before and before[0] == "no")


def number_expansions(text: str) -> list[str]:
    body = mask_dates(text)
    lowered = text.lower()
    matches = list(_RE_WORD.finditer(body))
    found: list[str] = []
    for index, match in enumerate(matches):
        raw = match.group(0)
        word = raw.lower()
        tail = ""
        if index + 1 < len(matches):
            candidate = matches[index + 1].group(0)
            if candidate.lower() not in _TAIL_STOP and not candidate.isdigit():
                tail = f" {candidate}"

        if word in NUMBER_WORDS:
            if not _is_count(body, match.start(), match.end()):
                continue
            phrase = f"{NUMBER_WORDS[word]}{tail}"
        elif word in MULTIPLIERS:
            phrase = f"{MULTIPLIERS[word]} times{tail}"
        elif raw.isdigit():
            value = int(raw)
            if value > 12 or not tail:
                continue
            phrase = f"{_NUMBER_NAMES[value]}{tail}"
        else:
            continue

        if phrase.lower() not in lowered and phrase not in found:
            found.append(phrase)
    return found[:MAX_EXPANSIONS]


_CJK_RUN_RE = re.compile(r"[\u3400-\u9fff]+")


def cjk_index_tokens(text: str) -> list[str]:
    from app.text import cjk_words

    seen: set[str] = set()
    tokens: list[str] = []
    for run in _CJK_RUN_RE.findall(text):
        candidates = [
            *cjk_words(run),
            *(run[i] for i in range(len(run))),
            *(run[i : i + 2] for i in range(len(run) - 1)),
        ]
        for token in candidates:
            if token not in seen:
                seen.add(token)
                tokens.append(token)
    return tokens


def expanded_text(content: str) -> str:
    extra = expand(content)
    combined = " ".join([*extra.split(), *cjk_index_tokens(content)])
    return f"{content}\n{combined}" if combined else content


def expand(text: str) -> str:
    if not text:
        return ""
    lowered = text.lower()
    seen: set[str] = set()
    unique: list[str] = []
    for part in date_expansions(text) + number_expansions(text):
        key = part.lower()
        if key in seen or key in lowered:
            continue
        seen.add(key)
        unique.append(part)
    return " ".join(unique)
