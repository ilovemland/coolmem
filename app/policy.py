from __future__ import annotations

import os
from dataclasses import dataclass

MAX_MESSAGES_PER_ADD = 500
MAX_CONTENT_CHARS = 100_000
MAX_QUERY_CHARS = 300_000
MAX_IDENTIFIER_CHARS = 256
MAX_TOP_K = 100

SEARCH_QUERY_CHARS = 12_000
CHANNEL_QUOTAS = (("dense", 0.80), ("lexical", 0.15), ("substring", 0.05))
SEARCH_CANDIDATES = 200
SUBSTRING_DEPTH_RATIO = 0.5

RECENCY_LAMBDA = float(os.environ.get("COOLMEM_RECENCY_LAMBDA", "0.0"))
RERANK_DEPTH = 100
RERANK_DOC_CHARS = 2_400

NEARBY_SPAN = 3
NEARBY_WEIGHT = 0.5

CORE_QUERY_MIN_CHARS = 8
CORE_QUERY_MAX_CHARS = 400


@dataclass(frozen=True, slots=True)
class Policy:
    max_top_k: int = MAX_TOP_K
    rerank_enabled: bool = True
    segment_search: bool = True


DEFAULT_POLICY = Policy()
