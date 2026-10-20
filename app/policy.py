from __future__ import annotations

from dataclasses import dataclass, field

from app.gating import DEFAULT_GATE, GateConfig


@dataclass(frozen=True, slots=True)
class Policy:
    max_messages_per_add: int = 500
    max_request_id_chars: int = 256
    max_user_id_chars: int = 256
    max_session_id_chars: int = 256
    max_role_chars: int = 32
    max_content_chars: int = 100_000

    max_query_chars: int = 300_000
    search_query_chars: int = 12_000
    max_top_k: int = 100
    search_candidates: int = 200
    substring_depth_ratio: float = 0.5

    neighbor_window: int = 3
    neighbor_weight: float = 0.5

    coverage_weight: float = 2.0
    rrf_k: int = 15
    bm25_tiebreak_weight: float = 0.02

    lane_gating: bool = False
    instruction_decomposition: bool = True
    context_window_turns: int = 0
    neighbor_message_chars: int = 200
    neighbor_skip_chars: int = 640
    neighbor_top_n: int = 0
    rerank_enabled: bool = True
    rerank_depth: int = 200
    rerank_blend: float = 1.0
    rerank_doc_chars: int = 2400
    gate: GateConfig = field(default_factory=lambda: DEFAULT_GATE)
    dense_channel_weight: float = 1.0
    hybrid_lexical_weight: float = 8.0
    hybrid_substring_weight: float = 8.0
    literal_lexical_weight: float = 1.0
    literal_substring_weight: float = 1.0
    fallback_lexical_weight: float = 1.0
    fallback_substring_weight: float = 1.0


DEFAULT_POLICY = Policy()
