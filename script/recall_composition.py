from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.embedding import EmbeddingClient, load_embedding_config
from app.gating import HYBRID, LITERAL, profile_query
from app.policy import Policy
from app.storage.database import Database
from app.storage.repository import MessageRepository


def candidate_limit(retriever, top_k: int, candidates: int) -> int:
    if retriever.depth_ratio is None:
        return candidates
    return max(1, int(top_k * retriever.depth_ratio))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    policy = Policy(lane_gating=True)
    database = Database(args.db)
    config = load_embedding_config()
    embedding = EmbeddingClient(config) if config.enabled else None
    repository = MessageRepository(database, policy, embedding=embedding)
    by_name = {retriever.name: retriever for retriever in repository._retrievers}
    candidates = max(args.top_k, policy.search_candidates)
    lanes = collections.Counter()
    total = collections.Counter()
    per_lane: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    rows = []
    for line in pathlib.Path(args.queries).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        user_id = str(item["user_id"])
        query = str(item["query"])
        profile = profile_query(query, policy.gate)
        lanes[profile.lane] += 1
        if profile.lane == LITERAL:
            active = [name for name in ("lexical", "substring") if name in by_name]
        elif profile.lane == HYBRID:
            active = list(by_name)
        else:
            active = ["dense"] if "dense" in by_name else [n for n in ("lexical", "substring") if n in by_name]
        candidate_ids: dict[str, set[int]] = {}
        for name in active:
            retriever = by_name[name]
            hits = retriever.retrieve(
                user_id=user_id,
                query=query,
                limit=candidate_limit(retriever, args.top_k, candidates),
            )
            candidate_ids[name] = {hit.message.id for hit in hits}
        dense_ids = candidate_ids.get("dense", set())
        lexical_ids = candidate_ids.get("lexical", set()) | candidate_ids.get("substring", set())
        hits = repository.search(user_id=user_id, query=query, limit=args.top_k)
        row = collections.Counter()
        for hit in hits:
            mid = hit.message.id
            dense = mid in dense_ids
            lexical = mid in lexical_ids
            kind = "both" if dense and lexical else ("dense" if dense else "lex")
            row[kind] += 1
        total.update(row)
        per_lane[profile.lane].update(row)
        rows.append(
            {
                "user_id": user_id,
                "query": query,
                "lane": profile.lane,
                "returned": len(hits),
                "dense": row["dense"],
                "both": row["both"],
                "lex": row["lex"],
            }
        )
    returned = sum(total.values())
    summary = {
        "queries": sum(lanes.values()),
        "returned": returned,
        "lanes": {key: round(value / max(1, sum(lanes.values())), 4) for key, value in lanes.items()},
        "overall": {key: round(total[key] / max(1, returned), 4) for key in ("dense", "both", "lex")},
        "per_lane": {
            lane: {
                "queries": lanes[lane],
                "returned": sum(counter.values()),
                "composition": {
                    key: round(counter[key] / max(1, sum(counter.values())), 4)
                    for key in ("dense", "both", "lex")
                },
            }
            for lane, counter in per_lane.items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
