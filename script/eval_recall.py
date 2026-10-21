from __future__ import annotations

import argparse
import json
import math
import string
import sys
import time
from dataclasses import fields, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.container import Container
from app.policy import Policy
from app.settings import Settings

EDGE_CHARS = string.whitespace + string.punctuation + "。，、；：？！“”‘’（）【】《》…—"
MIN_OVERLAP = 60
_HASH_BASE = 1_000_003
_HASH_MOD = (1 << 61) - 1


def canonical(text: str) -> str:
    parts = str(text).casefold().split()
    return " ".join(parts)


def _window_fingerprints(text: str, width: int) -> set[int]:
    if len(text) < width:
        return set()
    codes = [ord(char) for char in text]
    head = codes[:width]
    rolling = 0
    power = 1
    for code in head:
        rolling = (rolling * _HASH_BASE + code) % _HASH_MOD
        power = power * _HASH_BASE % _HASH_MOD
    fingerprints = {rolling}
    for position in range(width, len(codes)):
        rolling = (
            rolling * _HASH_BASE + codes[position] - codes[position - width] * power
        ) % _HASH_MOD
        fingerprints.add(rolling)
    return fingerprints


def gold_covered(gold: str, evidence: str) -> bool:
    target = canonical(gold).strip(EDGE_CHARS)
    if not target:
        return False
    source = canonical(evidence)
    if target in source:
        return True
    if len(target) <= MIN_OVERLAP:
        return False
    shared = _window_fingerprints(target, MIN_OVERLAP) & _window_fingerprints(
        source, MIN_OVERLAP
    )
    return bool(shared)


def hit_rank(golds: list[str], evidence: list[str]) -> int:
    if not golds:
        return -1
    for index, text in enumerate(evidence):
        if any(gold_covered(gold, text) for gold in golds):
            return index
    return -1


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def parse_policy(raw: str) -> dict:
    if not raw.strip():
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise SystemExit("--policy must be a JSON object")
    allowed = {item.name for item in fields(Policy)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise SystemExit(f"unknown policy fields: {','.join(unknown)}")
    return payload


def aggregate(rows: list[dict], top_k: int) -> dict:
    total = len(rows)
    hits = {key: sum(1 for row in rows if 0 <= row["rank"] < key) for key in (1, 3, 10, top_k)}
    chars = sum(row["chars"] for row in rows) / max(1, total)
    return {
        "n": total,
        "hits": hits,
        "rates": {key: value / max(1, total) for key, value in hits.items()},
        "chars_mean": round(chars, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--policy", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    policy = replace(Policy(), **parse_policy(args.policy))
    settings = Settings(
        bearer_token="x" * 32,
        data_dir=Path("/tmp"),
        db_path=Path(args.db),
    )
    container = Container.build(settings, policy)
    cases = load_cases(Path(args.cases))
    if args.limit:
        cases = cases[: args.limit]

    rows = []
    started = time.time()
    for index, case in enumerate(cases, start=1):
        results = container.search.search(
            user_id=case["user_id"], query=case["query"], top_k=args.top_k
        )
        evidence = [canonical(item.message.content) for item in results]
        rank = hit_rank(case.get("gold") or [], evidence)
        rows.append(
            {
                "scope": case.get("scope", ""),
                "query": case["query"],
                "rank": rank,
                "chars": sum(len(text) for text in evidence),
            }
        )
        if index % 100 == 0:
            print(f"  {index}/{len(cases)} {time.time() - started:.0f}s", flush=True)

    summary = aggregate(rows, args.top_k)
    summary["elapsed_s"] = round(time.time() - started, 1)
    summary["label"] = args.label
    print(
        f"[{args.label or 'run'}] n={summary['n']} "
        + " ".join(
            f"@{key}={value}({summary['rates'][key] * 100:.1f}%)"
            for key, value in summary["hits"].items()
        )
        + f" chars={summary['chars_mean']:.0f} {summary['elapsed_s']:.0f}s",
        flush=True,
    )
    if args.out:
        Path(args.out).write_text(
            json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    container.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
