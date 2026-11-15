"""Generate ELM answers from the top-k records of a partly migrated index.

Writes one candidate file per index state in the format of scripts/evaluate_answers.py,
so the course judge scores them unchanged. Identical prompts are generated once
(cached by prompt hash), so states that retrieve the same records share answers.

    python rag/extensions/migration_answers.py --split dev --fraction 0.1 \
        --out-dir rag/results/migration_answers_dev
    python rag/scripts/evaluate_answers.py --split dev \
        --candidates rag/results/migration_answers_dev/naive@10.jsonl \
        --out-dir rag/results/migration_answers_dev/judge_naive@10
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

import numpy as np

from migration import CACHE_DIR, DATA_DIR, Setup, fit_maps, ranked_ids, score
from src.context_builder import SimpleContextBuilder
from src.data import CourseData


def states(s, fraction, seed_pairs):
    """Top-k scores for the target index and for naive / map-docs at `fraction` migrated (seed 0 order)."""
    n = len(s.ids)
    order = np.random.default_rng(0).permutation(n)
    moved = np.zeros(n, dtype=bool)
    moved[order[:int(round(fraction * n))]] = True
    maps, _ = fit_maps(s, order[:max(seed_pairs, int(moved.sum()))])
    pct = int(round(fraction * 100))
    yield "target", s.A @ s.Q.T
    for strategy in ("naive", "map-docs"):
        yield f"{strategy}@{pct}", score(strategy, s, moved, maps)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["practice", "dev"], default="dev")
    ap.add_argument("--fraction", type=float, default=0.1)
    ap.add_argument("--seed-pairs", type=int, default=2000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="first N queries only (smoke test)")
    ap.add_argument("--workers", type=int, default=3, help="concurrent ELM requests")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    s = Setup(args.split, CACHE_DIR / f"{args.split}_transcripts.jsonl", None)
    with (CACHE_DIR / f"{args.split}_transcripts.jsonl").open(encoding="utf-8") as f:
        transcripts = {r["query_id"]: r["transcript"] for r in map(json.loads, f)}
    data = CourseData(DATA_DIR)
    builder = SimpleContextBuilder()
    qids = s.qids[:args.limit] if args.limit else s.qids

    from src.generator import ELMGenerator

    generator = ELMGenerator()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.out_dir / "generation_cache.jsonl"
    cache = {}
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            cache = {r["prompt_sha"]: r for r in map(json.loads, f)}

    plans = []  # (state, qid, record ids, prompt, sha)
    for state, scores in states(s, args.fraction, args.seed_pairs):
        for qid, ranked in zip(s.qids, ranked_ids(scores, s.ids, args.k)):
            if qid not in qids:
                continue
            prompt = builder.build(transcripts[qid], ranked, data)["prompt"]
            plans.append((state, qid, ranked, prompt, hashlib.sha256(prompt.encode()).hexdigest()))

    todo = {sha: prompt for _, _, _, prompt, sha in plans if sha not in cache}
    print(f"{len(plans)} answers needed, {len(set(p[4] for p in plans))} distinct prompts, {len(todo)} to generate")

    def run(item):
        sha, prompt = item
        out = generator.generate(prompt)
        return {"prompt_sha": sha, "answer": out["answer"], "latency_ms": out["latency_ms"], "prompt_chars": len(prompt)}

    with ThreadPoolExecutor(max_workers=args.workers) as pool, cache_path.open("a", encoding="utf-8") as f:
        for i, row in enumerate(pool.map(run, todo.items()), start=1):
            cache[row["prompt_sha"]] = row
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] {row['latency_ms']:.0f} ms")

    by_state = {}
    for state, qid, ranked, prompt, sha in plans:
        g = cache[sha]
        by_state.setdefault(state, []).append({
            "query_id": qid, "split": args.split, "transcript": transcripts[qid], "answer": g["answer"],
            "retrieved_record_ids": ranked, "timing": {"generation_ms": g["latency_ms"]},
            "prompt_chars": g["prompt_chars"]})
    for state, rows in by_state.items():
        path = args.out_dir / f"{state}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        hits = [len(set(r["retrieved_record_ids"]) & set(s.qrels[r["query_id"]])) > 0 for r in rows]
        print(f"{state:14s} {len(rows)} answers -> {path.name}   queries with a relevant record in top-{args.k}: {np.mean(hits):.0%}")


if __name__ == "__main__":
    main()
