"""Run the course answer judge, skipping answers it cannot score, then compare states pairwise.

The judge prompt, rubric, model and cache are those of src/evaluation/answer_judge.py,
unchanged; the cache file is shared with scripts/evaluate_answers.py runs in the same
--out-dir, so nothing already judged is requested again. When the judge model keeps
returning malformed JSON for an answer, that query is recorded as unscorable instead
of stopping the run. `compare` scores every state on the queries judged in all of them.

    python rag/extensions/judge_robust.py judge --split dev \
        --candidates rag/results/migration_answers_dev/map-docs@10.jsonl \
        --out-dir rag/results/migration_answers_dev/judge_map-docs@10
    python rag/extensions/judge_robust.py compare --dir rag/results/migration_answers_dev \
        --state target --state naive@10 --state map-docs@10
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
import os

import numpy as np

from src.evaluation.answer_judge import (
    Judge, append_cache, build_public_gold, cache_key, load_cache, load_jsonl,
)

DIMS = ("correctness", "grounding", "completeness")


def judge(args):
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    model = os.getenv("ELM_JUDGE_MODEL") or os.getenv("ELM_MODEL", "utter-project/EuroLLM-22B-Instruct-2512")
    gold = build_public_gold(ROOT / "data/rag_corpus", args.split)
    j = Judge(model, retries=args.api_retries)
    cache_path = args.out_dir / "judge_cache.jsonl"
    cache = load_cache(cache_path)
    scored, failed = [], []
    for row in load_jsonl(args.candidates):
        qid = row["query_id"]
        docs = gold[qid]["gold_documents"]
        key = cache_key(model + "@" + str(j.client.base_url), row["transcript"], row["answer"], docs, 0)
        if key in cache:
            result = cache[key]["result"]
        else:
            try:
                result = j.evaluate(row["transcript"], row["answer"], docs)
            except RuntimeError as exc:
                failed.append({"query_id": qid, "error": str(exc)})
                print(f"{qid}: unscorable ({exc})")
                continue
            entry = {"cache_key": key, "query_id": qid, "result": result}
            append_cache(cache_path, entry)
            cache[key] = entry
        score = {d: result[d] for d in DIMS}
        score["normalised_answer_score"] = sum(score.values()) / 12
        scored.append({"query_id": qid, **score})
        print(f"{qid}: {score['normalised_answer_score']:.3f}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "robust_per_query.jsonl").open("w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")
    (args.out_dir / "unscorable.json").write_text(json.dumps(failed, indent=2))
    print(f"scored {len(scored)}, unscorable {len(failed)}, judge model {model}")


def compare(args):
    per = {}
    for st in args.state:
        with (args.dir / f"judge_{st}" / "robust_per_query.jsonl").open() as f:
            per[st] = {r["query_id"]: r for r in map(json.loads, f)}
    common = sorted(set.intersection(*(set(v) for v in per.values())))
    print(f"queries scored in every state: {len(common)}")
    print(f"{'state':14s}" + "".join(f"{d:>14s}" for d in DIMS) + f"{'score (0-1)':>14s}")
    for st in args.state:
        print(f"{st:14s}" + "".join(f"{np.mean([per[st][q][d] for q in common]):>14.3f}"
                                    for d in DIMS + ("normalised_answer_score",)))
    ref = args.state[0]
    rng = np.random.default_rng(0)
    for st in args.state[1:]:
        d = np.array([per[st][q]["normalised_answer_score"] - per[ref][q]["normalised_answer_score"] for q in common])
        lo, hi = np.percentile([rng.choice(d, d.size).mean() for _ in range(5000)], [2.5, 97.5])
        print(f"{st} - {ref}: {d.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  better/worse/same "
              f"{(d > 0).sum()}/{(d < 0).sum()}/{(d == 0).sum()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--split", choices=["practice", "dev"], required=True)
    j.add_argument("--candidates", type=Path, required=True)
    j.add_argument("--out-dir", type=Path, required=True)
    j.add_argument("--api-retries", type=int, default=3)
    j.set_defaults(func=judge)
    c = sub.add_parser("compare")
    c.add_argument("--dir", type=Path, required=True)
    c.add_argument("--state", action="append", required=True, help="first state is the reference")
    c.set_defaults(func=compare)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
