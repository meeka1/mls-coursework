"""Selective corrective retrieval: rewrite the query with ELM only when first-stage retrieval looks unsure.

Confidence signal (free): margin = enc_a score of result #1 - score of result #10.
Correction (one ELM call): either
  rewrite  fix speech-recognition errors and restate the transcript as a clear, specific question
  hyde     write a short hypothetical forum answer and search with that (HyDE)
The corrected text is encoded with enc_a and searched again. Ways to use the second search:
  replace  rank by the corrected query alone
  fuse     reciprocal rank fusion of the original and corrected rankings
  mix      search with normalise(original vector + corrected vector)

ELM outputs are cached per query, so threshold sweeps and variants cost nothing extra.

    # smoke test: 5 queries, inspect the rewrites
    python rag/extensions/corrective.py --split dev --mode rewrite --limit 5 --show
    # full split
    python rag/extensions/corrective.py --split dev --mode rewrite --output rag/results/corrective_rewrite_dev.json
    python rag/extensions/corrective.py --split dev --mode hyde --output rag/results/corrective_hyde_dev.json
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time

import numpy as np

from selective_rerank import CACHE_DIR, load, metrics, query_vectors

PROMPTS = {
    "rewrite": (
        "The question below was transcribed from speech by an automatic speech recogniser, so it may "
        "contain misheard words. It will be used to search a collection of personal finance forum posts "
        "(investing, tax, banking, credit, mortgages, retirement, business).\n"
        "Rewrite it as one clear, specific question. Fix words that were probably misheard, spell out "
        "the financial terms it refers to, and keep its meaning. Do not answer it. "
        "Reply with the rewritten question only.\n\nQuestion: {q}"
    ),
    "hyde": (
        "The question below was transcribed from speech, so it may contain misheard words. Write a short "
        "answer (60 to 100 words) in the style of a reply on a personal finance Q&A forum, using the "
        "specific financial terms such a reply would contain. Reply with the answer text only.\n\n"
        "Question: {q}"
    ),
}
RRF_K = 60


class Rewriter:
    def __init__(self):
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv(ROOT / ".env")
        key = os.environ.get("ELM_API_KEY")
        if not key:
            raise SystemExit("Missing ELM_API_KEY: add it to rag/.env (see rag/.env.example).")
        self.model = os.environ.get("ELM_MODEL", "utter-project/EuroLLM-22B-Instruct-2512")
        self.client = OpenAI(api_key=key, base_url=os.environ.get("ELM_BASE_URL", "https://elm.edina.ac.uk/api/v1"),
                             timeout=120)

    def __call__(self, prompt, retries=3):
        for attempt in range(retries):
            try:
                t0 = time.perf_counter()
                r = self.client.chat.completions.create(model=self.model, temperature=0.0, max_tokens=200,
                                                        messages=[{"role": "user", "content": prompt}])
                text = r.choices[0].message.content.strip().strip('"')
                return text, (time.perf_counter() - t0) * 1000.0
            except Exception as exc:  # network / rate limit
                if attempt == retries - 1:
                    raise
                print(f"  retry after error: {exc}")
                time.sleep(2 * (attempt + 1))


def corrections(mode, split, qids, tr, workers):
    """ELM output per query, cached in .cache/corrective_{mode}_{split}.jsonl."""
    path = CACHE_DIR / f"corrective_{mode}_{split}.jsonl"
    done = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            done = {r["query_id"]: r for r in map(json.loads, f)}
    todo = [q for q in qids if q not in done]
    if todo:
        rw = Rewriter()
        print(f"{len(todo)} ELM calls ({mode}, model {rw.model})")

        def run(q):
            text, ms = rw(PROMPTS[mode].format(q=tr[q]))
            return {"query_id": q, "transcript": tr[q], "text": text, "latency_ms": ms, "model": rw.model}

        CACHE_DIR.mkdir(exist_ok=True)
        with ThreadPoolExecutor(max_workers=workers) as pool, path.open("a", encoding="utf-8") as f:
            for i, row in enumerate(pool.map(run, todo), start=1):
                done[row["query_id"]] = row
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                if i % 10 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}", flush=True)
    return done


def rrf(*score_mats):
    out = np.zeros_like(score_mats[0])
    for S in score_mats:
        ranks = np.argsort(np.argsort(-S, axis=0), axis=0) + 1
        out += 1.0 / (RRF_K + ranks)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["practice", "dev"], default="dev")
    ap.add_argument("--transcripts", type=Path, default=None)
    ap.add_argument("--mode", choices=list(PROMPTS), default="rewrite")
    ap.add_argument("--limit", type=int, default=None, help="first N queries only (smoke test)")
    ap.add_argument("--workers", type=int, default=3, help="concurrent ELM requests")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--show", action="store_true", help="print each transcript and its correction")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    ids, A, text, tr, qids, qrels = load(args.split, args.transcripts or CACHE_DIR / f"{args.split}_transcripts.jsonl")
    Q = query_vectors(args.split, qids, tr, args.device)
    if args.limit:
        qids, Q = qids[:args.limit], Q[:args.limit]
    corr = corrections(args.mode, args.split, qids, tr, args.workers)

    from sentence_transformers import SentenceTransformer
    import torch

    enc = SentenceTransformer(str(ROOT / "models" / "enc_a"), device=args.device,
                              model_kwargs={"dtype": torch.float32})
    t0 = time.perf_counter()
    Qc = enc.encode([corr[q]["text"] for q in qids], normalize_embeddings=True, convert_to_numpy=True,
                    show_progress_bar=False).astype(np.float32)
    encode_ms = (time.perf_counter() - t0) * 1000.0 / len(qids)

    S0, Sc = A @ Q.T, A @ Qc.T
    mixed = Q + Qc
    mixed /= np.linalg.norm(mixed, axis=1, keepdims=True)
    variants = {"replace": Sc, "fuse": rrf(S0, Sc), "mix": A @ mixed.T}
    srt = -np.sort(-S0, axis=0)
    margin = srt[0] - srt[9]
    order = np.argsort(margin)
    elm_ms = np.array([corr[q]["latency_ms"] for q in qids])

    def ranked(S, j):
        return [ids[i] for i in np.argsort(-S[:, j])[:100]]

    base = [metrics(ranked(S0, j), qrels[q]) for j, q in enumerate(qids)]
    per = {v: [metrics(ranked(S, j), qrels[q]) for j, q in enumerate(qids)] for v, S in variants.items()}
    keys = list(base[0])

    if args.show:
        for j, q in enumerate(qids):
            print(f"\n[{q}] margin {margin[j]:.3f}  gold@5 {base[j]['gold@5']:.0f} -> replace {per['replace'][j]['gold@5']:.0f}"
                  f" / fuse {per['fuse'][j]['gold@5']:.0f}")
            print(f"  asr : {tr[q]}")
            print(f"  {args.mode:4s}: {corr[q]['text']}")

    print(f"\nsplit={args.split} queries={len(qids)} mode={args.mode}")
    print(f"ELM call: median {np.median(elm_ms):.0f} ms, p95 {np.percentile(elm_ms, 95):.0f} ms; "
          f"re-encode {encode_ms:.0f} ms/query; second search is one more matrix-vector product")
    rows = []
    print(f"\n{'variant':8s} {'corrected':>9} {'margin <':>9}" + "".join(f"{k:>10}" for k in keys) + f"{'ELM calls':>10}{'+ms/q':>8}")
    shares = (0.0, 0.25, 0.5, 0.75, 1.0)
    for v in variants:
        for share in shares:
            n = int(round(share * len(qids)))
            chosen = np.zeros(len(qids), dtype=bool)
            chosen[order[:n]] = True
            thr = float(margin[order[n]]) if n < len(qids) else float("inf")
            m = {k: float(np.mean([(per[v] if chosen[j] else base)[j][k] for j in range(len(qids))])) for k in keys}
            ms = float(elm_ms[chosen].sum() / len(qids))
            rows.append({"variant": v, "share_corrected": share, "margin_threshold": thr, **m,
                         "elm_calls": int(n), "added_ms_per_query": ms})
            if v != list(variants)[0] and share == 0.0:
                continue
            print(f"{v:8s} {share:>9.0%} {thr:>9.3f}" + "".join(f"{m[k]:>10.3f}" for k in keys) + f"{n:>10d}{ms:>8.0f}")

    print("\nMargin vs random choice of which queries to correct (fuse, gold@5, mean of 200 draws):")
    rng = np.random.default_rng(0)
    for share in (0.25, 0.5):
        n = int(round(share * len(qids)))
        vals = []
        for _ in range(200):
            chosen = np.zeros(len(qids), dtype=bool)
            chosen[rng.choice(len(qids), n, replace=False)] = True
            vals.append(np.mean([(per["fuse"] if chosen[j] else base)[j]["gold@5"] for j in range(len(qids))]))
        print(f"  {share:.0%}: random {np.mean(vals):.3f}")

    print("\nEffect by margin quarter (gold@5, original -> replace / fuse):")
    for name, idx in zip(("lowest", "2nd", "3rd", "highest"), np.array_split(order, 4)):
        b = np.mean([base[j]["gold@5"] for j in idx])
        r = np.mean([per["replace"][j]["gold@5"] for j in idx])
        f_ = np.mean([per["fuse"][j]["gold@5"] for j in idx])
        print(f"  {name:8s} {b:.2f} -> {r:.2f} / {f_:.2f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"split": args.split, "mode": args.mode, "queries": len(qids),
                                           "elm_ms_median": float(np.median(elm_ms)), "rows": rows}, indent=2))
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
