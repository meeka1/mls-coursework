"""Selective reranking: rerank only the queries whose first-stage retrieval looks unsure.

Confidence signal (free): margin = score of result #1 - score of result #10 under enc_a.
Action (costly): a cross-encoder rescoring the top-N candidates of the query.

For a range of thresholds, queries with margin below the threshold are reranked and the
rest keep the enc_a order. The output shows retrieval quality against the share of
queries reranked and the reranker time spent.

    python rag/extensions/selective_rerank.py --split dev --device cpu
    python rag/extensions/selective_rerank.py --split dev --model BAAI/bge-reranker-base

Cross-encoder scores are cached per (model, split, depth), so threshold sweeps are free.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
import time

import numpy as np

from src.data import CourseData
from src.metrics import mrr_at_k, ndcg_at_k, recall_at_k

DATA_DIR = ROOT / "data" / "rag_corpus"
CACHE_DIR = ROOT / ".cache"


def load(split, transcripts):
    data = CourseData(DATA_DIR)
    emb = DATA_DIR / "embeddings"
    ids = json.loads((emb / "corpus_enc_a_ids.json").read_text())
    A = np.load(emb / "corpus_enc_a.npy").astype(np.float32)
    text = {}
    with (DATA_DIR / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text[r["record_id"]] = r["text"]
    with transcripts.open(encoding="utf-8") as f:
        tr = {r["query_id"]: r["transcript"] for r in map(json.loads, f)}
    qids = [it["query_id"] for it in data.audio_items(split) if it["query_id"] in tr]
    return ids, A, text, tr, qids, data.load_qrels(split)


def query_vectors(split, qids, tr, device):
    cache = CACHE_DIR / f"{split}_queries_enc_a.npy"
    if cache.exists():
        return np.load(cache)
    from src.encoder import LocalEncoder

    enc = LocalEncoder(ROOT / "models" / "enc_a", device=device)
    Q = enc.model.encode([tr[q] for q in qids], convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=False).astype(np.float32)
    CACHE_DIR.mkdir(exist_ok=True)
    np.save(cache, Q)
    return Q


def metrics(ranked, rel):
    return {
        "gold@5": float(bool(set(rel) & set(ranked[:5]))),
        "Recall@5": recall_at_k(ranked, rel, 5),
        "Recall@10": recall_at_k(ranked, rel, 10),
        "MRR@10": mrr_at_k(ranked, rel, 10),
        "NDCG@10": ndcg_at_k(ranked, rel, 10),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["practice", "dev"], default="dev")
    ap.add_argument("--transcripts", type=Path, default=None)
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--depth", type=int, default=50, help="candidates reranked per query")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    ids, A, text, tr, qids, qrels = load(args.split, args.transcripts or CACHE_DIR / f"{args.split}_transcripts.jsonl")
    Q = query_vectors(args.split, qids, tr, args.device)
    S = A @ Q.T
    cand = np.argsort(-S, axis=0)[:args.depth].T          # [queries, depth] in enc_a order
    srt = -np.sort(-S, axis=0)
    margin = srt[0] - srt[9]

    tag = args.model.replace("/", "__")
    cache = CACHE_DIR / f"rerank_{tag}_{args.split}_d{args.depth}.npz"
    if cache.exists():
        z = np.load(cache)
        ce, ce_ms = z["scores"], z["ms"]
    else:
        import torch
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(args.model, device=args.device, max_length=512,
                             model_kwargs={"dtype": torch.float32})  # float16 on CPU is very slow
        model.predict([(tr[qids[0]], text[ids[cand[0, 0]]])])  # warm-up, excluded from timing
        ce, ce_ms = np.zeros(cand.shape, np.float32), np.zeros(len(qids))
        for j, q in enumerate(qids):
            t0 = time.perf_counter()
            ce[j] = model.predict([(tr[q], text[ids[i]]) for i in cand[j]], batch_size=args.batch_size,
                                  show_progress_bar=False)
            ce_ms[j] = (time.perf_counter() - t0) * 1000.0
            if (j + 1) % 5 == 0:
                print(f"reranked {j + 1}/{len(qids)}  ({np.mean(ce_ms[:j + 1]):.0f} ms/query)", flush=True)
        CACHE_DIR.mkdir(exist_ok=True)
        np.savez(cache, scores=ce, ms=ce_ms)

    base_rank = [[ids[i] for i in cand[j]] for j in range(len(qids))]
    rr_rank = [[ids[i] for i in cand[j][np.argsort(-ce[j])]] for j in range(len(qids))]
    base_m = [metrics(base_rank[j], qrels[q]) for j, q in enumerate(qids)]
    rr_m = [metrics(rr_rank[j], qrels[q]) for j, q in enumerate(qids)]
    keys = list(base_m[0])

    print(f"\nsplit={args.split} queries={len(qids)} model={args.model} depth={args.depth} device={args.device}")
    print(f"reranker time per reranked query: median {np.median(ce_ms):.0f} ms, p95 {np.percentile(ce_ms, 95):.0f} ms")
    print(f"\n{'reranked':>10} {'margin <':>9}" + "".join(f"{k:>10}" for k in keys) + f"{'rerank ms/q':>13}")
    rows = []
    order = np.argsort(margin)
    for share in (0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
        n = int(round(share * len(qids)))
        chosen = np.zeros(len(qids), dtype=bool)
        chosen[order[:n]] = True
        thr = float(margin[order[n]]) if n < len(qids) else float("inf")
        m = {k: float(np.mean([(rr_m if chosen[j] else base_m)[j][k] for j in range(len(qids))])) for k in keys}
        ms = float(ce_ms[chosen].sum() / len(qids))
        rows.append({"share_reranked": share, "margin_threshold": thr, **m, "rerank_ms_per_query": ms})
        print(f"{share:>10.0%} {thr:>9.3f}" + "".join(f"{m[k]:>10.3f}" for k in keys) + f"{ms:>13.0f}")

    # Same budget, random choice of queries: is the margin better than chance?
    print("\nSame share reranked, but queries picked at random (mean of 200 draws):")
    rng = np.random.default_rng(0)
    for share in (0.25, 0.5):
        n = int(round(share * len(qids)))
        vals = []
        for _ in range(200):
            chosen = np.zeros(len(qids), dtype=bool)
            chosen[rng.choice(len(qids), n, replace=False)] = True
            vals.append(np.mean([(rr_m if chosen[j] else base_m)[j]["gold@5"] for j in range(len(qids))]))
        print(f"  {share:.0%} random: gold@5 {np.mean(vals):.3f}")

    # Where does reranking help or hurt, by margin quarter?
    print("\nEffect of reranking by margin quarter (gold@5, enc_a -> reranked):")
    for name, idx in zip(("lowest", "2nd", "3rd", "highest"), np.array_split(order, 4)):
        b = np.mean([base_m[j]["gold@5"] for j in idx])
        r = np.mean([rr_m[j]["gold@5"] for j in idx])
        print(f"  {name:8s} quarter  {b:.2f} -> {r:.2f}  ({r - b:+.2f})")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"split": args.split, "model": args.model, "depth": args.depth,
                                           "device": args.device, "queries": len(qids),
                                           "rerank_ms_median": float(np.median(ce_ms)), "rows": rows}, indent=2))
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
