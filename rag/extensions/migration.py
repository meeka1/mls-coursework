"""Serving retrieval while the corpus migrates from enc_b to enc_a.

Scenario: the index was built with a legacy encoder (enc_b) and is being
re-encoded with enc_a, record by record. Queries keep arriving during the
migration, so at any moment a fraction of records has an enc_a vector and the
rest only an enc_b vector. enc_b has no query encoder, so queries reach the
legacy part through a linear map fitted on records that exist in both spaces.

Strategies for scoring a partly migrated corpus:
  no-map        one index, raw enc_b vectors for legacy records scored with the enc_a query (no map)
  naive         one index, enc_a scores for migrated records, mapped-query enc_b scores for the rest
  z-calibrated  as naive, but each part's scores are standardised per query
  dual-read     search both parts separately and merge the two rankings (reciprocal rank fusion)
  map-docs      map legacy record vectors into enc_a space once (B -> A), one index, one query

    python rag/extensions/migration.py sweep --split practice --device mps \
        --output rag/results/migration_practice.json
    python rag/extensions/migration.py pairs --split practice --device mps
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
STRATEGIES = ["no-map", "naive","z-calibrated", "dual-read", "map-docs"]
METRICS = ["Recall@10", "Recall@100", "MRR@10", "NDCG@10"]
RRF_K = 60
RRF_DEPTH = 1000


def normalise(m):
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def fit_map(src, dst):
    """Least-squares linear map W with src @ W ~ dst."""
    return np.linalg.lstsq(src.astype(np.float64), dst.astype(np.float64), rcond=None)[0].astype(np.float32)


class Setup:
    """Aligned enc_a / enc_b corpus vectors, enc_a query vectors and qrels for one split."""

    def __init__(self, split, transcripts, device):
        data = CourseData(DATA_DIR)
        emb = DATA_DIR / "embeddings"
        self.ids = json.loads((emb / "corpus_enc_a_ids.json").read_text())
        self.A = np.load(emb / "corpus_enc_a.npy").astype(np.float32)
        ids_b = json.loads((emb / "corpus_enc_b_ids.json").read_text())
        row_b = {rid: i for i, rid in enumerate(ids_b)}
        self.B = np.load(emb / "corpus_enc_b.npy").astype(np.float32)[[row_b[r] for r in self.ids]]

        with transcripts.open(encoding="utf-8") as f:
            text = {r["query_id"]: r["transcript"] for r in map(json.loads, f)}
        self.qids = [it["query_id"] for it in data.audio_items(split) if it["query_id"] in text]
        self.qrels = data.load_qrels(split)

        cache = CACHE_DIR / f"{split}_queries_enc_a.npy"
        if cache.exists():
            self.Q = np.load(cache)
        else:
            from src.encoder import LocalEncoder

            enc = LocalEncoder(ROOT / "models" / "enc_a", device=device)
            self.Q = enc.model.encode([text[q] for q in self.qids], convert_to_numpy=True,
                                      normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
            np.save(cache, self.Q)


def ranked_ids(scores, ids, k=100):
    """Top-k record IDs for each column of a [records, queries] score matrix."""
    top = np.argpartition(-scores, k - 1, axis=0)[:k]
    out = []
    for j in range(scores.shape[1]):
        col = top[:, j]
        col = col[np.argsort(-scores[col, j])]
        out.append([ids[i] for i in col])
    return out


def rrf(parts, n_records):
    """Merge per-part rankings: each part is (record indices, score matrix restricted to them)."""
    fused = np.zeros((n_records, parts[0][1].shape[1]), dtype=np.float32)
    weights = 1.0 / (RRF_K + np.arange(1, RRF_DEPTH + 1, dtype=np.float32))
    for idx, scores in parts:
        if not len(idx):
            continue
        depth = min(RRF_DEPTH, len(idx))
        top = np.argpartition(-scores, depth - 1, axis=0)[:depth]
        for j in range(scores.shape[1]):
            order = top[np.argsort(-scores[top[:, j], j]), j]
            fused[idx[order], j] += weights[:depth]
    return fused


def score(strategy, s, moved, maps):
    """Score matrix [records, queries] for one strategy on a partly migrated corpus."""
    Q = s.Q.T
    if strategy == "no-map":
        return np.where(moved[:, None], s.A @ Q, s.B @ Q)
    if strategy == "map-docs":
        legacy = normalise(s.B @ maps["b2a"]) @ Q
        return np.where(moved[:, None], s.A @ Q, legacy)
    Qb = normalise(s.Q @ maps["a2b"]).T
    sa, sb = s.A @ Q, s.B @ Qb
    if strategy == "naive":
        return np.where(moved[:, None], sa, sb)
    if strategy == "z-calibrated":
        if moved.any():
            sa = (sa - sa[moved].mean(0)) / sa[moved].std(0)
        if (~moved).any():
            sb = (sb - sb[~moved].mean(0)) / sb[~moved].std(0)
        return np.where(moved[:, None], sa, sb)
    if strategy == "dual-read":
        new, old = np.where(moved)[0], np.where(~moved)[0]
        return rrf([(new, sa[new]), (old, sb[old])], len(s.ids))
    raise ValueError(strategy)


def evaluate(s, scores):
    rows = []
    for qid, ranked in zip(s.qids, ranked_ids(scores, s.ids)):
        rel = s.qrels[qid]
        rows.append({"Recall@10": recall_at_k(ranked, rel, 10), "Recall@100": recall_at_k(ranked, rel, 100),
                     "MRR@10": mrr_at_k(ranked, rel, 10), "NDCG@10": ndcg_at_k(ranked, rel, 10)})
    return {m: float(np.mean([r[m] for r in rows])) for m in METRICS}


def fit_maps(s, pair_idx):
    t0 = time.perf_counter()
    maps = {"a2b": fit_map(s.A[pair_idx], s.B[pair_idx]), "b2a": fit_map(s.B[pair_idx], s.A[pair_idx])}
    return maps, (time.perf_counter() - t0) * 1000.0


def serving_cost(s, strategy, moved, maps, repeats=200):
    """Per-query search latency and index memory for one strategy, one query at a time."""
    q = s.Q[0]
    if strategy in ("map-docs", "no-map"):
        legacy = normalise(s.B @ maps["b2a"]) if strategy == "map-docs" else s.B
        index = np.where(moved[:, None], s.A, legacy)
        t0 = time.perf_counter()
        for _ in range(repeats):
            sc = index @ q
            np.argpartition(-sc, 99)[:100]
        return (time.perf_counter() - t0) / repeats * 1000.0, index.nbytes / 1e6
    new, old = s.A[moved], s.B[~moved]
    t0 = time.perf_counter()
    for _ in range(repeats):
        qb_ = normalise(q @ maps["a2b"])
        sa, sb = new @ q, old @ qb_
        if strategy == "dual-read":
            for part in (sa, sb):
                if len(part) > RRF_DEPTH:
                    np.argpartition(-part, RRF_DEPTH - 1)[:RRF_DEPTH]
        else:
            np.argpartition(-np.concatenate([sa, sb]), 99)[:100]
    mb = (new.nbytes + old.nbytes + maps["a2b"].nbytes) / 1e6
    return (time.perf_counter() - t0) / repeats * 1000.0, mb


def sweep(args):
    s = Setup(args.split, args.transcripts, args.device)
    n = len(s.ids)
    results = {st: {f: [] for f in args.fractions} for st in STRATEGIES}
    fit_ms = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        for f in args.fractions:
            moved = np.zeros(n, dtype=bool)
            moved[order[:int(round(f * n))]] = True
            # Pairs for the maps: a seed sample encoded in both spaces up front, plus every migrated record.
            pair_idx = order[:max(args.seed_pairs, int(moved.sum()))]
            maps, ms = fit_maps(s, pair_idx)
            fit_ms.append(ms)
            for st in STRATEGIES:
                results[st][f].append(evaluate(s, score(st, s, moved, maps)))
        print(f"seed {seed} done")

    summary = {st: {str(f): {m: (float(np.mean([r[m] for r in runs])), float(np.std([r[m] for r in runs])))
                             for m in METRICS} for f, runs in per_f.items()} for st, per_f in results.items()}

    moved = np.zeros(n, dtype=bool)
    moved[np.random.default_rng(0).permutation(n)[: n // 2]] = True
    maps, _ = fit_maps(s, np.where(moved)[0])
    t0 = time.perf_counter()
    normalise(s.B @ maps["b2a"])
    map_docs_ms = (time.perf_counter() - t0) * 1000.0
    cost = {st: dict(zip(("search_ms", "index_mb"), serving_cost(s, st, moved, maps))) for st in STRATEGIES}

    print(f"\nsplit={args.split} queries={len(s.qids)} seeds={args.seeds} seed_pairs={args.seed_pairs}")
    for metric in ("NDCG@10", "Recall@10"):
        print(f"\n{metric} (mean ± std over seeds) by fraction of records re-encoded with enc_a")
        print(f"{'strategy':14s}" + "".join(f"{int(f * 100):>14d}%" for f in args.fractions))
        for st in STRATEGIES:
            print(f"{st:14s}" + "".join(f"{summary[st][str(f)][metric][0]:>8.3f}±{summary[st][str(f)][metric][1]:.3f}"
                                         for f in args.fractions))
    print(f"\nCost at 50% migrated (one query at a time, this machine):")
    for st in STRATEGIES:
        print(f"  {st:14s} search {cost[st]['search_ms']:.3f} ms   index {cost[st]['index_mb']:.1f} MB")
    print(f"  fit both maps: median {np.median(fit_ms):.1f} ms; map all legacy records B->A: {map_docs_ms:.1f} ms")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"split": args.split, "queries": len(s.qids), "seeds": args.seeds,
                                           "seed_pairs": args.seed_pairs, "fractions": args.fractions,
                                           "summary": summary, "cost_at_50": cost,
                                           "fit_ms_median": float(np.median(fit_ms)),
                                           "map_legacy_ms": map_docs_ms}, indent=2))
        print(f"saved {args.output}")


def pairs(args):
    """How many dual-encoded records the maps need: quality of a fully legacy index vs pair count."""
    s = Setup(args.split, args.transcripts, args.device)
    n = len(s.ids)
    moved = np.zeros(n, dtype=bool)  # nothing migrated: the whole index is legacy
    target = evaluate(s, s.A @ s.Q.T)["NDCG@10"]
    print(f"split={args.split}  target (all enc_a) NDCG@10 = {target:.3f}")
    print(f"{'pairs':>7}" + "".join(f"{st:>16s}" for st in ("naive", "map-docs")))
    for n_pairs in args.counts:
        vals = {st: [] for st in ("naive", "map-docs")}
        for seed in range(args.seeds):
            idx = np.random.default_rng(seed).permutation(n)[:n_pairs]
            maps, _ = fit_maps(s, idx)
            for st in vals:
                vals[st].append(evaluate(s, score(st, s, moved, maps))["NDCG@10"])
        print(f"{n_pairs:>7}" + "".join(f"{np.mean(v):>10.3f}±{np.std(v):.3f}" for v in vals.values()))


def diagnose(args):
    """Why mixing spaces fails: score scales per space and who fills the top 10 at each fraction."""
    s = Setup(args.split, args.transcripts, args.device)
    n = len(s.ids)
    order = np.random.default_rng(0).permutation(n)
    maps, _ = fit_maps(s, order[:args.seed_pairs])
    sa = s.A @ s.Q.T
    sb = s.B @ normalise(s.Q @ maps["a2b"]).T
    sm = normalise(s.B @ maps["b2a"]) @ s.Q.T
    row = {rid: i for i, rid in enumerate(s.ids)}
    rel = [(row[r], j) for j, qid in enumerate(s.qids) for r in s.qrels[qid]]
    ri, rj = np.array(rel).T

    def describe(name, m):
        top10 = -np.sort(-m, axis=0)[9]
        print(f"  {name:28s} all {m.mean():.3f}±{m.std():.3f}   10th best {top10.mean():.3f}   "
              f"relevant {m[ri, rj].mean():.3f}   relevant - 10th best {np.mean(m[ri, rj] - top10[rj]):+.3f}")

    print(f"split={args.split} queries={len(s.qids)}  (cosine scores, averaged over queries)")
    describe("enc_a (migrated part)", sa)
    describe("enc_b + mapped query (legacy)", sb)
    describe("legacy mapped B->A", sm)
    gap = np.mean(-np.sort(-sb, axis=0)[0] - (-np.sort(-sa, axis=0)[0]))
    print(f"  per-query top-1 score, enc_b minus enc_a: {gap:+.3f}")

    print("\nShare of the top 10 taken by migrated records, and share of relevant hits found among them")
    print(f"{'fraction':>9}" + "".join(f"{st:>24s}" for st in STRATEGIES))
    for f in args.fractions:
        moved = np.zeros(n, dtype=bool)
        moved[order[:int(round(f * n))]] = True
        maps_f, _ = fit_maps(s, order[:max(args.seed_pairs, int(moved.sum()))])  # as in sweep
        cells = []
        for st in STRATEGIES:
            sc = score(st, s, moved, maps_f)
            top = np.argpartition(-sc, 9, axis=0)[:10]
            share = moved[top].mean()
            rel_hits = [(i, j) for j in range(top.shape[1]) for i in top[:, j] if s.ids[i] in s.qrels[s.qids[j]]]
            found_new = np.mean([moved[i] for i, _ in rel_hits]) if rel_hits else float("nan")
            cells.append(f"{share:6.0%} top, {found_new:4.0%} hits")
        print(f"{int(f * 100):>8d}%" + "".join(f"{c:>24s}" for c in cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, func in (("sweep", sweep), ("pairs", pairs), ("diagnose", diagnose)):
        p = sub.add_parser(name)
        p.add_argument("--split", choices=["practice", "dev"], default="practice")
        p.add_argument("--transcripts", type=Path, default=None)
        p.add_argument("--device", default=None)
        p.add_argument("--seeds", type=int, default=5)
        p.set_defaults(func=func)
        if name in ("sweep", "diagnose"):
            p.add_argument("--fractions", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
            p.add_argument("--seed-pairs", type=int, default=2000,
                           help="records encoded in both spaces before migration starts")
            p.add_argument("--output", type=Path, default=None)
        else:
            p.add_argument("--counts", type=int, nargs="+", default=[100, 250, 500, 1000, 2000, 5000, 24000])
    args = ap.parse_args()
    if args.transcripts is None:
        args.transcripts = CACHE_DIR / f"{args.split}_transcripts.jsonl"
    args.func(args)


if __name__ == "__main__":
    main()
