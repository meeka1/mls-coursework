"""Sanity checks for the map-docs result: is it real, or just enc_a records found by an enc_a query?

Three checks, dev split by default, map fitted on `--seed-pairs` random records:

  1. no-map      enc_a query against raw enc_b records. Near zero means the two spaces really differ,
                 so any later success must come from the map.
  2. held-out    every record is legacy (0% migrated) and mapped B -> A. Recall is computed only over
                 relevant records that were NOT used to fit the map, next to the baseline on the same
                 records, so a record never benefits from having been a training pair.
  3. who-hits    at each fraction, the share of relevant top-10 hits that are mapped legacy records,
                 next to the legacy share of the corpus. If mapped records were dead weight, hits
                 would come almost only from migrated records.

    python rag/extensions/migration_checks.py --split dev
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse

import numpy as np

from migration import CACHE_DIR, Setup, fit_maps, normalise, score


def recall_restricted(scores, s, keep, k=10):
    """Mean Recall@k counting only relevant records whose row is in `keep`; queries with none are skipped."""
    top = np.argpartition(-scores, k - 1, axis=0)[:k]
    row = {rid: i for i, rid in enumerate(s.ids)}
    vals = []
    for j, qid in enumerate(s.qids):
        rel = {row[r] for r in s.qrels[qid] if r in row and keep[row[r]]}
        if rel:
            vals.append(len(rel & set(top[:, j].tolist())) / len(rel))
    return float(np.mean(vals)), len(vals)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["practice", "dev"], default="dev")
    ap.add_argument("--transcripts", type=Path, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed-pairs", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    s = Setup(args.split, args.transcripts or CACHE_DIR / f"{args.split}_transcripts.jsonl", args.device)
    n = len(s.ids)
    everyone = np.ones(n, dtype=bool)
    base = s.A @ s.Q.T

    print(f"split={args.split} queries={len(s.qids)} records={n} seed_pairs={args.seed_pairs}\n")
    print("1. no-map: enc_a query vs raw enc_b records")
    print(f"   baseline (enc_a docs)   Recall@10 {recall_restricted(base, s, everyone)[0]:.3f}")
    print(f"   raw enc_b docs, no map  Recall@10 {recall_restricted(s.B @ s.Q.T, s, everyone)[0]:.3f}\n")

    print("2. held-out: 0% migrated, all records mapped B->A, relevant records outside the fit set only")
    mapped_r, base_r, nq = [], [], []
    for seed in range(args.seeds):
        order = np.random.default_rng(seed).permutation(n)
        maps, _ = fit_maps(s, order[:args.seed_pairs])
        held = np.ones(n, dtype=bool)
        held[order[:args.seed_pairs]] = False
        mapped = normalise(s.B @ maps["b2a"]) @ s.Q.T
        m, q = recall_restricted(mapped, s, held)
        mapped_r.append(m); nq.append(q)
        base_r.append(recall_restricted(base, s, held)[0])
    print(f"   baseline   Recall@10 {np.mean(base_r):.3f}")
    print(f"   map-docs   Recall@10 {np.mean(mapped_r):.3f} ± {np.std(mapped_r):.3f}  "
          f"({np.mean(mapped_r) / np.mean(base_r):.0%} of baseline, {np.mean(nq):.0f} queries, {args.seeds} seeds)\n")

    print("3. who-hits: share of relevant top-10 hits that are mapped legacy records (seed 0)")
    order = np.random.default_rng(0).permutation(n)
    row = {rid: i for i, rid in enumerate(s.ids)}
    print(f"   {'migrated':>8}  {'legacy share of corpus':>22}  {'legacy share of hits':>20}  {'Recall@10':>9}")
    for f in (0.1, 0.25, 0.5, 0.75, 0.9):
        moved = np.zeros(n, dtype=bool)
        moved[order[:int(round(f * n))]] = True
        maps, _ = fit_maps(s, order[:max(args.seed_pairs, int(moved.sum()))])
        sc = score("map-docs", s, moved, maps)
        top = np.argpartition(-sc, 9, axis=0)[:10]
        hits = [i for j, qid in enumerate(s.qids) for i in top[:, j] if s.ids[i] in s.qrels[qid]]
        legacy_hits = np.mean([not moved[i] for i in hits]) if hits else float("nan")
        print(f"   {f:>8.0%}  {1 - f:>22.0%}  {legacy_hits:>20.0%}  {recall_restricted(sc, s, everyone)[0]:>9.3f}")


if __name__ == "__main__":
    main()
