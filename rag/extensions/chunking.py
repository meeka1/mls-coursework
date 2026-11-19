"""Chunk-level dense retrieval: does splitting long records help enc_a?

enc_a mean-pools every token of a record into one vector, so in a long record
the passage that answers a query is averaged with everything around it. This
module indexes overlapping token windows ("chunks") instead and scores each
record by its best chunk (max-sim), optionally combined with the whole-record
vector.

    # 1. build a chunk index (once per configuration)
    python rag/extensions/chunking.py build --size 128 --overlap 32 --device mps

    # 2. compare retrieval quality and cost against the whole-record baseline
    python rag/extensions/chunking.py evaluate --split practice \
        --transcripts rag/.cache/practice_transcripts.jsonl \
        --index c128_o32 --index c256_o64 --device mps
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
INDEX_DIR = ROOT / ".cache" / "chunks"
MODEL_DIR = ROOT / "models" / "enc_a"
METRICS = ["Recall@5", "Recall@10", "Recall@20", "Recall@100", "MRR@10", "NDCG@10"]


def load_model(device):
    from sentence_transformers import SentenceTransformer

    kwargs = {"device": device} if device else {}
    return SentenceTransformer(str(MODEL_DIR), local_files_only=True, **kwargs)


def split_record(tokenizer, text, size, overlap):
    """Character spans of windows of `size` tokens, each starting `size - overlap` after the last."""
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    if len(offsets) <= size:
        return [text]
    step = size - overlap
    chunks = []
    for start in range(0, len(offsets), step):
        window = offsets[start:start + size]
        chunks.append(text[window[0][0]:window[-1][1]])
        if start + size >= len(offsets):
            break
    return chunks


def build(args):
    if not 0 <= args.overlap < args.size:
        raise SystemExit("--overlap must be in [0, size)")
    name = f"c{args.size}_o{args.overlap}"
    out = INDEX_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    data = CourseData(DATA_DIR)
    record_ids = json.loads((DATA_DIR / "embeddings" / "corpus_enc_a_ids.json").read_text())
    model = load_model(args.device)

    t0 = time.perf_counter()
    texts, owner = [], []
    for row, rid in enumerate(record_ids):
        for chunk in split_record(model.tokenizer, data.record_text(rid), args.size, args.overlap):
            texts.append(chunk)
            owner.append(row)
    chunk_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    vectors = model.encode(texts, batch_size=args.batch_size, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    encode_s = time.perf_counter() - t0

    np.save(out / "vectors.npy", vectors)
    np.save(out / "owner.npy", np.asarray(owner, dtype=np.int32))
    with (out / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for row, text in zip(owner, texts):
            f.write(json.dumps({"record_id": record_ids[row], "text": text}, ensure_ascii=False) + "\n")
    meta = {"name": name, "size": args.size, "overlap": args.overlap, "chunks": len(texts),
            "records": len(record_ids), "device": args.device, "chunk_s": round(chunk_s, 2),
            "encode_s": round(encode_s, 2), "vectors_mb": round(vectors.nbytes / 1e6, 2)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


class ChunkRetriever:
    """Score each record by its best chunk; `mix` blends in the whole-record score."""

    def __init__(self, name, doc_vectors, mix=0.0):
        d = INDEX_DIR / name
        self.vectors = np.load(d / "vectors.npy")
        self.owner = np.load(d / "owner.npy")
        self.doc_vectors = doc_vectors
        self.mix = mix
        self.n_records = doc_vectors.shape[0]

    def rank(self, q, k):
        start = time.perf_counter()
        best = np.full(self.n_records, -np.inf, dtype=np.float32)
        np.maximum.at(best, self.owner, self.vectors @ q)
        if self.mix:
            best = (1.0 - self.mix) * best + self.mix * (self.doc_vectors @ q)
        idx = np.argpartition(-best, k - 1)[:k]
        idx = idx[np.argsort(-best[idx])]
        return idx, (time.perf_counter() - start) * 1000.0


def rank_records(doc_vectors, q, k):
    start = time.perf_counter()
    scores = doc_vectors @ q
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx, (time.perf_counter() - start) * 1000.0


def evaluate(args):
    data = CourseData(DATA_DIR)
    qrels = data.load_qrels(args.split)
    record_ids = json.loads((DATA_DIR / "embeddings" / "corpus_enc_a_ids.json").read_text())
    doc_vectors = np.load(DATA_DIR / "embeddings" / "corpus_enc_a.npy").astype(np.float32)

    with args.transcripts.open(encoding="utf-8") as f:
        transcripts = {r["query_id"]: r["transcript"] for r in map(json.loads, f) if r}
    qids = [it["query_id"] for it in data.audio_items(args.split) if it["query_id"] in transcripts]
    missing = len(data.audio_items(args.split)) - len(qids)
    if missing:
        print(f"warning: {missing} queries have no transcript and are skipped")

    model = load_model(args.device)
    tokenizer = model.tokenizer
    Q = model.encode([transcripts[q] for q in qids], convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    # Bucket queries by the token length of their longest relevant record, to see where chunking acts.
    rel_len = {}
    for qid in qids:
        rel_len[qid] = max(len(tokenizer(data.record_text(r))["input_ids"]) for r in qrels[qid])
    buckets = {"short": lambda n: n <= 128, "medium": lambda n: 128 < n <= 256, "long": lambda n: n > 256}

    systems = {"baseline (record)": (lambda q, k: rank_records(doc_vectors, q, k), doc_vectors.nbytes)}
    for name in args.index:
        for mix in args.mix:
            r = ChunkRetriever(name, doc_vectors, mix)
            label = name if mix == 0 else f"{name} mix={mix}"
            size = r.vectors.nbytes + r.owner.nbytes + (doc_vectors.nbytes if mix else 0)
            systems[label] = (r.rank, size)

    report = {}
    for label, (rank, nbytes) in systems.items():
        rows, latency = [], []
        for qid, q in zip(qids, Q):
            idx, ms = rank(q, args.k_max)
            latency.append(ms)
            ranked = [record_ids[i] for i in idx]
            rel = qrels[qid]
            rows.append({"qid": qid, "Recall@5": recall_at_k(ranked, rel, 5),
                         "Recall@10": recall_at_k(ranked, rel, 10), "Recall@20": recall_at_k(ranked, rel, 20),
                         "Recall@100": recall_at_k(ranked, rel, 100), "MRR@10": mrr_at_k(ranked, rel, 10),
                         "NDCG@10": ndcg_at_k(ranked, rel, 10)})
        res = {m: float(np.mean([r[m] for r in rows])) for m in METRICS}
        res["index_mb"] = round(nbytes / 1e6, 1)
        res["search_ms_p50"] = float(np.percentile(latency, 50))
        res["search_ms_p95"] = float(np.percentile(latency, 95))
        for b, test in buckets.items():
            sel = [r["Recall@10"] for r in rows if test(rel_len[r["qid"]])]
            res[f"R@10[{b}, n={len(sel)}]"] = float(np.mean(sel)) if sel else float("nan")
        report[label] = {"summary": res, "per_query": rows}

    print(f"\nsplit={args.split}  queries={len(qids)}  k_max={args.k_max}")
    cols = METRICS + ["index_mb", "search_ms_p50"]
    print(f"{'system':<24}" + "".join(f"{c:>12}" for c in cols))
    for label, r in report.items():
        print(f"{label:<24}" + "".join(f"{r['summary'][c]:>12.4f}" if c in METRICS else f"{r['summary'][c]:>12.2f}" for c in cols))
    print("\nRecall@10 by length of the longest relevant record (tokens):")
    for label, r in report.items():
        print(f"{label:<24}" + "".join(f"  {k} {v:.4f}" for k, v in r["summary"].items() if k.startswith("R@10[")))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"split": args.split, "queries": len(qids), "systems": report}, indent=2))
        print(f"\nsaved {args.output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="split records into chunks and encode them")
    b.add_argument("--size", type=int, default=128, help="chunk length in tokens")
    b.add_argument("--overlap", type=int, default=32, help="tokens shared by neighbouring chunks")
    b.add_argument("--batch-size", type=int, default=64)
    b.add_argument("--device", default=None)
    b.set_defaults(func=build)

    e = sub.add_parser("evaluate", help="compare chunk indexes with the record-level baseline")
    e.add_argument("--split", choices=["practice", "dev"], default="practice")
    e.add_argument("--transcripts", type=Path, required=True)
    e.add_argument("--index", action="append", default=[], help="chunk index name, e.g. c128_o32")
    e.add_argument("--mix", type=float, action="append", default=None,
                   help="weight of the whole-record score (0 = chunks only); repeatable")
    e.add_argument("--k-max", type=int, default=100)
    e.add_argument("--device", default=None)
    e.add_argument("--output", type=Path, default=None)
    e.set_defaults(func=evaluate)

    args = ap.parse_args()
    if getattr(args, "mix", None) is None and args.cmd == "evaluate":
        args.mix = [0.0]
    args.func(args)


if __name__ == "__main__":
    main()
