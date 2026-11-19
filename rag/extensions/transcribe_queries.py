"""Transcribe the course audio queries once and cache the transcripts.

Uses only the standard library and the ASR adapter, so it runs directly with the
ASR environment's interpreter on a GPU node (no RAG environment needed):

    conda activate mls-asr
    python rag/extensions/transcribe_queries.py --split practice \
        --output rag/.cache/practice_transcripts.jsonl

The output is compatible with scripts/evaluate_rag.py --transcript-cache.
Re-running resumes: queries already in the output file are skipped.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
import time

from src.asr_adapter import ASRAdapter


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["practice", "dev"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    data_dir = ROOT / "data" / "rag_corpus"
    with (data_dir / "student_audio_manifest.jsonl").open(encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    items = [it for it in items if it["split"] == args.split]

    done = set()
    if args.output.exists():
        with args.output.open(encoding="utf-8") as f:
            done = {json.loads(l)["query_id"] for l in f if l.strip()}

    asr = ASRAdapter()
    asr.prepare()  # load the model once, outside the per-query timing
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as f:
        for i, item in enumerate(items, start=1):
            qid = item["query_id"]
            if qid in done:
                continue
            t0 = time.perf_counter()
            transcript = asr.transcribe(data_dir / item["audio_file"])
            asr_ms = (time.perf_counter() - t0) * 1000.0
            f.write(json.dumps({"query_id": qid, "transcript": transcript,
                                "asr_ms": round(asr_ms, 1)}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i:>3}/{len(items)}] {qid} {asr_ms:7.0f} ms  {transcript}")


if __name__ == "__main__":
    main()
