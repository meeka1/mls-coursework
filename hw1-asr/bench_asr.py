"""Speed and transcript changes of the ASR speed-ups, on real course audio.

Variants (GlmAsr switches in glm_asr_cuda/model.py):
  base     the supplied model: 30 s padding through the encoder, no KV cache
  trim     encoder sees only the real audio frames
  trim_kv  trim + KV cache in the decoder (prompt once, then one new position per token)

The model is loaded once; every clip is transcribed with every variant, in rotating
order, so all variants share the GPU, clips and warm state. Transcripts are compared
with the base variant.

    # on a GPU node, in hw1-asr, mls-asr environment
    python bench_asr.py --cuda none --limit 100 --output ../results/asr/bench_torch.jsonl
    python bench_asr.py --cuda all  --limit 100 --output ../results/asr/bench_cuda.jsonl
    python bench_asr.py --summary ../results/asr/bench_torch.jsonl ../results/asr/bench_cuda.jsonl
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import wave

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
AUDIO_ROOT = HERE.parent / "rag" / "data" / "rag_corpus"
VARIANTS = {"base": (False, False), "trim": (True, False), "trim_kv": (True, True)}  # (trim, kv cache)


def read_wav(path):
    import numpy as np

    with wave.open(str(path), "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    return x / 32768.0


def words(text):
    return "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in text).split()


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(len(r), 1)


def clips(split, limit):
    yield "test_audio", HERE / "test_audio.wav"
    with (AUDIO_ROOT / "student_audio_manifest.jsonl").open() as f:
        items = [json.loads(l) for l in f if l.strip()]
    for it in [it for it in items if it["split"] == split][:limit]:
        yield it["query_id"], AUDIO_ROOT / it["audio_file"]


def run(args):
    import torch
    from benchmark import use_cache_if_present
    from glm_asr_cuda import ops
    from glm_asr_cuda.model import MODEL_NAME, load_model

    ops.set_cuda_groups(args.cuda)
    use_cache_if_present(MODEL_NAME)
    model, processor = load_model()
    names = list(VARIANTS)

    def transcribe(feats, ids, mask, variant):
        model.trim_padding, model.use_kv_cache = VARIANTS[variant]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.encode_audio(feats, mask)
        torch.cuda.synchronize()
        enc_ms = (time.perf_counter() - t0) * 1000
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        tokens, _ = model.generate(feats, ids, mask)
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000
        text = processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        return {f"{variant}_text": text, f"{variant}_encoder_ms": round(enc_ms, 1),
                f"{variant}_total_ms": round(total_ms, 1), f"{variant}_tokens": len(tokens),
                f"{variant}_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}

    gpu = torch.cuda.get_device_name()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for n, (qid, path) in enumerate(clips(args.split, args.limit)):
            audio = read_wav(path)
            inp = processor.apply_transcription_request(audio)
            feats = inp["input_features"].to("cuda", torch.float32)
            ids = inp["input_ids"].to("cuda")
            mask = inp["input_features_mask"].to("cuda")
            if n == 0:  # warm-up, not recorded
                for v in names:
                    transcribe(feats, ids, mask, v)
            row = {"query_id": qid, "seconds": round(len(audio) / 16000, 2), "gpu": gpu, "cuda": args.cuda,
                   "prompt_tokens": int(ids.shape[1])}
            for v in names[n % len(names):] + names[:n % len(names)]:
                row.update(transcribe(feats, ids, mask, v))
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"[{n}] {qid} {row['seconds']:5.2f}s  " + "  ".join(
                f"{v} {row[f'{v}_total_ms']:7.0f} ms" for v in names) +
                f"  same={all(row[f'{v}_text'] == row[f'{names[0]}_text'] for v in names)}", flush=True)
    model.trim_padding, model.use_kv_cache = True, True
    summary([args.output])


def summary(paths):
    for p in paths:
        rows = [json.loads(l) for l in open(p)]
        names = [k[:-len("_total_ms")] for k in rows[0] if k.endswith("_total_ms")]
        ref = names[0]
        med = lambda k: statistics.median(r[k] for r in rows)
        print(f"\n{p}: {len(rows)} clips, {rows[0]['gpu']}, kernels={rows[0]['cuda']}, "
              f"median clip {med('seconds'):.1f} s, median {med(f'{ref}_tokens'):.0f} tokens")
        print(f"  {'variant':9s}{'encoder ms':>12}{'total ms':>11}{'speed-up':>10}{'ms/token':>10}"
              f"{'peak GiB':>10}{'same text':>11}{'WER vs ' + ref:>14}")
        for v in names:
            same = sum(r[f"{v}_text"] == r[f"{ref}_text"] for r in rows)
            w = statistics.mean(wer(r[f"{ref}_text"], r[f"{v}_text"]) for r in rows)
            per_tok = statistics.median(r[f"{v}_total_ms"] / max(r[f"{v}_tokens"], 1) for r in rows)
            peak = f"{med(f'{v}_peak_gib'):>10.2f}" if f"{v}_peak_gib" in rows[0] else f"{'-':>10}"
            print(f"  {v:9s}{med(f'{v}_encoder_ms'):>12.1f}{med(f'{v}_total_ms'):>11.1f}"
                  f"{med(f'{ref}_total_ms') / med(f'{v}_total_ms'):>9.2f}x{per_tok:>10.1f}{peak}"
                  f"{same:>7}/{len(rows):<3}{w:>14.3f}")
        for v in names[1:]:
            prev = names[names.index(v) - 1]
            diff = [r for r in rows if r[f"{v}_text"] != r[f"{prev}_text"]]
            print(f"  {v} vs {prev}: {len(diff)} transcripts differ")
            for r in diff[:20]:
                print(f"    {r['query_id']}\n      {prev}: {r[f'{prev}_text']}\n      {v}: {r[f'{v}_text']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cuda", default="all", help="kernel groups, as in run.sh (all / none / ...)")
    ap.add_argument("--split", choices=["practice", "dev"], default="practice")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--output", type=Path, default=HERE.parent / "results" / "asr" / "bench.jsonl")
    ap.add_argument("--summary", type=Path, nargs="+", help="only summarise existing result files")
    args = ap.parse_args()
    if args.summary:
        summary(args.summary)
    else:
        run(args)


if __name__ == "__main__":
    main()
