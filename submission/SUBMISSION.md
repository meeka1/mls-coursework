# Code submission

Paths mirror the course repository. Supplied files that we did not change are not included.

## ASR kernels (Homework 1)

`hw1-asr/csrc/`: the five task files, used with the supplied `ops.py`, `model.py`,
`benchmark.py`, `provided.cu` and tests, all unchanged.

    cd hw1-asr && make test && ./run.sh --profile

## ASR speed-ups (extension, report Section 4.4)

`hw1-asr/speedups/` holds our versions, kept apart so that the supplied `model.py` stays
unchanged for the kernel tasks:

- `model.py`: the supplied model plus two switches, `trim_padding` (encode only the real
  audio frames) and `use_kv_cache` (decoder KV cache). Both are on by default.
- `bench_asr.py`: compares the supplied model, trim, and trim + KV cache on the same clips.

To reproduce, copy them into a course checkout:

    cp hw1-asr/speedups/model.py <repo>/hw1-asr/glm_asr_cuda/model.py
    cp hw1-asr/speedups/bench_asr.py <repo>/hw1-asr/
    cd <repo>/hw1-asr
    python bench_asr.py --cuda none --limit 100 --output ../results/asr/bench_torch.jsonl
    python bench_asr.py --cuda all  --limit 100 --output ../results/asr/bench_cuda.jsonl

## RAG investigation (report Sections 4.1–4.3 and 4.5)

`rag/extensions/`: all investigation code; `rag/extensions/README.md` lists the commands.
The supplied `rag/src/` is unchanged.

| File | Purpose |
|---|---|
| `transcribe_queries.py` | Transcribe the audio queries once and cache them |
| `migration.py` | Encoder-migration strategies: sweep, diagnostics, cost |
| `migration_checks.py` | Sanity checks for map-docs (held-out records, who finds the hits) |
| `migration_answers.py` | ELM answers from each index state |
| `judge_robust.py` | Course judge with retries, paired bootstrap comparison |
| `first_stage.py` | BM25 and fused first stages |
| `selective_rerank.py` | Margin-gated cross-encoder reranking |
| `corrective.py` | Margin-gated LLM query correction (rewrite, HyDE) |
| `chunking.py` | Chunked-record retrieval |

The ELM key is read from `rag/.env` (not included).

## Results and figures

- `results/README.md`: summary of all final numbers.
- `results/asr/`: `make test`, `./run.sh --profile` (before and after the speed-ups),
  `nvidia-smi -L` and the ASR benchmark rows.
- `rag/results/`: retrieval sweeps, ELM answers and judge scores.
- `report/make_figures.py`: builds both report figures from these files.
