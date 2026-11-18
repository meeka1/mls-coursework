# Final results

Headline numbers for the report, with pointers to the raw files. All GPU timings are on
one NVIDIA GeForce RTX 2080 Ti (Teaching partition); RAG retrieval timings are on CPU.

## ASR: CUDA kernels (tasks 1–4)

`make test` passes for all four tasks (`asr/make_test.txt`). End-to-end on `test_audio.wav`
before any speed-up, our kernels transcribe it correctly (100% accuracy) in 3314 ms per
transcription, against 820 ms for PyTorch on the same GPU (`asr/profile_torch.txt`). `linear` takes 83% of kernel time and
`attention` 16% (`asr/profile_cuda.txt`).

## ASR: speed-ups (`hw1-asr/bench_asr.py`)

Two changes in `hw1-asr/glm_asr_cuda/model.py`, both switchable:

- **trim**: the processor pads every clip to 30 s. The encoder now sees only the real
  frames (about 5 s for a typical question instead of 30 s).
- **KV cache**: the decoder runs the prompt once, then one new position per token, reusing
  the keys and values of earlier positions instead of recomputing the whole sequence.

101 clips (`test_audio.wav` + 100 practice questions), every clip run with every variant
in rotating order, one model load, warm-up excluded. Medians per clip:

| Kernels | Variant | Encoder ms | Total ms | Speed-up | ms / token | Same text as base |
|---|---|---|---|---|---|---|
| PyTorch | base | 293 | 1212 | 1.00× | 80.8 | 101/101 |
| PyTorch | trim | 56 | 973 | 1.25× | 65.2 | 90/101 |
| PyTorch | trim + KV | 55 | 1017 | 1.19× | 68.0 | 90/101 |
| ours | base | 1711 | 4104 | 1.00× | 285.6 | 101/101 |
| ours | trim | 233 | 2613 | 1.57× | 178.6 | 90/101 |
| ours | trim + KV | 234 | **911** | **4.51×** | 61.9 | 90/101 |

- The KV cache never changes a transcript (0/101 differ from trim, on both backends).
- The 11 transcripts that trim changes are formatting, mostly better than base:
  "I P O" → "IPO", "Form Ten Ninety Nine B" → "Form 1099-B". Word error rate against
  base is 0.025. Peak memory stays at about 8.5 GiB.
- Downstream, retrieval is not worse: enc_a search on the 100 practice questions gives
  NDCG@10 0.448 with base transcripts and 0.466 with trim + KV, +0.017 [−0.000, +0.044]
  (5 queries better, 1 worse). Both backends give the same transcripts.
- The KV cache helps our kernels 2.9× and PyTorch not at all. Our tiled GEMM's time grows
  with the number of rows, and a cached step has 1 row instead of about 65. PyTorch's
  step costs about the same for 1 or 65 rows: it is limited by reading the weights and
  by launching many small operations, not by arithmetic. The extra concatenations cost it
  about 44 ms.

Raw rows: `asr/bench_torch.jsonl`, `asr/bench_cuda.jsonl`. Re-print the table with
`python hw1-asr/bench_asr.py --summary results/asr/bench_torch.jsonl results/asr/bench_cuda.jsonl`.

## RAG: main investigation, serving retrieval during an encoder migration

Details, method and commands: `rag/extensions/README.md`. Dev split, 125 queries.

Recall@10 by fraction of the index re-encoded (mean over 5 seeds; baseline 0.475):

| Migrated | no-map | naive | z-calibrated | dual-read | map-docs |
|---|---|---|---|---|---|
| 10% | 0.079 | 0.205 | 0.390 | 0.423 | **0.464** |
| 50% | 0.289 | 0.339 | 0.368 | 0.456 | **0.460** |
| 90% | 0.449 | 0.459 | 0.463 | 0.407 | **0.476** |

- map-docs keeps 97–100% of the baseline on Recall@10/100, MRR@10 and NDCG@10 at every
  fraction, with the same search time (about 0.3 ms per query on CPU) and a slightly
  smaller index. Fitting the maps takes about 245 ms; mapping all legacy records about 11 ms.
- Answer quality at 10% migrated (ELM answers, course judge): map-docs vs naive
  +0.060 [+0.006, +0.115]; map-docs vs all-enc_a target +0.006 [−0.044, +0.057].
- Sanity checks (`rag/extensions/migration_checks.py`): raw enc_b vectors alone reach
  0.423, so the failure is mixing two score scales in one list, not unrelated spaces. The
  map generalises to records it was not fitted on (98% of baseline). Mapped legacy records
  supply relevant hits at or above their share of the corpus.

## RAG: secondary results

- **Corrective retrieval** (one ELM call per query, `rag/extensions/corrective.py`):
  hyde + mix, chosen on practice, gives dev NDCG@10 0.398 → 0.420, +0.022 [+0.000, +0.046],
  for about 2.4 s per query. Margin gating does no better than random selection.
- **Selective reranking** with a MiniLM cross-encoder: negative. NDCG@10 0.398 → 0.351 when
  every query is reranked, 5.4 s per query on CPU.
- **Chunked retrieval** (practice): at most +0.02 Recall@10 for 3× the index memory.

## Not done

- Answer quality for corrective retrieval, and for migration at fractions other than 10%.
- Judge noise: each answer was judged once.
