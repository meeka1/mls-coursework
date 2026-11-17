Use this directory for investigation-specific code if useful. You are not required
to place all modifications here; substantive changes may naturally involve src/.

# Investigation: serving retrieval during an encoder migration

## Question

The index was built with a legacy encoder (enc_b) and is being re-encoded with
enc_a, record by record, while queries keep arriving. Queries can only be encoded
with enc_a (enc_b has no query encoder). How should a partly migrated index be
searched, and what does each strategy cost in quality, latency and memory?

The baseline is the supplied pipeline: enc_a for queries and all records, exact
top-k search.

## Strategies ([migration.py](migration.py))

| Strategy | How a partly migrated index is searched |
|---|---|
| naive | One list: enc_a scores for migrated records, mapped-query (A→B) scores for legacy records |
| z-calibrated | As naive, but each part's scores are standardised per query |
| dual-read | Search both parts separately, merge rankings with reciprocal rank fusion |
| map-docs | Map legacy record vectors into enc_a space once (B→A), then one index, one query |

The A→B and B→A maps are 384×384 least-squares fits on records that exist in
both spaces: 2,000 seed pairs plus every migrated record.

## How to run

```bash
# transcripts: rag/.cache/{split}_transcripts.jsonl (see transcribe_queries.py)
python rag/extensions/migration.py sweep --split dev --output rag/results/migration_dev.json
python rag/extensions/migration.py diagnose --split dev      # score scales, who fills the top 10
python rag/extensions/migration.py pairs --split dev         # how many paired records the map needs
python rag/extensions/migration_checks.py --split dev        # sanity checks below
python rag/extensions/migration_answers.py --split dev --fraction 0.1 \
    --out-dir rag/results/migration_answers_dev               # ELM answers per index state
python rag/extensions/judge_robust.py compare --dir rag/results/migration_answers_dev \
    --state target --state naive@10 --state map-docs@10
```

Retrieval runs make no ELM calls. Answer generation and judging do.

## Results (dev, 125 queries)

### Retrieval: Recall@10 by fraction migrated (mean over 5 seeds)

| Migrated | naive | z-calibrated | dual-read | map-docs |
|---|---|---|---|---|
| 0% | 0.453 | 0.453 | 0.453 | 0.465 |
| 10% | 0.205 | 0.390 | 0.423 | 0.464 |
| 25% | 0.262 | 0.346 | 0.451 | 0.460 |
| 50% | 0.339 | 0.368 | 0.456 | 0.460 |
| 75% | 0.418 | 0.427 | 0.430 | 0.466 |
| 90% | 0.459 | 0.463 | 0.407 | 0.476 |
| 100% (baseline) | 0.475 | 0.475 | 0.475 | 0.475 |

map-docs stays at 97–100% of the baseline on Recall@10, Recall@100, MRR@10 and
NDCG@10 at every fraction.

Cost at 50% migrated (one query at a time, CPU): search 0.29–0.34 ms for all
strategies; index 36.9 MB for map-docs vs 37.5 MB for the others; fitting both
maps about 245 ms; mapping all legacy records B→A about 11 ms.

### Answers: ELM generation + course judge, 10% migrated, top-5 records

`@10` in file names means 10% migrated, not top-10.

| Index state | Judge score (0–1), 124 common queries |
|---|---|
| target (all enc_a) | 0.333 |
| map-docs@10 | 0.339 |
| naive@10 | 0.279 |

One judge pass per answer (EuroLLM-22B); one map-docs answer was unscorable
(malformed judge JSON). The map-docs vs target gap is within judge noise.

### Sanity checks ([migration_checks.py](migration_checks.py))

Run on 2026-11-16 with the dev transcripts from `migration_answers_dev/target.jsonl`.

1. **No map: enc_a query against raw enc_b records.** Recall@10 is **0.423**
   (89% of the 0.475 baseline). enc_b is not an unrelated space: the enc_a and
   enc_b vectors of the same record have mean cosine 0.81, compared with 0.53 for
   random record pairs. enc_b is a related but misaligned encoder.
2. **Held-out: 0% migrated, all records mapped B→A**, counting only relevant
   records not used to fit the map. map-docs gets 0.461 ± 0.010 against a 0.471
   baseline on the same records (98%). The map generalises to records it was not
   fitted on.
3. **Who finds the hits** (map-docs, seed 0). Mapped legacy records supply relevant
   top-10 hits at or above their share of the corpus:

   | Migrated | Legacy share of corpus | Legacy share of relevant hits |
   |---|---|---|
   | 10% | 90% | 91% |
   | 25% | 75% | 79% |
   | 50% | 50% | 60% |
   | 75% | 25% | 33% |
   | 90% | 10% | 12% |

   The good map-docs results are not just enc_a records found by an enc_a query.

## Interpretation

- **The main failure is score scale, not meaning.** Against enc_a queries, enc_a
  records score higher overall than enc_b records (mean best match 0.90 vs 0.72;
  mean score 0.75 vs 0.54). When the two parts share one ranked list, enc_a
  records win the top 10 because their numbers are larger, not because they are
  more relevant. This is why naive collapses early in the migration, and why
  per-part standardisation (z-calibrated) or rank fusion (dual-read) help.
- **map-docs works because it puts every record on one scale and searches with a
  single query.** The map itself adds a modest +9% over using raw enc_b vectors
  (0.423 → 0.461 on held-out records). Its main value is removing the mixed-scale
  ranking problem at no extra search cost.
- Do not describe enc_a and enc_b as incompatible spaces; check 1 shows that
  claim is overstated.

## Still to do

- [ ] Add **no-map** (raw enc_b vectors, enc_a query) as a strategy in the sweep
      table, so the map's own contribution is visible at every fraction.
- [ ] Answer quality at more than one fraction (e.g. 50% and 0%), not only 10%.
- [ ] Judge noise: repeated judging on a subset, or a paired bootstrap CI from
      the per-query scores.
- [ ] Report hardware, timing boundaries, transcript source (ASR kernels or
      PyTorch) and caching.

# Secondary result: chunked retrieval ([chunking.py](chunking.py))

Practice split, 350 queries. Records are split into overlapping token windows,
and each record is scored by its best chunk, optionally mixed with the
whole-record score.

| System | Recall@10 | Index MB | Search ms (p50) |
|---|---|---|---|
| baseline (whole record) | 0.480 | 37 | 0.43 |
| c128_o32 | 0.464 | 75 | 0.67 |
| c256_o64 | 0.480 | 48 | 0.51 |
| c128_o32 mix=0.5 | 0.500 | 112 | 0.91 |
| c256_o64 mix=0.5 | 0.490 | 85 | 0.72 |

Mostly a negative result: the best variant gains 0.02 Recall@10 for 3× the index
memory and 2× the search time. Baseline recall barely depends on record length
(0.49 short, 0.49 medium, 0.47 long), so mean-pooling long records is not what
limits enc_a. Not yet run on dev, and not yet tested for answer quality.

# Investigation: improving retrieval for the answer stage

All numbers below are dev (125 queries), using the transcripts in
`migration_answers_dev/target.jsonl` and the supplied pipeline (enc_a, exact search,
top 5 records in the prompt) as the baseline.

## 1. Where the baseline fails

Queries grouped by the rank of their best relevant record, with the judge score of
the baseline answer (`judge_target`):

| Best relevant record ranked | Queries | Answer score |
|---|---|---|
| 1–5 (in the prompt) | 73 (58%) | 0.449 |
| 6–20 | 23 (18%) | 0.178 |
| 21–100 | 14 (11%) | 0.244 |
| not in top 100 | 15 (12%) | 0.072 |

Retrieval is the bottleneck: when a relevant record reaches the prompt, the answer
scores about 2.5× higher. Coverage grows with depth (queries with any relevant
record: 58% @5, 66% @10, 84% @50, 88% @100), so relevant records are often found
but ranked too low.

Most of the 20 queries with nothing relevant in the top 50 are not search failures:
speech-recognition errors ("no your customer" for *know your customer*), questions
with no topic ("I need your assistance and recommendations."), or relevant records
that depend on context absent from the question. Roughly 88–90% coverage at
depth 50 looks like a realistic ceiling.

## 2. Cheap first-stage changes do not help ([first_stage.py](first_stage.py))

Share of queries with a relevant record in the top k / Recall@k:

| First stage | @5 | @10 | @50 | @100 |
|---|---|---|---|---|
| enc_a (baseline) | 58% / 0.38 | 66% / 0.47 | 84% / 0.68 | 88% / 0.72 |
| BM25 | 33% / 0.20 | 40% / 0.25 | 55% / 0.37 | 65% / 0.45 |
| RRF enc_a + BM25 | 48% / 0.31 | 56% / 0.38 | 82% / 0.63 | 86% / 0.70 |
| RRF enc_a + enc_b | 56% / 0.37 | 64% / 0.46 | 83% / 0.66 | 86% / 0.73 |
| RRF enc_a + enc_b + BM25 | 51% / 0.33 | 63% / 0.43 | 84% / 0.66 | 86% / 0.72 |

Spoken questions are paraphrases that rarely share words with the answers, so BM25
is weak and fusing it in adds noise. enc_b adds nothing enc_a does not already have.

Appending fixed domain words to every query also hurts (ad-hoc test): "+ bank"
gives 57% @5, "+ finance" 56%, a "personal finance question:" prefix 38%. The whole
corpus is finance, so the words carry no information and dilute the question.

## 3. A free confidence signal: the margin

margin = enc_a score of result #1 − score of result #10, computed from the
scores the search already produces. A small margin means the top 10 look alike and
the search is guessing.

| Margin quarter | Queries | Relevant record in top 5 | Answer score |
|---|---|---|---|
| lowest | 32 | 22% | 0.195 |
| 2nd | 31 | 52% | 0.261 |
| 3rd | 31 | 74% | 0.376 |
| highest | 31 | 87% | 0.495 |

AUC for predicting "relevant record in top 5": margin 0.79, top-1 minus corpus mean
0.70, top-1 score 0.67, query length 0.59. The raw top-1 score is weak because enc_a
scores every query 0.86–0.95 on this corpus; the gap is what carries information.
The least confident dev query is "I need your assistance and recommendations."

The margin does not improve results by itself. It decides which queries receive a
costly correction step.

## 4. Selective reranking ([selective_rerank.py](selective_rerank.py))

A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, float32, CPU, 8 cores)
rescores the enc_a top 50, applied only to the lowest-margin share of queries.

| Reranked | Relevant in top 5 | Recall@10 | NDCG@10 | Extra time per query |
|---|---|---|---|---|
| 0% (baseline) | 0.584 | 0.475 | 0.398 | 0 |
| 25% lowest margin | 0.584 | 0.473 | 0.397 | 1.3 s |
| 50% | 0.560 | 0.449 | 0.380 | 2.7 s |
| 100% | 0.560 | 0.429 | 0.351 | 5.4 s |

Negative result. Reranking every query costs 5.3 s per query on CPU (median) and
lowers NDCG@10 from 0.398 to 0.351. The model was trained on MS MARCO web passages,
not long opinionated finance forum posts answering spoken questions. Margin gating
limits the damage (25% reranked costs nothing in quality and saves 75% of the time),
but cannot create gains when the action itself does not help. Even in the
lowest-margin quarter reranking does not help (0.22 → 0.19): those queries need a
better query, not a better order. Results: `rag/results/selective_rerank_minilm_dev.json`.

Note: loading the cross-encoder in float16 on CPU is extremely slow; the script
forces float32. The Hugging Face download failed in this environment with an httpx
decompression error, so the model files were fetched with curl and passed as a path.

## 5. Selective corrective retrieval ([corrective.py](corrective.py)) — in progress

For queries below a margin threshold, one ELM call corrects the query, then enc_a
searches again:

- `rewrite`: fix probable speech-recognition errors and restate the transcript as a
  clear, specific finance question.
- `hyde`: write a short hypothetical forum answer and search with it.

The second search is used as `replace` (corrected query alone), `fuse` (reciprocal
rank fusion with the original ranking) or `mix` (sum of the two query vectors).
The sweep corrects 0/25/50/75/100% of queries, lowest margin first, and reports
quality against ELM calls and added latency. It also compares with correcting the
same number of randomly chosen queries. ELM outputs are cached in
`rag/.cache/corrective_{mode}_{split}.jsonl`, so each mode costs one call per query
once.

```bash
cp rag/.env.example rag/.env      # add ELM_API_KEY; .env is git-ignored
python rag/extensions/corrective.py --split dev --mode rewrite --limit 5 --show
python rag/extensions/corrective.py --split dev --mode rewrite --output rag/results/corrective_rewrite_dev.json
python rag/extensions/corrective.py --split dev --mode hyde --output rag/results/corrective_hyde_dev.json
```

The pipeline has been tested end to end with a stand-in for the ELM output; no ELM
results yet.

## Still to do

- [ ] Run `corrective.py` (rewrite and hyde) on dev.
- [ ] Choose the margin threshold on **practice** and report on dev. This needs
      `rag/.cache/practice_transcripts.jsonl` (ASR on a GPU node, or a teammate's copy).
- [ ] Generate and judge answers for the best corrective setting, against the
      baseline answers already in `migration_answers_dev/target.jsonl`.
- [ ] Optional: a stronger reranker (e.g. `BAAI/bge-reranker-base`) on GPU, to test
      whether the reranking result is about this model or about reranking itself.
