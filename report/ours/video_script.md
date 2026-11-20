# Video script, short version (about 7–7.5 minutes)

Screen and voice only, no camera. On screen: the report PDF (pages 1–5), zoomed on the figure or
table being discussed. Do not show the last page (group information), names, UUNs, a terminal,
Overleaf, GitHub or email. Neutral title on Media Hopper, e.g. "MLS coursework video".

---

## Part 1 · The system and my questions (≈0:40)
*(Show: page 1)*

Hello. This coursework is about a system that answers spoken finance questions. A speech recogniser
turns the audio into text, an encoder embeds that text, exact search finds the closest records in a
corpus of 24,000, and a language model writes an answer from the top five.

I asked two main questions. First: if the index is being moved from an old encoder to a new one,
record by record, how should we search it while it is only half migrated? Second, as an extension:
where does the time go in the speech recogniser running on my own CUDA kernels, and how much of it is
wasted?

## Part 2 · Main investigation: searching a half-migrated index (≈2:30)
*(Show: Section 2, the five strategies)*

The corpus has vectors from a legacy encoder, enc_b, and it is being re-encoded with enc_a. Only enc_a
can encode queries. So during the migration, some records have new vectors and the rest only have old
ones.

I compared five strategies. No-map scores the old vectors directly with the new query. Naive maps the
query into the old space for the old records. Z-calibrated standardises the scores of each part.
Dual-read ranks the two parts separately and merges the rankings. And map-docs fits one least-squares
map from the old space to the new one, maps the old record vectors once, and then searches a single
index with a single query.

*(Show: Figure 1 and Table 1)*

The result is clear. Map-docs stays at 97 to 100 percent of the baseline at every stage of the
migration. The others lose the most at the start: with 10 percent migrated, naive keeps less than half
of the baseline's recall.

The interesting part is why. It is not that the old vectors are useless: when every record is still
old, even no-map reaches 89 percent of the baseline. The problem is the score scale. New records simply
get higher scores, so when both kinds share one ranked list, the few new records fill the top ten,
whatever their relevance. Calibration and rank fusion fix the scale only partly, and each of them fails
at some stage of the migration. Map-docs puts everything on one scale before ranking.

*(Show: Table 2)*

This reaches the answers too. With naive, only a third of questions get any relevant record into the
prompt. Answers with map-docs are clearly better than with naive, and indistinguishable from the fully
migrated index. And the cost is the same for all strategies; the map is cheap enough to refit after
every migration batch.

## Part 3 · Extension: the speech recogniser on my kernels (≈2:00)
*(Show: Section 4.3)*

I completed the four kernel tasks: RoPE, the reductions, a tiled matrix multiply and a streaming
attention. All tests pass, and the transcription is correct. The memory-bound kernels are close to the
hardware limit. The main gap is the matrix multiply, where cuBLAS is much faster, because it reuses
every loaded value many times in registers.

Then I looked at what the supplied model actually does. It pads every clip to 30 seconds, although a
question is about 5 seconds long. And the decoder has no key-value cache, so every new token re-runs
the whole sequence. I trimmed the padding and added a KV cache.

*(Show: Table 3)*

On my kernels, this makes transcription 4.5 times faster: from 4.1 seconds to 0.9 seconds per clip.
The KV cache changes no transcript at all. Trimming changes a few transcripts, almost all formatting,
and retrieval on the new transcripts is not worse.

*(Show: Figure 2)*

The surprising part is that the same KV cache gives 2.9 times on my kernels and nothing on PyTorch. My
kernel's time grows with the number of rows, and the cache turns about 80 rows into one. PyTorch's
decoder step is dominated by launching many small operations, and the cache does not remove those. So
whether an optimisation pays off depends on what the implementation is limited by.

## Part 4 · Conclusion (≈0:20)
*(Show: Section 5)*

To sum up: during a migration, keep one score scale; and in the recogniser, remove wasted work first.

---

## Question 1 · My main design choice (≈0:50)

My main design choice was to map the old document vectors into the new space, rather than mapping the
query into the old space.

Why? The diagnosis showed that the failure came from mixing two score scales in one list, not from
missing information in the old vectors. Mapping the query keeps two scales, and calibrating scores or
fusing ranks repairs this only partly. Mapping the documents puts every record on the new encoder's
scale, so the search becomes a single index with a single query, exactly like the final system. It is
also cheap: one small matrix, fitted in a quarter of a second, with no extra work per query. That is
why it keeps 97 to 100 percent of baseline quality from the first migrated record to the last.

## Question 2 · How I checked it works (≈0:50)

One example is the KV cache. It is easy to get subtly wrong, because the positions of new tokens have
to continue where the prompt ended, and the supplied RoPE kernel always numbers positions from zero.

I checked it in three steps. First, on the CPU, with a tiny model and random weights, I compared every
generated token with and without the cache. Second, on the GPU, I transcribed 101 real clips with the
cache on and off, on both PyTorch and my kernels: zero out of 101 transcripts differ. Third, because
the change must not hurt downstream quality, I re-ran retrieval on the new transcripts and checked that
it was not worse.

Thank you.
