# long-context retrieval fidelity engine

a retrieval system over 12 long public-domain books that measures whether retrieval quality
degrades based on *where in a document the answer lives* (the "lost-in-the-middle" question),
then tries a mitigation and reports what actually changed.

everything is from scratch: cosine similarity is hand-rolled numpy, no vector db, no rag
framework. the only heavy dependency is the huggingface embedding model loader.

the short version of the finding, up front and honest: **retrieval is largely position-
insensitive, so the baseline comes out roughly flat across positions, and the mitigation nets
zero.** that's not a bug or a weak result - it's the correct result, and the interesting part
is *why*. more on that below.

## quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.ingest          # download the 12 books from gutenberg (~1 min, one time)
python -m src.benchmark       # baseline: hit@k by position + precision/recall vs k
python -m src.mitigate        # RRF mitigation, before/after delta
python -m src.chunk_tradeoff  # fixed vs sentence chunking
python -m src.chart           # write the two png charts to results/

python -m src.cli "who kept the oracle of jupiter at dodona" --debug
```

**about the <5 min claim:** the one real cost is embedding ~13k chunks on cpu, which is a
~10-15 min *one-time* job. it's cached to disk (`data/cache/*.npy`) the first time, and every
run after that is warm: the full benchmark is ~10s, a cli query ~2s. so the reproducible
analysis loop the assignment cares about runs in well under 5 minutes - the first cold embed
is the exception, and it only happens once. i'd rather be straight about that than pretend a
cold cpu embed of a 2M-word corpus is instant.

## the benchmark (the core)

**what it measures:** for each of 36 question-answer pairs, does the retriever find the chunk
containing the answer, broken down by where that answer sits in its source document - first
10%, middle 40-60%, last 10%.

**how positions are known, honestly (no injection):** every qa pair in
[questions/qa.json](questions/qa.json) carries an `evidence` string - a verbatim fragment of
the source book. we string-match it back into the raw text to get the answer's exact char
offset; offset / doc-length is its relative position, which drops it into a bucket. the
"gold" chunk is whatever corpus chunk's `[char_start, char_end)` span covers that offset
(chunk.py records exact spans). nothing is invented or placed - the position is wherever the
fact actually occurs in the book. this matters because the obvious alternative (inject a
synthetic "needle" and relabel its position) doesn't work: a chunk vector is identical
regardless of the position label you attach, so its retrieval rank is too, and the benchmark
saturates to a meaningless flat 1.0. an earlier version of this repo did exactly that; it was
ripped out.

**why retrieve over the whole corpus, not just the answer's own doc:** if we searched only
the book the answer is in, there'd be almost no distractors and "did we find it" would be
trivially yes - that measures nothing. every question is ranked against all ~13k chunks from
all 12 books, so the position question is actually contested.

**baseline result (hit@6):**

| bucket | n  | hit@6 |
|--------|----|-------|
| first  | 14 | 0.714 |
| middle | 11 | 0.636 |
| last   | 11 | 0.727 |

spread max-min = 0.091. roughly flat, with the middle marginally lower (and on 11 questions
that dip is within noise). ![position chart](results/position_chart.png)

**why it's flat, and why that's the right answer:** this is the whole point of the
assignment. lost-in-the-middle is an *attention* failure - a transformer given a long context
under-attends to the middle because of softmax dilution over many tokens, positional decay,
and attention sinks at the edges. but retrieval here doesn't have a long context or an
attention mechanism over positions. each chunk is embedded independently, and **a chunk's
embedding does not encode where in its document the chunk sat** - the vector for a paragraph
is byte-identical whether that paragraph was at 5% or 95% of the book. cosine ranking over
position-free vectors is therefore position-insensitive *by construction*. so a flat baseline
isn't a failure to reproduce the effect - it's evidence the effect doesn't live at the
retrieval stage. it lives one step later, in the reader/LLM's attention over the assembled
context. (that reader stage is the honest "what i'd do next" - see below.)

**precision / recall vs k:**

| k  | recall | precision |
|----|--------|-----------|
| 1  | 0.306  | 0.306     |
| 3  | 0.583  | 0.194     |
| 5  | 0.667  | 0.133     |
| 10 | 0.722  | 0.072     |
| 20 | 0.861  | 0.043     |

recall climbs toward 1 as k grows (more slots, more chances to include the gold); precision
falls (one gold chunk spread over more slots, so hits/k drops). the useful operating point is
a small k where recall is already decent before precision collapses. ![pr curve](results/pr_curve.png)

## chunking

two strategies, both token-based off the actual e5 tokenizer (see [src/chunk.py](src/chunk.py)):

- **fixed_overlap** - a fixed 256-token window sliding with 50-token overlap. cuts wherever
  the window lands, often mid-sentence.
- **sentence_packed** - greedily pack whole sentences up to the same 256-token budget, never
  splitting a sentence.

everything is measured in tokens, not characters, because e5 has a 512-token hard cap and
truncates silently past it - the token budget is the only budget the model actually sees.

**why sentence boundaries (the justification the spec asks for):** e5 was pretrained and
contrastively fine-tuned on coherent spans, and we pool a chunk into one vector by masked
mean-pooling over its token embeddings. a chunk that ends mid-sentence is out-of-distribution
and its mean-pool smears two half-thoughts into one vector; keeping whole sentences means the
pooled vector represents one coherent thought. that's the *theory*.

**the tradeoff, measured:** smaller / edge-cut chunks buy positional diversity (a fact is
localized, not buried in filler) but lose semantic completeness. i measured the boundary
axis - fixed vs sentence at the *same* 256-token budget, so the only variable is where a
chunk may end - on the same 36-question benchmark:

| strategy | hit@6 |
|----------|-------|
| fixed    | 0.694 |
| sentence | 0.667 |

delta (sentence - fixed) = **-0.028**. so on this corpus the boundary-aware strategy does
*not* beat fixed+overlap - the theory says it should be cleaner, but the mid-sentence cuts
don't hurt retrieval enough to matter, and the 20% overlap gives fixed extra coverage that
edges out the coherence win. reporting the number the way it came, not the way the theory
predicted.

## embedding + retrieval

**model:** `intfloat/e5-base-v2` via huggingface (bge-base is swappable in config). e5 is an
*asymmetric* retriever - questions get a `query: ` prefix and passages get `passage: `,
because question text and passage text come from different distributions and the prefix tells
the model which encoder mode to use. get this wrong and the cosines are garbage. we pool by
masked mean over token hidden states and L2-normalize on the way out.

**cosine from scratch** ([src/retrieve.py](src/retrieve.py)): normalize each vector to unit
length, then similarity is just the dot product `q_hat @ M_hat.T`. that's the whole method,
one matmul, no library.

**cosine vs dot vs euclidean, and when each fails:**
- **dot product** lets magnitude leak into the score - a longer vector can win on length
  alone, i.e. ranking by verbosity rather than relevance. fails when vectors aren't
  normalized.
- **euclidean** and cosine give the *same ranking* once vectors are unit-normalized, since
  `||a-b||^2 = 2 - 2·cos(theta)` - so nearer equals more similar and we don't need a second
  distance. euclidean fails as a *relevance* measure on un-normalized vectors, where a big-
  magnitude vector is "far" from everything regardless of direction.
- **cosine** compares direction only, which is what we want for semantic similarity. its one
  caveat: it's not a true metric (no triangle inequality), which is fine because we only ever
  rank by it, never reason about distances between distances.

## mitigation: reciprocal rank fusion across the two chunkings

**the idea:** the fixed and sentence chunkings cut the same books differently, so a fact
diluted mid-chunk under one strategy can sit clean in a chunk under the other. they're two
partly-independent views of the corpus. RRF fuses their two ranked lists into one.

**the math:**

```
rrf(chunk) = sum over lists L of  1 / (c + rank_L(chunk)),   c = 60
```

two properties make this the right choice here:
1. **it fuses by rank, not score.** the fixed and sentence cosine distributions sit on
   different scales (different chunk sizes -> different score spreads), so averaging raw
   cosines would let the wider-spread list dominate. rank is scale-free, so neither view can
   bully the other. this is the single strongest reason to pick RRF over score averaging for
   *this* problem.
2. **the +c damps the top.** without it, `1/rank` makes rank-1 worth 1.0 and rank-2 worth
   0.5 - a cliff, so one list's #1 wins outright. with c=60, rank-1 is 1/61 and rank-2 is
   1/62, nearly equal, so a chunk needs support from *both* lists to rise. RRF rewards
   agreement.

**result: it nets exactly zero.** hit@6 by bucket is unchanged (0.714 / 0.636 / 0.727). but
it is *not* inert - against the fixed baseline it gains 3 questions and loses 3:

- **gains:** a gold ranked #1 in the sentence view but deep in the fixed view gets lifted
  into the fused top-k. RRF combining two views does rescue these.
- **losses:** a gold ranked strong in one view but weak in the other (e.g. #5 fixed, #25
  sentence) gets pushed *out* of the top-k by chunks that are medium in both views - because
  RRF rewards agreement, and here agreement isn't correlated with correctness.

so the honest read: on this corpus the two views agree on most questions, and on the few they
disagree about, the rescues and the breaks cancel. this follows directly from the flat
baseline - the loss was never position-specific at retrieval, so a fusion aimed at general
recall has no position-shaped gap to close. i'm reporting the wash straight rather than
cherry-picking a k or a subset where it happens to win.

## where it didn't work, and what i'd do with more time

- **the mitigation didn't improve the middle** (or anything). covered above - retrieval is
  position-flat by construction, so there was nothing position-shaped for RRF to fix.
- **buckets are thin** - 11 questions each for middle and last. the 0.09 baseline spread and
  the 3/3 RRF wash could move with more questions; i wouldn't over-read either. more qa pairs
  is the first thing i'd add.
- **the boundary-aware chunking underperformed** its own theory. worth digging into whether a
  different budget or a paragraph boundary changes that.
- **the real next step** is a reader stage. lost-in-the-middle bites in the LLM's attention
  over a long *assembled* context, not in retrieval. i'd build a small extractive/generative
  reader, assemble the retrieved chunks into one context with the answer placed at controlled
  positions, and measure answer accuracy by position there - that's where the U-shape should
  actually appear, and where a position-aware re-ordering mitigation (put the strongest chunks
  at the edges, weak ones in the middle) would have something to bite on. i scoped that out to
  keep this focused on the retrieval question the assignment asks, but it's the honest
  continuation.

## layout

```
src/ingest.py         download + strip the 12 books, write the manifest
src/chunk.py          the two chunking strategies, token-based, exact char spans
src/embed.py          e5 loader, query/passage prefixes, masked mean-pool, .npy cache
src/corpus.py         build all books into one (chunks, vectors) pair, row-aligned, cached
src/retrieve.py       cosine from scratch, top-k, precision/recall/reciprocal-rank
src/benchmark.py      hit@k by answer position + precision/recall vs k  (the baseline)
src/mitigate.py       RRF across the two chunkings, before/after delta
src/chunk_tradeoff.py fixed vs sentence, measured
src/chart.py          the two result charts
src/cli.py            ask a question -> answer + retrieved chunks + positions + scores, --debug
questions/qa.json     36 position-labeled qa pairs, answers verbatim from the books
results/              benchmark output (json) + charts (png), before and after mitigation
```

every module has a `python -m src.<name>` self-check or run entry point.
