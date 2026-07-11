"""
the mitigation that actually works: late-interaction (MaxSim) re-ranking.

the diagnosis first, because the mitigation is only as good as the failure it targets. i
looked at where the gold chunk ranked for every question, and the misses were dominated by
one pattern: the right DOCUMENT was often already ranked #1, but a sibling chunk from that
same document outranked the gold, or the gold sat just outside the top-k. the gold's score
was close to the winner's but not quite there.

why that happens - and it's the same mechanism as lost-in-the-middle, one level down:
  every chunk is compressed into ONE vector by masked mean-pooling over its tokens (see
  embed.py). when the answer is one sentence inside a 256-token chunk, its signal gets
  averaged together with ~250 tokens of surrounding prose. the mean-pooled vector is a blur
  of the whole chunk, so the specific answer sentence is diluted - exactly the way a fact in
  the middle of a long context gets diluted by everything around it. mean-pooling is
  lost-in-the-middle at the chunk level.

the fix - MaxSim / late interaction (the ColBERT scoring function):
  instead of comparing one pooled query vector to one pooled chunk vector, compare every
  query token to every chunk token and keep, for each query token, its best match:

        maxsim(q, c) = sum over query tokens i of  max over chunk tokens j of  (q_i . c_j)

  a single sentence in the chunk that strongly answers the query lights up the max for the
  relevant query tokens, even if the chunk's MEAN vector is muddy. so a buried answer stops
  being diluted - it's scored on its best-matching span, not on the chunk average. that's
  the precise antidote to the dilution the baseline suffers from.

architecture - retrieve then re-rank (standard two-stage, all from scratch):
  stage 1: fast pooled cosine over all ~13k chunks -> top-N candidates (N=50). cheap.
  stage 2: MaxSim re-rank those N by scoring query tokens against chunk tokens. expensive
           per chunk, but only paid on N, not on the whole corpus.
  we never build a token index over the corpus (that's the ColBERT storage cost we're
  avoiding); we recompute token embeddings for just the N candidates at query time.

result on the 45-question benchmark: hit@6 goes 0.747 -> 0.919 overall, and every bucket
improves - first 0.71->0.86, middle 0.80->0.90, last 0.73->1.00. the delta is reported per
bucket by run_rerank / the benchmark diff below.

    python -m src.rerank
"""

import json
from pathlib import Path

import numpy as np
import torch

from src.benchmark import BUCKETS, load_qa, resolve_qa
from src.chunk import load_config
from src.corpus import build_corpus
from src.embed import add_prefix
from src.retrieve import cosine_sim, top_k

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"


@torch.no_grad()
def token_embeddings(embedder, texts, kind):
    """per-token unit vectors + attention mask, for late interaction.

    same tokenizer / prefix / truncation as embed.py, but we keep the FULL sequence of token
    hidden states instead of mean-pooling them into one vector. each token is L2-normalized so
    a dot product between a query token and a chunk token is their cosine. returns
    (batch, seq, dim) and the (batch, seq) mask so we can ignore padding.
    """
    texts = add_prefix(texts, kind, embedder.model_name)
    enc = embedder.tokenizer(
        texts, padding=True, truncation=True,
        max_length=embedder.max_tokens, return_tensors="pt",
    )
    hidden = embedder.model(**enc).last_hidden_state
    hidden = torch.nn.functional.normalize(hidden, p=2, dim=2)
    return hidden, enc["attention_mask"]


def maxsim(query_tokens, chunk_tokens):
    """the ColBERT late-interaction score: sum over query tokens of the max similarity to any
    chunk token. query_tokens (Lq, dim), chunk_tokens (Lc, dim), both unit-normalized, so
    query_tokens @ chunk_tokens.T is the (Lq, Lc) cosine matrix. max over Lc picks each query
    token's best match in the chunk; sum over Lq totals it. a buried but strongly-matching
    span dominates the score even if the chunk mean is diluted."""
    sim = query_tokens @ chunk_tokens.T          # (Lq, Lc)
    return sim.max(dim=1).values.sum().item()    # sum_i max_j


def rerank_query(embedder, question, cand_idx, chunks):
    """MaxSim-score each candidate chunk against the question, return candidate indices sorted
    best-first. cand_idx are corpus row indices from the stage-1 pooled retrieval."""
    qtok, qmask = token_embeddings(embedder, [question], "query")
    qtok = qtok[0][qmask[0].bool()]              # (Lq, dim), drop padding

    texts = [chunks[int(i)]["text"] for i in cand_idx]
    scores = np.empty(len(texts))
    # batch the candidate token-embedding passes so we don't hold all N sequences at once
    pos = 0
    for start in range(0, len(texts), 16):
        ctok, cmask = token_embeddings(embedder, texts[start:start + 16], "passage")
        for b in range(ctok.shape[0]):
            c = ctok[b][cmask[b].bool()]         # (Lc, dim)
            scores[pos] = maxsim(qtok, c)
            pos += 1
    order = np.argsort(-scores)
    return [int(cand_idx[o]) for o in order], scores[order]


def run_rerank(k=None, n_candidates=50, cfg=None):
    """two-stage retrieval scored the same way as the baseline: hit@k by position bucket, plus
    the strict hit@1 overall.

    stage 1 pooled cosine -> top-n_candidates; stage 2 MaxSim re-rank -> top-k. gold comes from
    resolve_qa (evidence-span overlap), identical to benchmark.py, so the numbers are directly
    comparable to the baseline.
    """
    cfg = cfg or load_config()
    k = k or cfg["benchmark"]["k"]

    chunks, vectors, embedder = build_corpus(cfg)
    qa = resolve_qa(load_qa(), chunks)

    per_bucket = {b: [] for b in BUCKETS}
    hit1_all = []
    for q in qa:
        qvec = embedder.encode([q["question"]], "query")[0]
        scores = cosine_sim(qvec, vectors)
        cand, _ = top_k(scores, n_candidates)
        ranked, _ = rerank_query(embedder, q["question"], cand, chunks)

        gold = set(q["gold"])
        per_bucket[q["bucket"]].append(1.0 if gold & set(ranked[:k]) else 0.0)
        hit1_all.append(1.0 if ranked and ranked[0] in gold else 0.0)

    out = {"k": k, "n": len(qa), "n_candidates": n_candidates,
           "hit@1": float(np.mean(hit1_all)) if hit1_all else 0.0, "buckets": {}}
    for b in BUCKETS:
        hits = per_bucket[b]
        out["buckets"][b] = {"hit_rate": float(np.mean(hits)) if hits else 0.0, "n": len(hits)}
    return out


def _print_delta(baseline, rerank):
    print("=" * 60)
    print(f"MITIGATION (MaxSim late-interaction re-rank) - hit@{rerank['k']} by position")
    print(f"(stage 1: pooled cosine -> top-{rerank['n_candidates']}; stage 2: MaxSim -> top-{rerank['k']})")
    print("=" * 60)
    print(f"  {'bucket':<8} {'n':>4} {'base':>8} {'rerank':>8} {'delta':>8}")
    for b in BUCKETS:
        base = baseline["buckets"][b]["hit_rate"]
        new = rerank["buckets"][b]["hit_rate"]
        n = rerank["buckets"][b]["n"]
        print(f"  {b:<8} {n:>4} {base:>8.3f} {new:>8.3f} {new - base:>+8.3f}")
    base_all = np.mean([baseline["buckets"][b]["hit_rate"] for b in BUCKETS])
    new_all = np.mean([rerank["buckets"][b]["hit_rate"] for b in BUCKETS])
    print(f"\n  overall (unweighted bucket mean): {base_all:.3f} -> {new_all:.3f} "
          f"({new_all - base_all:+.3f})")
    mid_d = rerank["buckets"]["middle"]["hit_rate"] - baseline["buckets"]["middle"]["hit_rate"]
    print(f"  middle bucket: {mid_d:+.3f}  (the buried-answer case the mean-pool was diluting)")
    if "hit@1" in baseline and "hit@1" in rerank:
        print(f"  hit@1 (gold ranked #1): {baseline['hit@1']:.3f} -> {rerank['hit@1']:.3f} "
              f"({rerank['hit@1'] - baseline['hit@1']:+.3f})  -- the strict bar moves too")


if __name__ == "__main__":
    cfg = load_config()
    from src.benchmark import run_baseline
    baseline = run_baseline(cfg=cfg)
    rerank = run_rerank(cfg=cfg)
    _print_delta(baseline, rerank)

    out = ROOT / "results" / "rerank.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"baseline": baseline, "rerank": rerank}, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")
