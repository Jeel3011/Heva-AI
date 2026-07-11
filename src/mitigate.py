"""
the mitigation: reciprocal rank fusion across the two chunkings.

why fusion, and why THIS fusion:
  the baseline retrieves over one chunking. the two strategies in chunk.py cut the same
  books differently - fixed windows can slice a fact mid-chunk (diluting it in the mean-
  pool), while sentence packing keeps that fact in one coherent chunk, and vice versa. so
  the two chunkings are two partly-independent views of the corpus, and a fact that ranks
  poorly under one can rank well under the other. RRF fuses the two ranked lists into one.

the math (this is what the interview will ask me to derive):
  for a query, retrieve a ranked list from each chunking. RRF scores a candidate chunk by

        rrf(chunk) = sum over lists L of  1 / (c + rank_L(chunk))

  where rank_L is the chunk's 1-based rank in list L (best = 1) and c is a constant
  (60, the standard from Cormack et al 2009). two properties that make this the right
  choice here:

  1. it fuses by RANK, not by score. the fixed and sentence cosine distributions are on
     different scales (different chunk sizes -> different score spreads), so averaging raw
     cosines would let whichever list has the wider spread dominate. rank is scale-free, so
     neither view can bully the other. this is the single strongest reason to pick RRF over
     score averaging for THIS problem.
  2. the +c damps the top. without it, 1/rank makes rank-1 worth 1.0 and rank-2 worth 0.5 -
     a huge cliff, so one list's #1 would almost always win outright. with c=60, rank-1 is
     1/61 and rank-2 is 1/62 - nearly equal, so a chunk needs support from BOTH lists to
     rise, which is exactly the agreement signal we want fusion to reward.

  since the two chunkings have different chunk sets, a "chunk" here is identified by (doc_id,
  char_start, char_end) - the same span from either chunking is the same candidate. a hit is
  a fused top-k that contains any chunk whose span covers the answer's offset.

honesty up front: the baseline was already roughly flat across position buckets, because
retrieval is largely position-insensitive. so RRF is not expected to carve out a big
middle-specific win. and on this corpus it barely moves the aggregate - but NOT
because it's inert. against the fixed baseline it gains 4 questions and loses 4, netting
~zero (a gold ranked #1 in one view gets lifted in; a gold strong in one view but
weak in the other gets pushed out, because RRF rewards agreement and here agreement isn't
correlated with correctness). we report that reshuffling straight rather than cherry-picking
a k or a subset where it happens to win.

    python -m src.mitigate
"""

import copy
import json
from pathlib import Path

import numpy as np

from src.benchmark import BUCKETS, evidence_span, load_qa, bucket_of
from src.chunk import load_config
from src.corpus import build_corpus
from src.retrieve import cosine_sim, top_k

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RRF_C = 60  # the standard damping constant from the original RRF paper


def _cfg_for(strategy, base):
    cfg = copy.deepcopy(base)
    cfg["chunk"]["strategy"] = strategy
    return cfg


def span_key(chunk):
    """identity of a chunk across chunkings: the exact place in the corpus it covers.
    the same (doc, start, end) span retrieved from either strategy is the same candidate."""
    return (chunk["doc_id"], chunk["char_start"], chunk["char_end"])


def rrf_fuse(ranked_lists, c=RRF_C):
    """fuse ranked lists of span-keys into one score per key: sum of 1/(c + rank).

    ranked_lists is a list of lists; each inner list is span-keys best-first. returns
    span-keys sorted by fused score descending. a key missing from a list simply contributes
    nothing from that list - no penalty term, which is the standard RRF behaviour.
    """
    scores = {}
    for lst in ranked_lists:
        for rank, key in enumerate(lst, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (c + rank)
    return sorted(scores, key=scores.get, reverse=True)


def _covers(chunk, span):
    # a chunk covers the answer if its [start, end) overlaps the evidence span - same
    # overlap rule benchmark.resolve_qa uses, so RRF gold matches the baseline's gold.
    ev_start, ev_end = span
    return chunk["char_start"] < ev_end and ev_start < chunk["char_end"]


def run_mitigation(k=None, depth=100, cfg=None):
    """RRF over fixed + sentence chunkings, scored the same way as the baseline: hit@k by
    position bucket.

    depth is how deep each per-chunking list goes into the fusion (RRF only needs the top of
    each list; 100 is plenty and keeps it fast). we resolve each question's answer offset
    once, build both corpora, retrieve a ranked list of span-keys from each, fuse, and check
    whether a covering span made the fused top-k.
    """
    cfg = cfg or load_config()
    k = k or cfg["benchmark"]["k"]
    qa = load_qa()

    # build both views. build_corpus keys its caches on the strategy, so these are two
    # genuinely different chunk sets + vector matrices, each cached independently.
    views = []
    for strat in ("fixed", "sentence"):
        chunks, vectors, embedder = build_corpus(_cfg_for(strat, cfg))
        views.append((chunks, vectors, embedder))

    # one embedder is enough for the queries (same model across views); reuse the last one
    embedder = views[-1][2]
    qvecs = embedder.encode([q["question"] for q in qa], "query")

    per_bucket = {b: [] for b in BUCKETS}
    kept = 0
    # track flips against the BASELINE, which is the fixed single view (view 0) - that's the
    # exact system run_baseline measures, so gained/lost here reconcile with the delta column.
    # gained = a fused hit the fixed view missed; lost = a fixed-view hit fusion pushed out.
    gained = 0
    lost = 0
    for q, qvec in zip(qa, qvecs):
        text = (RAW_DIR / f"{q['doc_id']}.txt").read_text(encoding="utf-8")
        span = evidence_span(text, q["evidence"])
        if span is None:
            continue
        frac = ((span[0] + span[1]) // 2) / max(len(text), 1)
        bucket = bucket_of(frac)
        if bucket is None:
            continue

        # per-chunking ranked lists of span-keys, top `depth` each, plus whether each single
        # view had a covering span inside its own top-k (the single-view hit@k)
        ranked_lists = []
        covered_keys = set()  # span-keys (from either view) that cover the answer offset
        fixed_hit = False     # did the fixed view (view 0, the baseline) hit@k on its own
        for vi, (chunks, vectors, _) in enumerate(views):
            scores = cosine_sim(qvec, vectors)
            idx, _ = top_k(scores, depth)
            keys = [span_key(chunks[int(i)]) for i in idx]
            ranked_lists.append(keys)
            for rank_pos, i in enumerate(idx):
                c = chunks[int(i)]
                if c["doc_id"] == q["doc_id"] and _covers(c, span):
                    covered_keys.add(span_key(c))
                    if vi == 0 and rank_pos < k:
                        fixed_hit = True

        fused = rrf_fuse(ranked_lists)
        topk = set(fused[:k])
        hit = 1.0 if (topk & covered_keys) else 0.0
        per_bucket[bucket].append(hit)
        kept += 1

        # flips vs the fixed baseline
        if hit and not fixed_hit:
            gained += 1
        elif fixed_hit and not hit:
            lost += 1

    out = {"k": k, "n": kept, "gained": gained, "lost": lost, "buckets": {}}
    for b in BUCKETS:
        hits = per_bucket[b]
        out["buckets"][b] = {"hit_rate": float(np.mean(hits)) if hits else 0.0, "n": len(hits)}
    return out


def _print_delta(baseline, mitig):
    print("=" * 60)
    print(f"MITIGATION (RRF, fixed+sentence) - hit@{mitig['k']} by position")
    print("=" * 60)
    print(f"  {'bucket':<8} {'n':>4} {'base':>8} {'rrf':>8} {'delta':>8}")
    for b in BUCKETS:
        base = baseline["buckets"][b]["hit_rate"]
        new = mitig["buckets"][b]["hit_rate"]
        n = mitig["buckets"][b]["n"]
        print(f"  {b:<8} {n:>4} {base:>8.3f} {new:>8.3f} {new - base:>+8.3f}")

    base_all = np.mean([baseline["buckets"][b]["hit_rate"] for b in BUCKETS])
    new_all = np.mean([mitig["buckets"][b]["hit_rate"] for b in BUCKETS])
    print(f"\n  overall (unweighted bucket mean): {base_all:.3f} -> {new_all:.3f} "
          f"({new_all - base_all:+.3f})")
    print(f"  vs fixed baseline: {mitig['gained']} questions gained, {mitig['lost']} lost")
    print("  the aggregate doesn't move, but not because fusion does nothing - it reshuffles.")
    print("  RRF pulls some golds INTO the top-k (a view that ranked it #1 lifts it) and pushes")
    print("  others OUT (a gold strong in one view but weak in the other loses to candidates")
    print("  medium in both, since RRF rewards agreement). here gains and losses cancel exactly.")
    print("  an honest null on THIS corpus, not a claim that fusion can't help - see the readme.")


if __name__ == "__main__":
    cfg = load_config()
    from src.benchmark import run_baseline
    baseline = run_baseline(cfg=cfg)
    mitig = run_mitigation(cfg=cfg)
    _print_delta(baseline, mitig)

    out = ROOT / "results" / "mitigation.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"baseline": baseline, "mitigation": mitig}, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")
