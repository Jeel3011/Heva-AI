"""
the benchmark: retrieval fidelity as a function of where the answer sits in its document.

this is the core of the assignment, so read the design before the code.

the question the spec asks: does the retriever still FIND the answer when the answer lives
in the middle of a document rather than at the start or end? to measure that honestly we
need, per question, (a) a real position for the answer and (b) a real "gold" chunk that
contains it - both derived from the actual corpus, not injected.

how we get an honest position (no needles):
  every qa pair in questions/qa.json carries an `evidence` string - a verbatim fragment of
  the source book. we string-match it back into the raw text (whitespace-insensitively,
  because the chunker collapses whitespace the same way) to get the answer's exact char
  offset. offset / doc_length is the answer's relative position, which drops it into a
  bucket: first 10%, middle 40-60%, last 10%. nothing is asserted - the bucket is wherever
  the fact actually occurs.

how we get the gold chunk:
  chunk.py already tags every chunk with its doc_id and exact [char_start, char_end) span.
  the gold chunk(s) for a question are the corpus chunks from the right doc whose span
  covers the evidence offset. a question can have >1 gold when chunks overlap; a hit is
  finding ANY gold chunk in the top-k.

the measurement:
  retrieve over the WHOLE corpus (all ~13k chunks from all 12 books), not just the answer's
  own doc - otherwise there's nothing to be lost among and the number is meaningless. for
  each question: embed it, cosine-rank every chunk, check whether a gold chunk landed in the
  top-k. average hit@k per position bucket -> the baseline chart. we also sweep k to show
  the precision/recall tradeoff the spec asks for.

the honest prediction:
  a chunk vector does NOT encode where in its document the text sat - the embedding of a
  paragraph is identical whether it was at 5% or 95%. so a clean cosine retriever is largely
  position-insensitive, and the three buckets may come out roughly FLAT. if they do, that's
  the finding, not a bug: lost-in-the-middle is an attention failure in the READER, not a
  retrieval failure. we report the flatness straight and don't manufacture a U-shape.

    python -m src.benchmark
"""

import json
import re
from pathlib import Path

import numpy as np

from src.chunk import load_config
from src.corpus import build_corpus
from src.retrieve import cosine_sim, precision_at_k, recall_at_k, top_k

ROOT = Path(__file__).resolve().parent.parent
QA_FILE = ROOT / "questions" / "qa.json"
RAW_DIR = ROOT / "data" / "raw"

BUCKETS = ["first", "middle", "last"]


def load_qa():
    return json.loads(QA_FILE.read_text())


def bucket_of(frac):
    """the spec's three windows. anything between them is unlabeled and dropped - we only
    keep questions whose answer sits squarely in a bucket, so the buckets stay clean."""
    if frac < 0.10:
        return "first"
    if 0.40 <= frac <= 0.60:
        return "middle"
    if frac > 0.90:
        return "last"
    return None


def evidence_offset(text, evidence):
    """char offset of the evidence fragment in the raw doc, whitespace-insensitively.

    the raw books wrap lines mid-sentence, so an exact substring find fails on most quotes.
    we match with every run of whitespace treated as flexible (\\s+) - which is legitimate
    because chunk.py's spans come from the same text and the cli collapses whitespace the
    same way, so the fragment we locate is the same text the model actually embeds. returns
    the midpoint char of the match, or None if it doesn't occur (caller drops the question).
    """
    parts = evidence.split()
    pat = r"\s+".join(re.escape(p) for p in parts)
    m = re.search(pat, text)
    if m is None:
        return None
    return (m.start() + m.end()) // 2


def resolve_qa(qa, chunks):
    """attach each question's position bucket and its gold chunk row-indices.

    gold = chunks from the answer's doc whose [char_start, char_end) span covers the
    evidence offset. returns the list of resolved questions (dropping any whose evidence
    doesn't match or whose position falls between buckets), each with 'bucket', 'frac',
    'gold' (list of corpus row indices).
    """
    # group chunk row indices by doc so the gold lookup is a small scan, not a full pass
    by_doc = {}
    for i, c in enumerate(chunks):
        by_doc.setdefault(c["doc_id"], []).append(i)

    resolved = []
    for q in qa:
        text = (RAW_DIR / f"{q['doc_id']}.txt").read_text(encoding="utf-8")
        off = evidence_offset(text, q["evidence"])
        if off is None:
            continue
        frac = off / max(len(text), 1)
        bucket = bucket_of(frac)
        if bucket is None:
            continue
        gold = [i for i in by_doc.get(q["doc_id"], [])
                if chunks[i]["char_start"] <= off < chunks[i]["char_end"]]
        if not gold:
            continue  # no chunk covers the offset (shouldn't happen, but don't fake a gold)
        resolved.append({**q, "bucket": bucket, "frac": frac, "gold": gold})
    return resolved


def _hit(ranked_idx, gold, k):
    """1 if any gold chunk is in the top-k, else 0. multiple gold chunks (from overlap) all
    count as the same target - we care whether the answer's location surfaced, not which of
    its overlapping chunks did."""
    topk = set(ranked_idx[:k].tolist())
    return 1.0 if any(g in topk for g in gold) else 0.0


def run_baseline(k=None, cfg=None):
    """hit@k by position bucket over the whole corpus. this is the baseline chart.

    build the corpus once, embed all questions once, then for each question cosine-rank
    every chunk and check whether a gold chunk landed in the top-k. average per bucket.
    """
    cfg = cfg or load_config()
    k = k or cfg["benchmark"]["k"]

    chunks, vectors, embedder = build_corpus(cfg)
    qa = resolve_qa(load_qa(), chunks)

    qvecs = embedder.encode([q["question"] for q in qa], "query")

    per_bucket = {b: [] for b in BUCKETS}
    for q, qvec in zip(qa, qvecs):
        scores = cosine_sim(qvec, vectors)
        ranked_idx, _ = top_k(scores, k)
        per_bucket[q["bucket"]].append(_hit(ranked_idx, q["gold"], k))

    out = {"k": k, "n": len(qa), "buckets": {}}
    for b in BUCKETS:
        hits = per_bucket[b]
        out["buckets"][b] = {"hit_rate": float(np.mean(hits)) if hits else 0.0, "n": len(hits)}
    return out


def run_k_sweep(ks=(1, 3, 5, 10, 20), cfg=None):
    """precision/recall as k grows, averaged over all questions (the spec's k-sweep).

    with one answer location per question, recall@k is the hit-rate (climbs then plateaus at
    1 as k covers more of the ranking) and precision@k is hits/k (falls, since at most one of
    the k slots is the gold). reported together so the tradeoff is explicit.
    """
    cfg = cfg or load_config()
    chunks, vectors, embedder = build_corpus(cfg)
    qa = resolve_qa(load_qa(), chunks)
    qvecs = embedder.encode([q["question"] for q in qa], "query")

    # rank the full corpus once per question, then read off each k from the same ranking
    full = []
    for qvec in qvecs:
        scores = cosine_sim(qvec, vectors)
        ranked_idx, _ = top_k(scores, len(scores))
        full.append(ranked_idx)

    rows = []
    for k in ks:
        recs, precs = [], []
        for q, ranked in zip(qa, full):
            # collapse multiple gold chunks to the best-ranked one for the metric functions,
            # which take a single gold index. best = earliest gold in the ranking.
            ranks = [np.where(ranked == g)[0] for g in q["gold"]]
            positions = [int(r[0]) for r in ranks if r.size]
            best = ranked[min(positions)] if positions else -1
            recs.append(recall_at_k(ranked, best, k))
            precs.append(precision_at_k(ranked, best, k))
        rows.append({"k": k, "recall": float(np.mean(recs)), "precision": float(np.mean(precs))})
    return {"n": len(qa), "sweep": rows}


def _print_baseline(res):
    print("=" * 60)
    print(f"BASELINE - hit@{res['k']} by answer position ({res['n']} questions)")
    print("=" * 60)
    print(f"  {'bucket':<8} {'n':>4} {'hit@k':>8}")
    for b in BUCKETS:
        s = res["buckets"][b]
        print(f"  {b:<8} {s['n']:>4} {s['hit_rate']:>8.3f}")
    rates = [res["buckets"][b]["hit_rate"] for b in BUCKETS]
    spread = max(rates) - min(rates)
    print(f"\n  spread (max-min across buckets): {spread:.3f}")
    if spread < 0.10:
        print("  ~flat: retrieval is largely position-insensitive here, as expected - a chunk")
        print("  vector doesn't encode where in the doc it came from. lost-in-the-middle bites")
        print("  in the reader's attention, not at this stage. see the readme.")


def _print_sweep(res):
    print("\n" + "-" * 60)
    print(f"K-SWEEP - precision/recall vs k ({res['n']} questions)")
    print("-" * 60)
    print(f"  {'k':>4} {'recall':>8} {'precision':>10}")
    for r in res["sweep"]:
        print(f"  {r['k']:>4} {r['recall']:>8.3f} {r['precision']:>10.3f}")
    print("  recall climbs toward 1 as k grows; precision falls (one gold spread over more")
    print("  slots). the useful k is where recall is high before precision collapses.")


if __name__ == "__main__":
    cfg = load_config()
    baseline = run_baseline(cfg=cfg)
    sweep = run_k_sweep(cfg=cfg)
    _print_baseline(baseline)
    _print_sweep(sweep)

    out = ROOT / "results" / "baseline.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"baseline": baseline, "k_sweep": sweep}, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")
