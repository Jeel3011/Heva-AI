"""
the chunking tradeoff, measured - not asserted.

the spec asks to show the tradeoff empirically with at least one data point. the two
strategies in chunk.py differ in one thing: where a chunk is allowed to end. fixed_overlap
cuts wherever the token window lands (often mid-sentence -> a mean-pooled vector smeared
across two half-thoughts); sentence_packed never splits a sentence (one coherent thought per
vector). same 256-token budget on both, on purpose, so the only variable is the boundary
policy, not the size.

we run the same 36-question benchmark (hit@k over the whole corpus) under each strategy and
report the delta. both corpora are already embedded and cached from the benchmark run, so
this is fast - it does NOT re-embed anything.

the tradeoff the spec names - smaller/edge-cut chunks buy positional diversity but lose
semantic completeness - shows up here as: does keeping sentences whole (semantic
completeness) actually help retrieval, or does the edge-cutting not hurt enough to matter on
this corpus? the number below answers that for real instead of asserting it.

    python -m src.chunk_tradeoff
"""

import copy
import json
from pathlib import Path

from src.benchmark import run_baseline
from src.chunk import load_config

ROOT = Path(__file__).resolve().parent.parent
STRATEGIES = ["fixed", "sentence"]


def _overall(res):
    """single hit-rate across all questions, weighting each bucket by its question count so a
    small bucket doesn't swing it. run_baseline reports per-bucket; recombine to one number."""
    total_hits = sum(b["hit_rate"] * b["n"] for b in res["buckets"].values())
    total_n = sum(b["n"] for b in res["buckets"].values())
    return total_hits / total_n if total_n else 0.0


def run_tradeoff(cfg=None):
    """fixed vs sentence at the same token budget - isolates boundary policy.

    deep-copy the config and flip only chunk.strategy. build_corpus keys its chunk cache on
    the strategy and embed keys its .npy cache on the chunk texts, so each strategy uses its
    own already-built corpus - no re-embedding, no stale vectors leaking across.
    """
    cfg = cfg or load_config()
    out = {}
    for strat in STRATEGIES:
        c = copy.deepcopy(cfg)
        c["chunk"]["strategy"] = strat
        res = run_baseline(cfg=c)
        out[strat] = {"hit_rate": _overall(res), "buckets": res["buckets"]}
    return {
        "size": cfg["chunk"]["size"],
        "k": cfg["benchmark"]["k"],
        "by_strategy": out,
        "delta_sentence_minus_fixed": out["sentence"]["hit_rate"] - out["fixed"]["hit_rate"],
    }


def _print_report(t):
    print("=" * 60)
    print(f"CHUNKING TRADEOFF - fixed vs sentence at {t['size']} tokens")
    print(f"(36-question benchmark, hit@{t['k']} over the whole corpus)")
    print("=" * 60)
    print(f"  {'strategy':<10} {'hit_rate':>9}")
    for s in STRATEGIES:
        print(f"  {s:<10} {t['by_strategy'][s]['hit_rate']:>9.3f}")
    d = t["delta_sentence_minus_fixed"]
    winner = "sentence" if d > 0 else ("fixed" if d < 0 else "tie")
    print(f"\n  delta (sentence - fixed): {d:+.3f}  ->  {winner}")
    print("  same 256-token budget, so this isolates BOUNDARY POLICY. fixed cuts mid-sentence")
    print("  -> a vector smeared across two half-thoughts; sentence keeps one coherent thought")
    print("  per vector. the delta says whether that coherence buys retrieval on this corpus.")


if __name__ == "__main__":
    cfg = load_config()
    t = run_tradeoff(cfg)
    _print_report(t)

    out = ROOT / "results" / "chunk_tradeoff.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(t, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")
