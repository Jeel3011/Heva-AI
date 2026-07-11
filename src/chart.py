"""
draw the two charts the spec asks for, straight from the results json (no model, no embed):

  1. hit@k by answer position bucket - baseline vs RRF. this is THE chart: the flat line
     across first/middle/last is the finding (retrieval is position-insensitive), and the
     baseline/rrf bars sitting on top of each other is the honest mitigation result.
  2. precision/recall vs k. the tradeoff: recall climbs, precision falls as k grows.

run the benchmark + mitigation first so the json exists, then:

    python -m src.chart
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed, just write png files
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
BUCKETS = ["first", "middle", "last"]


def position_chart():
    base = json.loads((RESULTS / "baseline.json").read_text())["baseline"]
    rrf = json.loads((RESULTS / "mitigation.json").read_text())["mitigation"]
    rer = json.loads((RESULTS / "rerank.json").read_text())["rerank"]

    base_rates = [base["buckets"][b]["hit_rate"] for b in BUCKETS]
    rrf_rates = [rrf["buckets"][b]["hit_rate"] for b in BUCKETS]
    rer_rates = [rer["buckets"][b]["hit_rate"] for b in BUCKETS]
    x = range(len(BUCKETS))
    w = 0.27

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar([i - w for i in x], base_rates, w, label="baseline (pooled cosine)", color="#4c72b0")
    ax.bar(list(x), rrf_rates, w, label="RRF fusion (no gain)", color="#dd8452")
    ax.bar([i + w for i in x], rer_rates, w, label="MaxSim re-rank", color="#55a868")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{b}\n(n={base['buckets'][b]['n']})" for b in BUCKETS])
    ax.set_ylabel(f"hit@{base['k']}")
    ax.set_ylim(0, 1.08)
    ax.set_title("retrieval hit-rate by answer position\nMaxSim re-rank lifts every bucket, middle included")
    ax.legend(loc="lower right", fontsize=8)
    for i, (bv, fv, rv) in enumerate(zip(base_rates, rrf_rates, rer_rates)):
        ax.text(i - w, bv + 0.02, f"{bv:.2f}", ha="center", fontsize=7)
        ax.text(i, fv + 0.02, f"{fv:.2f}", ha="center", fontsize=7)
        ax.text(i + w, rv + 0.02, f"{rv:.2f}", ha="center", fontsize=7)
    fig.tight_layout()
    out = RESULTS / "position_chart.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def pr_chart():
    sweep = json.loads((RESULTS / "baseline.json").read_text())["k_sweep"]["sweep"]
    ks = [r["k"] for r in sweep]
    rec = [r["recall"] for r in sweep]
    prec = [r["precision"] for r in sweep]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ks, rec, "o-", label="recall@k", color="#4c72b0")
    ax.plot(ks, prec, "s-", label="precision@k", color="#c44e52")
    ax.set_xlabel("k")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_title("precision / recall vs k\n(recall climbs, precision falls as k grows)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS / "pr_curve.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


if __name__ == "__main__":
    p1 = position_chart()
    p2 = pr_chart()
    print(f"wrote {p1.relative_to(ROOT)}")
    print(f"wrote {p2.relative_to(ROOT)}")
