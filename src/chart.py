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
    mit = json.loads((RESULTS / "mitigation.json").read_text())["mitigation"]

    base_rates = [base["buckets"][b]["hit_rate"] for b in BUCKETS]
    mit_rates = [mit["buckets"][b]["hit_rate"] for b in BUCKETS]
    x = range(len(BUCKETS))
    w = 0.38

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar([i - w / 2 for i in x], base_rates, w, label="baseline (fixed)", color="#4c72b0")
    ax.bar([i + w / 2 for i in x], mit_rates, w, label="RRF (fixed+sentence)", color="#dd8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{b}\n(n={base['buckets'][b]['n']})" for b in BUCKETS])
    ax.set_ylabel(f"hit@{base['k']}")
    ax.set_ylim(0, 1)
    ax.set_title("retrieval hit-rate by answer position\n(flat = retrieval is position-insensitive)")
    ax.legend()
    for i, (bv, mv) in enumerate(zip(base_rates, mit_rates)):
        ax.text(i - w / 2, bv + 0.02, f"{bv:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, mv + 0.02, f"{mv:.2f}", ha="center", fontsize=8)
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
