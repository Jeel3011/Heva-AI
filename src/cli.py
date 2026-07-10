"""
one runnable query pass over the whole corpus.

    python -m src.cli "why do bees swarm and leave the hive"
    python -m src.cli "what is the wealth of a nation" --k 10

wires chunk -> embed -> retrieve into a single call: build the corpus (all 12 books,
chunked and embedded, cached so this is fast after the first run), embed the query with
the same model, cosine-rank every chunk, and print the top-k. for each hit we show where
it came from - source title, how far into that doc it sits (rel_pos as a %), its exact
char span, and the cosine score - so a retrieved chunk is always traceable back to a
specific place in a specific book.

no reader here. this is retrieval only; generating an answer from these chunks is step 8.
k comes from config.yaml (retrieval.k) and is overridable with --k.
"""

import argparse

from src.chunk import load_config
from src.corpus import build_corpus
from src.retrieve import cosine_sim, top_k


def retrieve(question, k=None, cfg=None):
    """return the top-k chunk records for a question, each annotated with its cosine score.

    the join back to metadata is the point: top_k hands us row indices into the vector
    matrix, and because the corpus was built in one ordered pass, row i is chunks[i]. so we
    look each index straight up in the chunk list - no id lookup, no re-chunking.
    """
    cfg = cfg or load_config()
    k = k or cfg["retrieval"]["k"]

    chunks, vectors, embedder = build_corpus(cfg)
    qvec = embedder.encode([question], "query")[0]

    scores = cosine_sim(qvec, vectors)
    idx, sc = top_k(scores, k)

    hits = []
    for rank, (i, score) in enumerate(zip(idx, sc), start=1):
        hit = dict(chunks[int(i)])
        hit["rank"] = rank
        hit["score"] = float(score)
        hits.append(hit)
    return hits


def _snippet(text, width=200):
    """collapse whitespace so a chunk spanning newlines prints as one readable line."""
    return " ".join(text.split())[:width]


def print_hits(question, hits):
    print(f'\nquery: "{question}"')
    print(f"top {len(hits)} chunks across the corpus:\n")
    for h in hits:
        # rel_pos is a fraction of the way into its source doc; show it as a % because
        # "63% in" is the lost-in-the-middle signal we actually care about.
        print(f"  #{h['rank']}  cosine {h['score']:.4f}   {h['title']}")
        print(f"      position {h['rel_pos'] * 100:5.1f}% into doc"
              f"   chars {h['char_start']}-{h['char_end']}   ({h['n_tokens']} tok)")
        print(f"      {_snippet(h['text'])}\n")


def main():
    ap = argparse.ArgumentParser(description="retrieve chunks for a question over the corpus")
    ap.add_argument("question", help="the query to retrieve for")
    ap.add_argument("--k", type=int, default=None, help="how many chunks to return (default: config)")
    args = ap.parse_args()

    hits = retrieve(args.question, k=args.k)
    print_hits(args.question, hits)


if __name__ == "__main__":
    main()
