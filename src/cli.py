"""
one runnable query pass over the whole corpus.

    python -m src.cli "why does a prince need to imitate the fox and the lion"
    python -m src.cli "what is the wealth of a nation" --k 10
    python -m src.cli "who kept the oracle at dodona" --debug

wires chunk -> embed -> retrieve into a single call: build the corpus (all 12 books,
chunked and embedded, cached so this is fast after the first run), embed the query with the
same model, cosine-rank every chunk, and print an answer plus the top-k. for each hit we
show where it came from - source title, how far into that doc it sits (rel_pos as a %), its
exact char span, and the cosine score - so a retrieved chunk is always traceable back to a
specific place in a specific book.

the "answer" is deliberately simple: no generator model, just the single best-matching
SENTENCE from the top chunk (extractive). the assignment is about retrieval fidelity, not
generation, so we surface the most relevant span rather than synthesize prose - and it stays
honest, you can always see the chunk the sentence came from right below it.

--debug dumps the full pipeline for a query: the exact prefixed string the model embedded,
and the ranked top-k with raw scores in the order retrieval chose. k comes from config
(retrieval.k), overridable with --k.
"""

import argparse

from src.chunk import load_config, split_sentences
from src.corpus import build_corpus
from src.embed import add_prefix
from src.retrieve import cosine_sim, top_k


def retrieve(question, k=None, cfg=None, _cache={}):
    """return the top-k chunk records for a question, each annotated with its cosine score.

    the join back to metadata is the point: top_k hands us row indices into the vector
    matrix, and because the corpus was built in one ordered pass, row i is chunks[i]. so we
    look each index straight up in the chunk list - no id lookup, no re-chunking. the corpus
    is memoized in _cache so --debug can reuse it without rebuilding.
    """
    cfg = cfg or load_config()
    k = k or cfg["retrieval"]["k"]

    if "corpus" not in _cache:
        _cache["corpus"] = build_corpus(cfg)
    chunks, vectors, embedder = _cache["corpus"]

    qvec = embedder.encode([question], "query")[0]
    scores = cosine_sim(qvec, vectors)
    idx, sc = top_k(scores, k)

    hits = []
    for rank, (i, score) in enumerate(zip(idx, sc), start=1):
        hit = dict(chunks[int(i)])
        hit["rank"] = rank
        hit["score"] = float(score)
        hits.append(hit)
    return hits, embedder


def best_sentence(question, chunk_text, embedder):
    """extractive answer: the sentence in the chunk most similar to the question.

    split the top chunk into sentences, embed them as passages (same prefix the corpus used),
    and return the one with the highest cosine to the query. this is the smallest honest
    'answer' - it's literally text from the source, and if the chunk is one sentence it just
    returns that. no synthesis, nothing that can hallucinate.
    """
    sents = [s for s, _, _ in split_sentences(chunk_text)]
    sents = [" ".join(s.split()) for s in sents if s.strip()]
    if not sents:
        return " ".join(chunk_text.split())
    if len(sents) == 1:
        return sents[0]
    qvec = embedder.encode([question], "query")[0]
    svecs = embedder.encode(sents, "passage")
    sims = cosine_sim(qvec, svecs)
    return sents[int(sims.argmax())]


def _snippet(text, width=200):
    """collapse whitespace so a chunk spanning newlines prints as one readable line."""
    return " ".join(text.split())[:width]


def print_hits(question, hits, answer):
    print(f'\nquery: "{question}"')
    print(f"\nanswer: {answer}\n")
    print(f"top {len(hits)} chunks across the corpus:\n")
    for h in hits:
        # rel_pos is a fraction of the way into its source doc; show it as a % because
        # "63% in" is the lost-in-the-middle signal we actually care about.
        print(f"  #{h['rank']}  cosine {h['score']:.4f}   {h['title']}")
        print(f"      position {h['rel_pos'] * 100:5.1f}% into doc"
              f"   chars {h['char_start']}-{h['char_end']}   ({h['n_tokens']} tok)")
        print(f"      {_snippet(h['text'])}\n")


def print_debug(question, hits, cfg):
    """the full retrieval pipeline for one query, so a reviewer can see every step."""
    model = cfg["embed"]["model"]
    prefixed = add_prefix([question], "query", model)[0]
    print("\n" + "=" * 60)
    print("DEBUG - retrieval pipeline")
    print("=" * 60)
    print(f"  model:          {model}")
    print(f"  raw query:      {question!r}")
    print(f"  embedded as:    {prefixed!r}")
    print(f"  (e5 is asymmetric: the 'query: ' prefix puts the question in the query")
    print(f"   sub-space so its cosine against 'passage: ' chunks is meaningful.)")
    print(f"\n  cosine-ranked top {len(hits)} of the whole corpus, in retrieval order:")
    print(f"  {'rank':>4} {'cosine':>8}  {'pos%':>5}  doc / span")
    for h in hits:
        print(f"  {h['rank']:>4} {h['score']:>8.4f}  {h['rel_pos'] * 100:>4.0f}%  "
              f"{h['title']}  [{h['char_start']}-{h['char_end']}]")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="retrieve chunks for a question over the corpus")
    ap.add_argument("question", help="the query to retrieve for")
    ap.add_argument("--k", type=int, default=None, help="how many chunks to return (default: config)")
    ap.add_argument("--debug", action="store_true", help="show the full retrieval pipeline")
    args = ap.parse_args()

    cfg = load_config()
    hits, embedder = retrieve(args.question, k=args.k, cfg=cfg)
    answer = best_sentence(args.question, hits[0]["text"], embedder)

    print_hits(args.question, hits, answer)
    if args.debug:
        print_debug(args.question, hits, cfg)


if __name__ == "__main__":
    main()
