"""
retrieval math, from scratch in numpy. no vector DB, no sklearn - the whole point of this
module is that similarity is a matmul once you've normalized, and i want to be able to
derive every line of it on a whiteboard.

    cosine_sim(q, M)   - cosine of a query against every row of a matrix
    top_k(scores, k)   - the k highest scores + their indices, without a full sort
    recall_at_k / precision_at_k - the metrics the K-sweep chart reads

why cosine and not raw dot or euclidean:
  - dot product lets magnitude leak into the score. two chunks can point the same
    direction but a longer one wins purely on length - that's ranking by verbosity, not
    relevance. cosine divides magnitude out and compares direction only.
  - euclidean and cosine are the SAME ranking once vectors are unit-normalized:
    ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b = 2 - 2cos(theta), so nearer == more similar.
    so we don't need a second distance, we just normalize and take the dot.
  - cosine isn't a true metric (no triangle inequality), which is fine - we only ever
    rank by it, never reason about distances between distances.

e5 already hands us unit vectors, so cosine_sim is technically doing a redundant
normalize. i keep it anyway: it makes this function correct on its own inputs instead of
silently trusting the caller, and normalizing an already-unit vector is a no-op numerically.
"""

import numpy as np


def _l2_normalize(M, eps=1e-12):
    """scale each row to unit L2 norm. the eps in the denominator is the whole trick for
    the degenerate case: a near-zero vector (all-pad chunk, empty string) has norm ~0, and
    dividing by it would give inf/nan and poison every downstream score. clamping the norm
    up to eps leaves such a vector at ~0 magnitude -> it just scores ~0 against everything
    and quietly loses the ranking, which is exactly what we want a junk vector to do."""
    M = np.asarray(M, dtype=np.float64)
    norms = np.linalg.norm(M, axis=-1, keepdims=True)
    return M / np.maximum(norms, eps)


def cosine_sim(q, M):
    """cosine similarity of query vector(s) q against every row of matrix M.

    q: (dim,) or (n_q, dim).  M: (n_docs, dim).
    returns (n_docs,) for a single query, or (n_q, n_docs) for a batch.

    the entire method: normalize both sides to the unit sphere, then similarity is just
    the dot product q_hat @ M_hat.T. cos(theta) = a.b / (||a|| ||b||), and if ||a||=||b||=1
    the denominator is 1 and cosine collapses to the plain dot. that's it - no library call.
    """
    q = np.asarray(q, dtype=np.float64)
    single = q.ndim == 1
    if single:
        q = q[None, :]  # (1, dim) so the matmul path is uniform

    q_hat = _l2_normalize(q)
    M_hat = _l2_normalize(M)

    sims = q_hat @ M_hat.T  # (n_q, n_docs)
    return sims[0] if single else sims


def top_k(scores, k):
    """indices + scores of the k highest entries, best first.

    argpartition does an O(n) partial sort that just guarantees the top-k land in the last
    k slots (unordered), which is all we need before a final tiny sort of those k. that
    beats argsort-ing the whole score vector when n_docs >> k - the usual retrieval case,
    thousands of chunks and k=6. we only pay a full sort on the k we keep.
    """
    scores = np.asarray(scores)
    n = scores.shape[0]
    k = min(k, n)  # asking for more than we have just returns everything, ranked

    # -scores so the LARGEST scores go to the front partition. [:k] grabs that unordered
    # top-k block, then argsort orders those k (and only those k) descending.
    part = np.argpartition(-scores, k - 1)[:k]
    order = np.argsort(-scores[part])
    idx = part[order]
    return idx, scores[idx]


def recall_at_k(ranked_idx, gold_idx, k):
    """single gold chunk per question, so recall@k is a hit-rate: 1 if the gold chunk is
    anywhere in the top-k, else 0. this is the curve that shows lost-in-the-middle - it
    climbs slower for gold chunks buried mid-document than for ones near the edges."""
    return 1.0 if gold_idx in ranked_idx[:k] else 0.0


def precision_at_k(ranked_idx, gold_idx, k):
    """with exactly one relevant chunk, at most one of the top-k can be a hit, so
    precision@k is either 0 or 1/k. it necessarily falls as k grows (one hit spread over
    more slots) - reported alongside recall to make the precision/recall tradeoff explicit
    rather than cherry-picking whichever k flatters the system."""
    hits = 1.0 if gold_idx in ranked_idx[:k] else 0.0
    return hits / k


def _sanity():
    """toy vectors with a ranking i can verify by eye, so this proves correctness, not just
    that it runs. no model, no data - pure numpy asserts."""
    # four docs on a 2D plane. query points straight along +x.
    M = np.array([
        [1.0, 0.0],    # 0: identical direction to the query      -> cos 1.0
        [0.9, 0.1],    # 1: almost the query                      -> high
        [0.0, 1.0],    # 2: orthogonal                            -> cos 0.0
        [-1.0, 0.0],   # 3: opposite                              -> cos -1.0
    ])
    q = np.array([2.0, 0.0])  # deliberately NON-unit: magnitude 2 must not change ranking

    sims = cosine_sim(q, M)

    # a query identical in direction to row 0 must rank it first, at cosine 1.0
    assert np.isclose(sims[0], 1.0), sims
    assert np.isclose(sims[2], 0.0), sims       # orthogonal -> 0
    assert np.isclose(sims[3], -1.0), sims      # antiparallel -> -1
    assert int(sims.argmax()) == 0, sims        # magnitude-2 query still ranks by direction

    # cosine of unit vectors == the plain normalized dot product (the identity we rely on)
    a = np.array([3.0, 4.0])  # norm 5
    b = np.array([4.0, 3.0])  # norm 5
    manual = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert np.isclose(cosine_sim(a, b[None, :])[0], manual), (cosine_sim(a, b[None, :]), manual)

    # near-zero vector must not blow up - it should score ~0 and lose, not nan the row
    Mz = np.vstack([M, [1e-20, 1e-20]])
    sims_z = cosine_sim(q, Mz)
    assert np.all(np.isfinite(sims_z)), sims_z
    assert np.isclose(sims_z[-1], 0.0, atol=1e-6), sims_z[-1]

    # top_k: order is best-first, and it returns k items even when n >> k
    idx, sc = top_k(sims, k=2)
    assert list(idx) == [0, 1], idx              # row 0 then row 1
    assert sc[0] >= sc[1], sc                    # scores come back descending
    # asking for more than we have clamps to n, doesn't crash
    idx_all, _ = top_k(sims, k=99)
    assert len(idx_all) == len(M), idx_all

    # metrics: gold in the top-k vs not
    ranked, _ = top_k(sims, k=len(M))
    assert recall_at_k(ranked, gold_idx=0, k=1) == 1.0      # gold is rank 1 -> hit at k=1
    assert recall_at_k(ranked, gold_idx=2, k=1) == 0.0      # gold is rank 3 -> miss at k=1
    assert recall_at_k(ranked, gold_idx=2, k=3) == 1.0      # ...but caught by k=3
    assert np.isclose(precision_at_k(ranked, gold_idx=0, k=1), 1.0)
    assert np.isclose(precision_at_k(ranked, gold_idx=0, k=4), 0.25)  # one hit over 4 slots

    print("retrieve.py self-test passed")
    print(f"  cosine(q, M) = {sims.round(3)}")
    print(f"  top-2 idx    = {list(idx)}  scores = {sc.round(3)}")


if __name__ == "__main__":
    _sanity()
