"""
embed text with e5 (bge swappable). the two things that actually matter here:

  1. the query/passage prefixes. e5 is an ASYMMETRIC retriever - questions get "query: "
     and chunks get "passage: ". question text and passage text come from different
     distributions; the prefix tells the model which encoder mode to use so the two land
     in a comparable region of the space. get this wrong and the cosines are garbage.
  2. mean pooling over the token hidden states, MASKED so padding doesn't drag the
     average. mean pooling is why a needle sentence buried in a big chunk gets diluted by
     the surrounding filler - that dilution is the chunking tradeoff we demonstrate later,
     so we do the pooling by hand instead of hiding it behind a black-box .encode().

we L2-normalize every vector on the way out (e5 trained with a normalized objective, and
it makes cosine downstream a plain matmul), and cache to .npy so reruns are instant and
the whole benchmark stays under the 5-min budget.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"
CONFIG = ROOT / "config.yaml"


def load_config():
    with open(CONFIG) as f:
        return yaml.safe_load(f)


# prefix logic lives behind one function so swapping e5 -> bge is a config change, not a
# rewrite. e5 prefixes both sides; bge only prefixes the query (its passages go bare).
def add_prefix(texts, kind, model_name):
    name = model_name.lower()
    if "e5" in name:
        tag = "query: " if kind == "query" else "passage: "
        return [tag + t for t in texts]
    if "bge" in name:
        if kind == "query":
            return ["Represent this sentence for searching relevant passages: " + t for t in texts]
        return list(texts)  # bge passages get no prefix
    # unknown model - don't invent a prefix, just pass through and let cosine speak
    return list(texts)


class Embedder:
    def __init__(self, cfg=None):
        cfg = cfg or load_config()
        self.model_name = cfg["embed"]["model"]
        self.max_tokens = cfg["embed"]["max_tokens"]
        self.batch_size = cfg["embed"]["batch_size"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()  # no dropout, deterministic

    def _mean_pool(self, hidden, mask):
        # hidden: (batch, seq, dim), mask: (batch, seq). zero out padding tokens then
        # divide by the real token count, so padding contributes nothing to the average.
        mask = mask.unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)  # guard against an all-pad row
        return summed / counts

    @torch.no_grad()
    def _encode_batch(self, texts):
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_tokens,  # 512 hard cap, anything longer is truncated here
            return_tensors="pt",
        )
        out = self.model(**enc)
        pooled = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
        # L2-normalize: cosine == dot of unit vectors, and e5 was trained normalized
        return torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()

    def encode(self, texts, kind):
        """kind is 'query' or 'passage'. returns (n, dim) float32, unit-normalized."""
        texts = add_prefix(texts, kind, self.model_name)
        vecs = []
        for i in range(0, len(texts), self.batch_size):
            vecs.append(self._encode_batch(texts[i:i + self.batch_size]))
        return np.vstack(vecs).astype(np.float32)


def _cache_key(texts, model_name, kind):
    # hash the exact inputs that change the output: model, side, and the text itself.
    # any edit to the corpus/chunking busts the cache automatically - no stale vectors.
    h = hashlib.sha256()
    h.update(model_name.encode())
    h.update(kind.encode())
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def encode_cached(texts, kind, embedder):
    """same as encode() but memoized to .npy on disk. the embed step is the slow part of
    the pipeline, so caching is what keeps reruns under the 5-min budget."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(texts, embedder.model_name, kind)
    path = CACHE_DIR / f"emb_{key}.npy"
    if path.exists():
        return np.load(path)
    vecs = embedder.encode(texts, kind)
    np.save(path, vecs)
    return vecs


def _sanity():
    """embed a couple of real chunks + a matching query and check the numbers behave:
    unit norm, right shape, and the query is closer to the on-topic chunk than an off-topic
    one. this is the smoke test that the prefixes + pooling are wired correctly."""
    from src.chunk import chunk_doc, get_tokenizer

    cfg = load_config()
    emb = Embedder(cfg)

    passages = [
        "The bees leave the hive in a great swarm to found a new colony in the spring.",
        "The wealth of a nation consists in the annual produce of its land and labour.",
    ]
    query = "why do bees swarm and leave the hive"

    p = encode_cached(passages, "passage", emb)
    q = emb.encode([query], "query")

    print(f"passage matrix: {p.shape}   query: {q.shape}")
    print(f"passage norms (should be ~1.0): {np.linalg.norm(p, axis=1).round(4)}")

    sims = (q @ p.T)[0]
    print(f"cosine to each passage: {sims.round(4)}")
    print(f"closest -> passage[{int(sims.argmax())}] "
          f"(expect 0, the bee one): '{passages[int(sims.argmax())][:50]}...'")


if __name__ == "__main__":
    _sanity()
