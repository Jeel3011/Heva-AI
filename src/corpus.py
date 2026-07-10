"""
build the retrievable corpus: every book, chunked once, embedded once, held as one flat
list of chunks paired row-for-row with one vector matrix.

the whole system retrieves across all 12 books at once, not one at a time, so the unit of
work here is the *corpus*, not the document. we chunk each book, concatenate the chunks
into a single list, embed them all, and from then on a chunk is just an integer row index
- index i in the chunk list is row i in the vector matrix. that pairing is the only thing
holding retrieval together, so we build both from the same ordered pass and never reorder.

two caches keep the cli fast after the first run:
  - the chunk list -> data/cache/chunks_<hash>.json, so we don't re-chunk 2M words on
    every query (that's ~0.9s of cold start). keyed on the tokenizer + strategy + sizes +
    the manifest, so editing the corpus or the chunking config busts it automatically.
  - the passage vectors -> the .npy cache that embed.encode_cached already owns.

the model load itself is the one cost we can't cache away; everything else amortizes to
near zero on a warm run.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from src.chunk import chunk_doc, get_tokenizer, load_config
from src.embed import Embedder, encode_cached

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "cache"
MANIFEST = ROOT / "data" / "manifest.json"


def _chunks_key(manifest, cfg):
    # the inputs that change the chunk list: which model's tokenizer measures the budget,
    # the strategy and its sizes, and the corpus itself (id + word_count per book catches
    # a re-ingest or a swapped book). same content-addressed discipline as the .npy cache.
    h = hashlib.sha256()
    h.update(cfg["embed"]["model"].encode())
    h.update(str(cfg["chunk"]["strategy"]).encode())
    h.update(str(cfg["chunk"]["size"]).encode())
    h.update(str(cfg["chunk"]["overlap"]).encode())
    for doc in manifest:
        h.update(str(doc["id"]).encode())
        h.update(str(doc["word_count"]).encode())
    return h.hexdigest()[:16]


def build_chunks(cfg=None):
    """chunk every book in the manifest into one flat, ordered list of chunk records.

    cached to data/cache/chunks_<hash>.json. a chunk record already carries doc_id, the
    exact char span, rel_pos, and the text - everything the cli needs to trace a hit back
    to its source, so retrieval never has to reopen data/raw.
    """
    cfg = cfg or load_config()
    manifest = json.loads(MANIFEST.read_text())

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"chunks_{_chunks_key(manifest, cfg)}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    tok = get_tokenizer(cfg["embed"]["model"])
    chunks = []
    for doc in manifest:
        text = (RAW_DIR / doc["file"]).read_text(encoding="utf-8")
        for c in chunk_doc(text, doc["id"], tok, cfg):
            # carry the title through so the cli can print it without re-reading the
            # manifest; doc_id stays the join key of record.
            c["title"] = doc["title"]
            chunks.append(c)

    cache_path.write_text(json.dumps(chunks))
    return chunks


def build_corpus(cfg=None, embedder=None):
    """the full retrievable corpus: (chunks, vectors, embedder).

    chunks[i] is described by vectors[i] - one ordered build, no reordering, so the row
    index is the chunk's identity everywhere downstream. returns the embedder too so the
    caller can reuse the loaded model to encode the query instead of loading it twice.
    """
    cfg = cfg or load_config()
    chunks = build_chunks(cfg)
    embedder = embedder or Embedder(cfg)

    texts = [c["text"] for c in chunks]
    vectors = encode_cached(texts, "passage", embedder)
    return chunks, np.asarray(vectors), embedder


if __name__ == "__main__":
    # quick look: how big is the corpus we retrieve over, and does the pairing line up.
    chunks, vectors, _ = build_corpus()
    n_docs = len({c["doc_id"] for c in chunks})
    print(f"{len(chunks)} chunks over {n_docs} docs -> vectors {vectors.shape}")
    assert len(chunks) == vectors.shape[0], "chunk/vector row mismatch"
    print("row pairing holds (chunks == vector rows)")
