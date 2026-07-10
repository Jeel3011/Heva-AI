"""
two chunking strategies, both token-based off the actual e5 tokenizer.

    fixed_overlap()   - fixed window of tokens with overlap, cuts wherever it lands
    sentence_packed() - greedily pack whole sentences up to a token budget, never
                        splitting a sentence

everything is measured in TOKENS, not characters, because the embedding model has a
512-token hard cap and truncates silently past it - so the only budget that matters is
the token budget the model actually sees. we keep chunks well under 512 (256 default).

each chunk carries enough to trace it back to the source: doc id, exact char span in the
original text, and its relative % position into the doc (the benchmark's position buckets
use that later). the char span is exact because we get it from the tokenizer's
offset-mapping, not by re-decoding and guessing.

quick look at the output:

    python -m src.chunk
"""

import json
import logging
import re
from pathlib import Path

import yaml
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.json"
CONFIG = ROOT / "config.yaml"


def load_config():
    with open(CONFIG) as f:
        return yaml.safe_load(f)


# one tokenizer, loaded once and reused. it's the e5 tokenizer so the token counts here
# are the exact same ones embed.py will see - no drift between "how big i thought the
# chunk was" and "how big the model thinks it is".
_TOK = None


def get_tokenizer(model_name):
    global _TOK
    if _TOK is None:
        _TOK = AutoTokenizer.from_pretrained(model_name)
        # we tokenize a whole book at once just to grab offsets, then window it into
        # sub-512 chunks - the full-doc call trips hf's "longer than max length" warning
        # even though nothing that big ever reaches the model. quiet that one line.
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    return _TOK


def _chunk_meta(doc_id, text, char_start, char_end, n_tokens, strategy):
    """the trace-back record. rel_pos is the midpoint of the span as a fraction of the
    doc length - one number saying "this chunk lives at ~63% into the doc", which is what
    the position buckets key off."""
    doc_len = max(len(text), 1)
    mid = (char_start + char_end) / 2
    return {
        "doc_id": doc_id,
        "strategy": strategy,
        "char_start": char_start,
        "char_end": char_end,
        "rel_pos": mid / doc_len,
        "n_tokens": n_tokens,
        "text": text[char_start:char_end],
    }


def fixed_overlap(text, doc_id, tokenizer, size=256, overlap=50):
    """slide a fixed window of `size` tokens with `overlap` tokens of carryover.

    we tokenize the whole doc once with offset mapping, so every token knows the char
    span it came from. a window of tokens [i:j] then maps straight back to a char span in
    the original text (start of token i .. end of token j-1). no re-decoding, the span is
    exact. downside vs sentence-packing: the window cuts wherever it lands, often
    mid-sentence, which is an input the model basically never saw in training -> a noisier
    mean-pooled embedding. that's the tradeoff we measure later.
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    n = len(offsets)
    if n == 0:
        return []

    stride = size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than size")

    chunks = []
    for start_tok in range(0, n, stride):
        end_tok = min(start_tok + size, n)
        char_start = offsets[start_tok][0]
        char_end = offsets[end_tok - 1][1]
        chunks.append(
            _chunk_meta(doc_id, text, char_start, char_end, end_tok - start_tok, "fixed")
        )
        if end_tok == n:
            break  # last window already reached the end, don't emit a trailing dupe
    return chunks


# sentence splitter. break after . ! ? when followed by whitespace + a capital/quote/digit.
# the negative lookbehind skips the obvious abbreviation traps (Mr. Dr. etc) so we don't
# shatter a sentence on a title. it's a regex not a parser - fine for public-domain prose,
# and it fails gracefully: a bad split just makes a slightly odd chunk, never a crash.
_ABBREV = r"(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<!\bSt)(?<!\bvs)(?<!\bMt)"
_SENT_SPLIT = re.compile(_ABBREV + r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(])")


def split_sentences(text):
    """return (sentence_text, char_start, char_end) so we keep exact offsets into the doc.
    we split on match positions rather than re.split so the char spans stay honest."""
    spans = []
    prev = 0
    for m in _SENT_SPLIT.finditer(text):
        end = m.start() + 1  # keep the terminal punctuation with the sentence
        seg = text[prev:end]
        if seg.strip():
            spans.append((seg, prev, end))
        prev = m.end()
    tail = text[prev:]
    if tail.strip():
        spans.append((tail, prev, len(text)))
    return spans


def sentence_packed(text, doc_id, tokenizer, budget=256):
    """greedily pack whole sentences into a chunk until the next sentence would blow the
    token budget, then start a fresh chunk. never splits a sentence.

    why bother: e5 was pretrained + contrastively finetuned on coherent spans, so a chunk
    that ends mid-sentence is out-of-distribution and mean-pools into a smeared vector
    across two half-thoughts. keeping subject+predicate together means the mean-pool
    represents one coherent thought -> a cleaner, more faithful embedding for the same
    token budget. same budget as fixed_overlap on purpose, so the tradeoff chart later is
    apples-to-apples (only the boundary policy differs, not the size).
    """
    sents = split_sentences(text)
    if not sents:
        return []

    chunks = []
    cur_start = None
    cur_end = None
    cur_tokens = 0

    def flush():
        if cur_start is not None:
            chunks.append(
                _chunk_meta(doc_id, text, cur_start, cur_end, cur_tokens, "sentence")
            )

    for seg, s_start, s_end in sents:
        # token count of this sentence on its own (no special tokens, same as we count chunks)
        n = len(tokenizer(seg, add_special_tokens=False)["input_ids"])

        # a lone sentence longer than the budget can't be packed with anything - emit it as
        # its own chunk. it'll get truncated at 512 by the model, but that's rare in prose
        # and splitting it would defeat the whole boundary-aware point. flag it honestly.
        if n > budget:
            flush()
            cur_start = None
            chunks.append(
                _chunk_meta(doc_id, text, s_start, s_end, n, "sentence")
            )
            cur_tokens = 0
            continue

        if cur_start is None:
            cur_start, cur_end, cur_tokens = s_start, s_end, n
        elif cur_tokens + n <= budget:
            cur_end, cur_tokens = s_end, cur_tokens + n
        else:
            flush()
            cur_start, cur_end, cur_tokens = s_start, s_end, n

    flush()
    return chunks


def chunk_doc(text, doc_id, tokenizer, cfg):
    """dispatch on the configured strategy so the rest of the pipeline stays agnostic."""
    strategy = cfg["chunk"]["strategy"]
    if strategy == "fixed":
        return fixed_overlap(
            text, doc_id, tokenizer,
            size=cfg["chunk"]["size"], overlap=cfg["chunk"]["overlap"],
        )
    elif strategy == "sentence":
        return sentence_packed(text, doc_id, tokenizer, budget=cfg["chunk"]["size"])
    raise ValueError(f"unknown chunk strategy: {strategy}")


def _sanity():
    """eyeball both strategies on one real book so i can see the chunks look sane."""
    cfg = load_config()
    tok = get_tokenizer(cfg["embed"]["model"])
    manifest = json.loads(MANIFEST.read_text())

    doc = manifest[0]
    text = (RAW_DIR / doc["file"]).read_text(encoding="utf-8")
    print(f"doc: {doc['title']}  ({doc['word_count']:,} words, {len(text):,} chars)\n")

    for fn, label in [
        (lambda: fixed_overlap(text, doc["id"], tok,
                               size=cfg["chunk"]["size"], overlap=cfg["chunk"]["overlap"]),
         "fixed_overlap"),
        (lambda: sentence_packed(text, doc["id"], tok, budget=cfg["chunk"]["size"]),
         "sentence_packed"),
    ]:
        chunks = fn()
        counts = [c["n_tokens"] for c in chunks]
        over = sum(1 for n in counts if n > cfg["embed"]["max_tokens"])
        print(f"[{label}]  {len(chunks)} chunks")
        print(f"  tokens/chunk: min {min(counts)}  max {max(counts)}  "
              f"avg {sum(counts) / len(counts):.0f}")
        print(f"  over 512-cap: {over}")
        # show one chunk from the middle so we can read a real span + its position
        c = chunks[len(chunks) // 2]
        preview = " ".join(c["text"].split())[:160]
        print(f"  sample @ rel_pos {c['rel_pos']:.2f}  chars {c['char_start']}-{c['char_end']}"
              f"  ({c['n_tokens']} tok):")
        print(f"    {preview}...\n")


if __name__ == "__main__":
    _sanity()
