"""
download the corpus from project gutenberg, strip the boilerplate, save raw text.

we don't commit the books themselves (data/raw is gitignored) - we commit a small
manifest so anyone can rebuild the exact same corpus with one command. run:

    python -m src.ingest
"""

import json
import re
import sys
from pathlib import Path

import requests

# repo root, so this works no matter where it's invoked from
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.json"

MIN_WORDS = 5000

# curated public-domain list, leaning non-fiction / history / essays so the natural
# QA set later has actual facts to ask about (dialogue-heavy fiction is bad for that).
# a few extra over the 10 minimum as buffer in case one comes up short after stripping.
BOOKS = [
    (2130,  "Utopia (Thomas More)"),
    (3207,  "Leviathan (Hobbes)"),
    (3300,  "The Wealth of Nations (Adam Smith)"),
    (1232,  "The Prince (Machiavelli)"),
    (2680,  "Meditations (Marcus Aurelius)"),
    (7370,  "Second Treatise of Government (Locke)"),
    (4280,  "The Critique of Pure Reason (Kant)"),
    (1497,  "The Republic (Plato)"),
    (10615, "The Life of the Bee (Maeterlinck)"),
    (15784, "The Journal of Henry David Thoreau"),
    (3600,  "The Essays of Montaigne"),
    (1998,  "Thus Spake Zarathustra (Nietzsche)"),
]

# plain-text mirror. this url pattern is stable across gutenberg.
URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

# the header/footer markers gutenberg wraps every book in. the wording drifts a little
# between books so match on a regex, not a fixed string. everything between these two is
# the real text - the stuff outside is metadata + the identical license on every book,
# which we absolutely don't want polluting the corpus (it'd make near-dup chunks).
START_RE = re.compile(r"\*\*\*\s*START OF TH(?:IS|E) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
END_RE = re.compile(r"\*\*\*\s*END OF TH(?:IS|E) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)


def strip_boilerplate(text):
    """cut to the text between the start/end markers. falls back to raw if not found."""
    start = START_RE.search(text)
    end = END_RE.search(text)
    if start and end:
        text = text[start.end():end.start()]
    # a few books have a "produced by ..." credit line right after the marker, drop it
    text = re.sub(r"^\s*Produced by .*?\n", "", text, flags=re.I)
    return text.strip()


def word_count(text):
    return len(text.split())


def fetch_one(book_id, title):
    url = URL.format(id=book_id)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    # gutenberg files are utf-8 but declare it inconsistently, force it
    resp.encoding = "utf-8"
    clean = strip_boilerplate(resp.text)
    wc = word_count(clean)
    return clean, wc, url


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for book_id, title in BOOKS:
        print(f"fetching {book_id} - {title} ...", end=" ", flush=True)
        try:
            text, wc, url = fetch_one(book_id, title)
        except requests.RequestException as e:
            print(f"FAILED ({e})")
            continue

        if wc < MIN_WORDS:
            # don't silently keep a thin doc, the spec needs >=5000 each
            print(f"SKIPPED (only {wc} words after stripping)")
            continue

        fname = f"{book_id}.txt"
        (RAW_DIR / fname).write_text(text, encoding="utf-8")
        manifest.append({
            "id": book_id,
            "title": title,
            "source_url": url,
            "word_count": wc,
            "file": fname,
        })
        print(f"ok ({wc} words)")

    if len(manifest) < 10:
        # hard stop - the corpus requirement isn't met, better to know now
        sys.exit(f"\nonly got {len(manifest)} usable books, need >=10. check the failures above.")

    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total = sum(m["word_count"] for m in manifest)
    print(f"\ndone. {len(manifest)} books, {total:,} words total. manifest -> {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
