# project notes

my working notes for the heva take-home. not a submission file, just so i dont lose track.

## what the task is

build a retrieval system that shows the lost-in-the-middle problem and then fixes it.
lost-in-the-middle = LLMs/long context pay attention to start and end but miss the
middle. need to understand this at the attention level, not just "its a known thing".

full spec is in `HevaAI AI-ML Engineer Assignment 1.pdf`. deadline is 48h from 2026-07-10.

## must do (checklist)

ingestion + chunking
- [ ] corpus of 10+ docs, each 5000+ words (public domain books / papers / legal)
- [ ] chunking strategy 1: fixed size with overlap
- [ ] chunking strategy 2: boundary aware (sentence / paragraph / semantic)
- [ ] justify boundary choice by how the embedding model reads tokens
- [ ] show chunk tradeoff with a real number (small chunk = more position spread, less meaning)

embedding + retrieval
- [ ] real embedding model, prefer HF (sentence-transformers / e5 / bge)
- [ ] cosine similarity written from scratch, no vector db
- [ ] be able to explain cosine vs dot vs euclidean and when each breaks
- [ ] top-k, k configurable, show precision/recall as k grows

benchmark (the main thing)
- [ ] measures retrieval accuracy vs answer POSITION in the doc
- [ ] 30+ q-a pairs, answer sits in known spot: first 10%, middle 40-60%, last 10%
- [ ] accuracy per position bucket, make a chart -> thats the baseline
- [ ] pick 1 mitigation (position re-rank / interleaving / rrf / my own), justify with math
- [ ] re-run benchmark, report the delta. if middle didnt improve, say why + next step

interface
- [ ] cli: ask question -> answer + chunks with their positions + similarity scores
- [ ] debug mode showing full pipeline

## cannot use
no langchain / llamaindex / any rag framework. no vector db (pinecone, weaviate, chroma,
qdrant). no pre-built retrieval pipeline. every piece mine.
allowed: embedding apis or HF loaders, numpy + normal scientific python for math.

## what to submit
- github repo, clean commits
- deployed demo OR local setup that runs in under 5 min
- README: benchmark design + why valid, chunking + tradeoff, mitigation + math, results
  incl where it failed
- results file / notebook showing benchmark output before AND after mitigation

## interview will grill on
- does the benchmark actually isolate position (the variable it claims to measure)
- derive cosine similarity from first principles, and when it breaks
- what would i do with 2 more weeks
- defend every decision with the mechanics

## style rule (important)
everything i write - code, readme, comments, commits - should read like i wrote it myself.
plain casual english, not polished ai text. no ai attribution in commits.
