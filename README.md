# saudi-dialect-rag

A small, dependency-free retrieval pipeline that finds the right help-centre passage (written in Modern Standard Arabic) for customer questions written in Saudi dialect, plus an evaluation harness that measures each step.

> Portfolio demo. All data is synthetic: "Nakhla Mobile" is a fictional telecom operator and the FAQ passages and questions were written for this repo. It is not production code and has not been used in production.

## Why

Customers rarely type MSA. They write things like «النت مره ثقيل ليش» ("the internet is really slow, why?") while the knowledge base says «إذا كانت سرعة الإنترنت بطيئة…». Plain keyword search, and even many embedding models, miss these matches because of:

1. **Orthographic variation**: أ/إ/آ vs ا, ة vs ه, ى vs ي, diacritics, tatweel, Arabic-Indic digits (١٠٠ vs 100).
2. **Dialect vocabulary**: ابي/ابغى/ودي (I want), وش (what), النت (internet), ثقيل (slow), فلوس (money), الفايبر (fibre).
3. **Morphology**: prefixes (ال، وال، بال) and broken plurals (فرع / فروع).

The retrieval step decides what the LLM sees, so this repo fixes it first and measures the gain.

## Pipeline

```
query ─► normalize ─► dialect→MSA expansion ─► light stem ─┬─► word BM25 ───────┐
                                                            └─► char 3-gram BM25 ┴─► RRF fusion ─► top-k ─► grounded prompt
```

| Module | What it does |
|---|---|
| `saudi_rag/normalize.py` | Orthographic normalization, stopwords, light prefix/suffix stemming |
| `saudi_rag/dialect.py` | Small, auditable Saudi-dialect → MSA lexicon used for query expansion |
| `saudi_rag/bm25.py` | Okapi BM25 written from scratch |
| `saudi_rag/retriever.py` | Four retrieval modes, Reciprocal Rank Fusion, and a grounded-prompt builder |
| `evaluate.py` | Recall@1, Recall@3 and MRR for every mode, with a miss report |
| `ask.py` | CLI: ask in dialect, see the passages and a prompt ready for any chat LLM |

## Quick start

Python 3.10+ and no installs needed.

```bash
python evaluate.py --show-misses     # dev query set
python evaluate.py --heldout         # held-out query set
python ask.py "ابي اشحن رصيدي وش الطريقه"
python -m unittest discover -s tests
```

## Results

18 passages. Numbers come straight from `evaluate.py`.

**Dev set** (20 queries; the dialect lexicon was written while looking at these):

| Mode | Recall@1 | Recall@3 | MRR |
|---|---|---|---|
| raw (whitespace BM25) | 0.25 | 0.40 | 0.39 |
| normalized | 0.45 | 0.70 | 0.60 |
| normalized + dialect | 0.90 | 1.00 | 0.94 |
| hybrid (+ char n-grams, RRF) | 0.95 | 1.00 | 0.97 |

**Held-out set** (12 queries written after the lexicon was frozen):

| Mode | Recall@1 | Recall@3 | MRR |
|---|---|---|---|
| raw (whitespace BM25) | 0.25 | 0.42 | 0.42 |
| normalized | 0.50 | 0.75 | 0.64 |
| normalized + dialect | 0.75 | 0.92 | 0.84 |
| hybrid (+ char n-grams, RRF) | 0.75 | 0.92 | 0.83 |

What this shows:

- Normalization alone roughly doubles Recall@1 on both sets, at zero cost.
- Dialect expansion is the biggest single gain, but the dev-vs-held-out gap (0.90 vs 0.75) is the lexicon overfitting to the questions it was written against. That's why the held-out set exists, and it's the number to trust.
- The remaining held-out misses are vocabulary the lexicon doesn't have yet (دوام for working hours, غلط for wrong, موقف for suspended). The next step is growing the lexicon from real query logs, or adding a multilingual embedding model as a third ranker in the RRF fusion.
- Char n-grams help on the dev set and are neutral on held-out, so they're a cheap safety net rather than a big win at this corpus size.

## Plugging in an LLM

`ask.py` prints a grounded prompt that tells the model to answer in Saudi dialect, use only the retrieved passages, cite passage IDs, and say it doesn't know when the answer isn't there. Send it to any chat model; no API keys are stored in this repo.

## Limitations

- The corpus and query sets are small and synthetic, so the numbers show the method, not a benchmark.
- The light stemmer is rule-based and will over- or under-stem some words.
- The lexicon covers common Najdi/Hijazi/Gulf words only.

## License

MIT
