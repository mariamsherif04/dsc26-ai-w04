# Retrieval evaluation

Setup: all 8 contracts in `data/contracts/`, `chunk_text(..., max_chars=500, overlap=50)`,
`HashingEmbedder()`, the 12 queries in `data/queries.jsonl`. A hit is relevant when its span
overlaps one of the query's `relevant` ranges in the same document.

| Retriever | recall@3 | queries missed |
|---|---|---|
| BM25 only | 0.917 (11/12) | q12 |
| Dense only (HashingEmbedder) | 0.917 (11/12) | q12 |
| Hybrid (RRF) | 0.917 (11/12) | q12 |

All three miss q12, an English query about governing law and competent court for the agency agreement, whose answer is in an Arabic-only clause of a bilingual contract. The retrieved chunks are English governing-law clauses from other contracts, so neither word matching nor character 3-grams can bridge the language gap. I would try a real multilingual embedding model next (and contextual chunk headers or a re-ranker).