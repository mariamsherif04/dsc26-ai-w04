# AI usage

## Tools I used, and for what

I used Claude to write first drafts of all the modules (normaliser, chunker, BM25, hybrid retriever and citation checker). I then reviewed each draft myself and traced the full retrieval pipeline from end to end to make sure it behaved as the spec describes.

## Where AI-written chunking or BM25 code broke one of the spec's rules

I tested the AI's code carefully against the spec, and I found no rule that it broke. The public tests passed and my recall@3 results matched what I expected, so I kept the code as it was.

## My recall@3 table (from EVAL.md) and why hybrid did or did not win

| Retriever | recall@3 | queries missed |
|---|---|---|
| BM25 only | 0.917 (11/12) | q12 |
| Dense only (HashingEmbedder) | 0.917 (11/12) | q12 |
| Hybrid (RRF) | 0.917 (11/12) | q12 |

Hybrid did not win: it only tied BM25 and dense retrieval. All three miss the same query (q12, an English question whose answer is in an Arabic-only clause), and each of them already finds the other 11. Fusion can only help when one retriever finds something the other misses, and on these 12 queries that never happened. The HashingEmbedder rewards shared spelling, so it makes the same mistakes as BM25. A real multilingual embedding model is what I would try next.