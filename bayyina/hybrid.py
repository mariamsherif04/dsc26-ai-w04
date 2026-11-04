"""Reciprocal rank fusion and the hybrid retriever."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from bayyina.bm25 import BM25Index
from bayyina.types import Chunk, Embedder, Hit, Reranker


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Ranks start at 1; score = sum of 1 / (k + rank); by score descending, then ID ascending."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        rank = 0
        for cid in ranking:
            if cid in seen:                         # a duplicate inside one list counts once
                continue
            seen.add(cid)
            rank += 1
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda t: (-t[1], t[0]))


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, *, k1: float = 1.5,
                 b: float = 0.75, reranker: Reranker | None = None):
        self.chunks = list(chunks)
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.embedder = embedder
        self.reranker = reranker
        self.bm25 = BM25Index(k1=k1, b=b)
        self.bm25.add(self.chunks)
        self.ids = [c.chunk_id for c in self.chunks]
        self.matrix = (np.asarray(embedder.embed([c.text for c in self.chunks]), dtype=float)
                       if self.chunks else np.zeros((0, 0)))

    def _dense(self, query: str, candidates: int) -> list[str]:
        if not self.chunks:
            return []
        q = np.asarray(self.embedder.embed([query]), dtype=float)[0]
        sims = self.matrix @ q
        order = sorted(range(len(self.ids)), key=lambda i: (-float(sims[i]), self.ids[i]))
        return [self.ids[i] for i in order[:candidates]]

    def search(self, query: str, top_k: int = 5, candidates: int = 50) -> list[Hit]:
        if not query or not query.strip() or not self.chunks:
            return []
        lexical = [cid for cid, _ in self.bm25.search(query, top_k=candidates)]
        dense = self._dense(query, candidates)
        fused = rrf([lexical, dense])[:candidates]
        hits = [Hit(cid, self.by_id[cid].doc_id, self.by_id[cid].start, self.by_id[cid].end,
                    self.by_id[cid].text, float(score)) for cid, score in fused]
        if self.reranker is not None and hits:
            scores = self.reranker(query, hits)
            paired = sorted(zip(hits, scores), key=lambda p: (-float(p[1]), p[0].chunk_id))
            hits = [replace(h, score=float(s)) for h, s in paired]
        return hits[:top_k]