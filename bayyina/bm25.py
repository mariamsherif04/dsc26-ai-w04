"""BM25, written by hand."""
from __future__ import annotations

import math
from collections import Counter

from bayyina.normalize import tokenize
from bayyina.types import Chunk


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._ids: list[str] = []
        self._len: list[int] = []
        self._post: dict[str, list[tuple[int, int]]] = {}   # term -> [(doc index, tf)]

    def add(self, chunks: list[Chunk]) -> None:
        for ch in chunks:
            idx = len(self._ids)
            toks = tokenize(ch.text)
            self._ids.append(ch.chunk_id)
            self._len.append(len(toks))
            for term, tf in Counter(toks).items():
                self._post.setdefault(term, []).append((idx, tf))

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """[(chunk_id, score), ...] with score > 0, by score descending then chunk ID ascending."""
        n = len(self._ids)
        terms = list(dict.fromkeys(tokenize(query)))        # each distinct term once
        if n == 0 or not terms or top_k <= 0:
            return []
        avgdl = sum(self._len) / n
        scores: dict[int, float] = {}
        for term in terms:
            post = self._post.get(term)
            if not post:
                continue
            df = len(post)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for idx, tf in post:
                norm = 1 - self.b + self.b * (self._len[idx] / avgdl if avgdl else 0.0)
                scores[idx] = scores.get(idx, 0.0) + idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        ranked = sorted(((self._ids[i], s) for i, s in scores.items() if s > 0),
                        key=lambda t: (-t[1], t[0]))
        return ranked[:top_k]