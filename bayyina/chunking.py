"""Offset-preserving chunker."""
from __future__ import annotations

from bayyina.types import Chunk

_SENTENCE_END = ".!?\u061f\u06d4\n"   # . ! ? (Arabic ?) (Urdu full stop) and newline


def _cut(window: str, max_chars: int) -> int:
    """Length of the chunk inside `window` (before trailing whitespace is stripped)."""
    lo = max_chars // 2
    para = sent = space = None
    for c in range(max_chars, lo - 1, -1):          # largest c first
        if para is None and c >= 2 and window[c - 2:c] == "\n\n":
            para = c
            break                                   # most preferred kind, largest c
        if sent is None and c >= 1 and window[c - 1] in _SENTENCE_END:
            sent = c
        if space is None and c >= 1 and window[c - 1].isspace():
            space = c
    for found in (para, sent, space):
        if found is not None:
            return found
    return max_chars                                # hard cut


def chunk_text(doc_id: str, text: str, max_chars: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split `text` into chunks whose offsets point into the ORIGINAL text.

    ValueError unless max_chars > 0 and 0 <= overlap < max_chars.
    """
    if not (max_chars > 0 and 0 <= overlap < max_chars):
        raise ValueError("need max_chars > 0 and 0 <= overlap < max_chars")
    n = len(text)
    s = 0
    while s < n and text[s].isspace():
        s += 1
    chunks: list[Chunk] = []
    while s < n:
        if n - s <= max_chars:
            piece = text[s:n].rstrip()
            chunks.append(Chunk(doc_id, f"{doc_id}#{len(chunks)}", s, s + len(piece), piece))
            break
        c = _cut(text[s:s + max_chars], max_chars)
        piece = text[s:s + c].rstrip()
        e = s + len(piece)
        chunks.append(Chunk(doc_id, f"{doc_id}#{len(chunks)}", s, e, piece))
        # choose the next start: inside the overlap if a word starts there, else after the cut
        lo = max(e - overlap, s + 1)
        ns = None
        for i in range(lo, e):
            if not text[i].isspace() and text[i - 1].isspace():
                ns = i
                break
        if ns is None:
            if text[e].isspace():                   # clean cut: next word after the gap
                ns = e
                while ns < n and text[ns].isspace():
                    ns += 1
                if ns >= n:
                    break
            else:                                   # mid-word cut: mid-word restart allowed
                ns = lo
        s = ns
    return chunks