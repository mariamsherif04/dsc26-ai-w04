"""Citation checks."""
from __future__ import annotations

from bayyina.types import Hit


def _collapse(s: str) -> str:
    return " ".join(s.split())


def verify_citation(quote: str, document: str, start: int, end: int) -> bool:
    if not (0 <= start < end <= len(document)):
        return False
    q = _collapse(quote)
    if not q:
        return False
    return q in _collapse(document[start:end])


def format_citation(hit: Hit, max_quote: int = 160) -> str:
    """[doc_id:start-end] "quote" """
    quote = _collapse(hit.text)
    if len(quote) > max_quote:
        cut = quote[:max_quote]
        i = cut.rfind(" ")
        if i != -1:
            cut = cut[:i]
        quote = cut + "\u2026"
    return f'[{hit.doc_id}:{hit.start}-{hit.end}] "{quote}"'