"""Arabic-aware normalisation and tokenisation."""
from __future__ import annotations

import re

_REMOVE = {cp: None for cp in list(range(0x064B, 0x0653)) + [0x0670, 0x0640]}
_MAP = {ord(a): "\u0627" for a in "\u0623\u0625\u0622\u0671"}  # alef variants -> alef
_MAP[0x0649] = "\u064a"   # alef maqsura -> yeh
_MAP[0x0629] = "\u0647"   # teh marbuta -> heh
_MAP.update({0x0660 + i: str(i) for i in range(10)})   # Arabic-Indic digits
_MAP.update({0x06F0 + i: str(i) for i in range(10)})   # Persian digits
_TABLE = {**_REMOVE, **_MAP}
_WS = re.compile(r"\s+")
_WORD = re.compile(r"\w+")


def normalize_ar(text: str) -> str:
    """Remove diacritics and tatweel, unify letters and digits, lowercase, collapse whitespace."""
    text = text.translate(_TABLE).lower()
    return _WS.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """normalize_ar, then every match of the regex \\w+ (Unicode)."""
    return _WORD.findall(normalize_ar(text))