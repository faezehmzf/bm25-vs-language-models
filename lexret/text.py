"""Tokenisation, stop-word removal and Porter stemming.

The same Analyzer is applied to documents and queries. Stemming is cached per word form, which
matters: the collection has about 30 million tokens but only a few hundred thousand distinct forms.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import FrozenSet, Iterable, List, Optional

from nltk.stem import PorterStemmer

# letters and digits in any script (so "niño" stays one word), with internal apostrophes kept
# so that stop words such as "aren't" match; a trailing possessive 's is removed first.
_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
_POSSESSIVE = re.compile(r"'s\b")
_porter = PorterStemmer()


@lru_cache(maxsize=None)
def _stem(word: str) -> str:
    return _porter.stem(word)


def load_stopwords(path: str | Path) -> FrozenSet[str]:
    return frozenset(w.strip().lower() for w in Path(path).read_text(encoding="utf-8").split() if w.strip())


class Analyzer:
    """Lower-case alphanumeric tokens, optional stop-word removal, optional Porter stemming."""

    def __init__(self, stopwords: Optional[Iterable[str]] = None, stem: bool = True):
        self.stopwords = frozenset(stopwords or ())
        self.stem = stem

    @property
    def name(self) -> str:
        return f"{'stop' if self.stopwords else 'nostop'}-{'porter' if self.stem else 'nostem'}"

    def __call__(self, text: str) -> List[str]:
        text = _POSSESSIVE.sub("", text.lower().replace("’", "'"))
        tokens = [t for t in _WORD.findall(text) if t not in self.stopwords]
        return [_stem(t) for t in tokens] if self.stem else tokens
