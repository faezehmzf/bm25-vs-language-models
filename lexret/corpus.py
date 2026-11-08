"""Readers for the LA Times 1994 collection, its title-only topics and the relevance judgments.

The collection file wraps each day's articles in an outer <DOC> element, so articles are found by
their DOCNO (``LA010194-0001``) instead of by matching <DOC> tags. Only the HEADLINE and TEXT fields
are indexed; the other fields (date, section, byline, length) are metadata.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, Set, Tuple

_ARTICLE = re.compile(r"<DOCNO>\s*(LA\d{6}-\d{4})\s*</DOCNO>(.*?)</DOC>", re.S)
_FIELD = {name: re.compile(rf"<{name}>(.*?)</{name}>", re.S) for name in ("HEADLINE", "TEXT")}
_TAG = re.compile(r"<[^>]+>")
_TOPIC = re.compile(r"<DOC (\d+)>(.*?)</DOC>", re.S)


def iter_articles(path: str | Path) -> Iterator[Tuple[str, str]]:
    """Yield (docno, headline + text) for every article in the collection file."""
    raw = Path(path).read_text(encoding="latin-1")
    for m in _ARTICLE.finditer(raw):
        body = m.group(2)
        parts = []
        for rx in _FIELD.values():
            parts.extend(_TAG.sub(" ", f) for f in rx.findall(body))
        yield m.group(1), " ".join(parts)


def read_topics(path: str | Path) -> Dict[str, str]:
    """Topic id -> title. The file has one word per line inside <DOC n> ... </DOC>."""
    text = Path(path).read_text(encoding="utf-8")
    return {qid: " ".join(body.split()) for qid, body in _TOPIC.findall(text)}


def read_qrels(path: str | Path, topics: Set[str] | None = None) -> Dict[str, Dict[str, int]]:
    """Topic id -> {docno: relevance}. Lines are ``qid 0 docno rel``; other topic sets are skipped."""
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        qid, _, docno, rel = parts
        if topics is None or qid in topics:
            qrels[qid][docno] = int(rel)
    return dict(qrels)
