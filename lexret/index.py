"""Inverted index stored as a sparse document-term count matrix.

The matrix is kept in CSC form (one column per term), so the postings of a term are a column slice.
Articles that are empty after analysis are not indexed: they cannot match any query.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from .text import Analyzer


@dataclass
class Index:
    docnos: List[str]
    vocab: Dict[str, int]
    tf: sp.csc_matrix          # docs x terms, raw counts
    doc_len: np.ndarray        # tokens per document
    doc_unique: np.ndarray     # distinct terms per document
    df: np.ndarray             # documents containing each term
    cf: np.ndarray             # occurrences of each term in the collection

    @property
    def n_docs(self) -> int:
        return len(self.docnos)

    @property
    def n_tokens(self) -> int:
        return int(self.doc_len.sum())

    @property
    def avgdl(self) -> float:
        return float(self.doc_len.mean())

    def p_collection(self, term_ids: np.ndarray) -> np.ndarray:
        return self.cf[term_ids] / self.n_tokens

    def term_ids(self, terms: Iterable[str]) -> List[int]:
        return [self.vocab[t] for t in terms if t in self.vocab]

    def postings(self, term_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """(document rows, term counts) of one term."""
        start, end = self.tf.indptr[term_id], self.tf.indptr[term_id + 1]
        return self.tf.indices[start:end], self.tf.data[start:end]

    def column(self, term_id: int) -> np.ndarray:
        """Dense term-count vector over all documents."""
        out = np.zeros(self.n_docs)
        rows, counts = self.postings(term_id)
        out[rows] = counts
        return out

    # ------------------------------------------------------------------ build / save / load

    @classmethod
    def build(cls, docs: Iterable[Tuple[str, str]], analyzer: Analyzer) -> "Index":
        vocab: Dict[str, int] = {}
        docnos: List[str] = []
        rows: List[int] = []
        cols: List[int] = []
        vals: List[int] = []
        for docno, text in docs:
            counts = Counter(analyzer(text))
            if not counts:
                continue
            r = len(docnos)
            docnos.append(docno)
            for term, c in counts.items():
                rows.append(r)
                cols.append(vocab.setdefault(term, len(vocab)))
                vals.append(c)
        tf = sp.csc_matrix(
            (np.asarray(vals, dtype=np.float64), (np.asarray(rows), np.asarray(cols))),
            shape=(len(docnos), len(vocab)),
        )
        return cls.from_matrix(docnos, vocab, tf)

    @classmethod
    def from_matrix(cls, docnos: List[str], vocab: Dict[str, int], tf: sp.spmatrix) -> "Index":
        tf = sp.csc_matrix(tf)
        tf.sort_indices()
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        doc_unique = np.diff(tf.tocsr().indptr).astype(np.float64)
        df = np.diff(tf.indptr).astype(np.float64)
        cf = np.asarray(tf.sum(axis=0)).ravel()
        return cls(docnos, vocab, tf, doc_len, doc_unique, df, cf)

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        sp.save_npz(d / "tf.npz", self.tf)
        terms = sorted(self.vocab, key=self.vocab.get)
        (d / "meta.json").write_text(json.dumps({"docnos": self.docnos, "terms": terms}))

    @classmethod
    def load(cls, directory: str | Path) -> "Index":
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        vocab = {t: i for i, t in enumerate(meta["terms"])}
        return cls.from_matrix(meta["docnos"], vocab, sp.load_npz(d / "tf.npz"))


def toy_index(docs: Sequence[Sequence[str]]) -> Index:
    """Index pre-tokenised documents (used by the tests)."""
    analyzer = Analyzer(stopwords=None, stem=False)
    return Index.build(((f"d{i}", " ".join(toks)) for i, toks in enumerate(docs)), analyzer)
