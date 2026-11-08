"""Ranking functions.

Every model takes an Index and a weighted query {term_id: weight} and returns (rows, scores) for the
documents that contain at least one query term. Query terms missing from the vocabulary are ignored
by the caller. Notation: tf = count of t in d, |d| = document length, avgdl = mean length,
N = number of documents, df / cf = document / collection frequency, p(t|C) = cf / total tokens.

Vector space and probabilistic models
  * TF-IDF (SMART lnc.ltc): cosine of (1 + ln tf) document weights and (1 + ln qtf) * ln(N/df)
    query weights.
  * BM25 (Robertson et al.) with the non-negative idf ln(1 + (N - df + 0.5) / (df + 0.5)).
  * BM25L and BM25+ (Lv & Zhai, 2011): two fixes for BM25's over-penalising of long documents.
  * PL2 (Amati & van Rijsbergen, 2002): divergence from randomness with Poisson model, Laplace
    after-effect and normalisation 2, as implemented in Terrier.

Query likelihood (Zhai & Lafferty, 2004): score = sum_t q(t) * log p(t|d), over all query terms,
including those absent from d, with p(t|d) smoothed by
  * Laplace / Lidstone:   (tf + delta) / (|d| + delta |V|)
  * Jelinek-Mercer:       (1 - lam) tf / |d| + lam p(t|C)           (lam = weight of the collection)
  * Dirichlet:            (tf + mu p(t|C)) / (|d| + mu)
  * absolute discounting: max(tf - delta, 0) / |d| + delta |d|_unique / |d| p(t|C)
  * two-stage:            (1 - lam) Dirichlet(mu) + lam p(t|C)
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

from .index import Index

Query = Dict[int, float]
Scores = Tuple[np.ndarray, np.ndarray]


def _gather(index: Index, query: Query):
    """Term ids, query weights, candidate rows and the dense tf block (candidates x query terms)."""
    ids = np.fromiter(query.keys(), dtype=np.int64)
    w = np.fromiter(query.values(), dtype=np.float64)
    sub = index.tf[:, ids]                 # CSC: .indices are document rows
    rows = np.unique(sub.indices)
    tf = sub.tocsr()[rows].toarray()
    return ids, w, rows, tf


# ---------------------------------------------------------------- vector space / probabilistic

def tfidf(index: Index, query: Query) -> Scores:
    ids, w, rows, tf = _gather(index, query)
    if not hasattr(index, "_lnc_norm"):
        logtf = index.tf.copy()
        logtf.data = 1.0 + np.log(logtf.data)
        index._lnc_norm = np.sqrt(np.asarray(logtf.multiply(logtf).sum(axis=1)).ravel())
    dw = np.where(tf > 0, 1.0 + np.log(np.maximum(tf, 1.0)), 0.0) / index._lnc_norm[rows, None]
    qw = (1.0 + np.log(w)) * np.log(index.n_docs / index.df[ids])
    return rows, dw @ qw


def _bm25_idf(index: Index, ids: np.ndarray) -> np.ndarray:
    df = index.df[ids]
    return np.log(1.0 + (index.n_docs - df + 0.5) / (df + 0.5))


def bm25(index: Index, query: Query, k1: float = 1.2, b: float = 0.75) -> Scores:
    ids, w, rows, tf = _gather(index, query)
    norm = k1 * (1.0 - b + b * index.doc_len[rows] / index.avgdl)
    sat = tf * (k1 + 1.0) / (tf + norm[:, None])
    return rows, sat @ (w * _bm25_idf(index, ids))


def bm25l(index: Index, query: Query, k1: float = 1.2, b: float = 0.75, delta: float = 0.5) -> Scores:
    ids, w, rows, tf = _gather(index, query)
    c = tf / (1.0 - b + b * index.doc_len[rows] / index.avgdl)[:, None]
    sat = np.where(tf > 0, (k1 + 1.0) * (c + delta) / (k1 + c + delta), 0.0)
    idf = np.log((index.n_docs + 1.0) / (index.df[ids] + 0.5))
    return rows, sat @ (w * idf)


def bm25plus(index: Index, query: Query, k1: float = 1.2, b: float = 0.75, delta: float = 1.0) -> Scores:
    ids, w, rows, tf = _gather(index, query)
    norm = k1 * (1.0 - b + b * index.doc_len[rows] / index.avgdl)
    sat = np.where(tf > 0, tf * (k1 + 1.0) / (tf + norm[:, None]) + delta, 0.0)
    idf = np.log((index.n_docs + 1.0) / index.df[ids])
    return rows, sat @ (w * idf)


_LOG2E = 1.0 / np.log(2.0)


def pl2(index: Index, query: Query, c: float = 1.0) -> Scores:
    ids, w, rows, tf = _gather(index, query)
    tfn = tf * np.log2(1.0 + c * index.avgdl / index.doc_len[rows])[:, None]
    lam = index.cf[ids] / index.n_docs
    with np.errstate(divide="ignore", invalid="ignore"):
        info = (tfn * np.log2(1.0 / lam) + lam * _LOG2E + 0.5 * np.log2(2.0 * np.pi * tfn)
                + tfn * (np.log2(tfn) - _LOG2E))
        s = np.where(tf > 0, info / (tfn + 1.0), 0.0)
    return rows, s @ w


# ---------------------------------------------------------------- query likelihood

def _ql(index: Index, query: Query, p_fn) -> Scores:
    ids, w, rows, tf = _gather(index, query)
    dl = index.doc_len[rows][:, None]
    pc = index.p_collection(ids)[None, :]
    p = p_fn(tf, dl, pc, rows)
    return rows, np.log(p) @ w


def ql_laplace(index: Index, query: Query, delta: float = 1.0) -> Scores:
    v = len(index.vocab)
    return _ql(index, query, lambda tf, dl, pc, rows: (tf + delta) / (dl + delta * v))


def ql_jm(index: Index, query: Query, lam: float = 0.5) -> Scores:
    return _ql(index, query, lambda tf, dl, pc, rows: (1.0 - lam) * tf / dl + lam * pc)


def ql_dirichlet(index: Index, query: Query, mu: float = 1000.0) -> Scores:
    return _ql(index, query, lambda tf, dl, pc, rows: (tf + mu * pc) / (dl + mu))


def ql_absdisc(index: Index, query: Query, delta: float = 0.7) -> Scores:
    def p(tf, dl, pc, rows):
        uniq = index.doc_unique[rows][:, None]
        return np.maximum(tf - delta, 0.0) / dl + delta * uniq / dl * pc
    return _ql(index, query, p)


def ql_two_stage(index: Index, query: Query, mu: float = 1000.0, lam: float = 0.5) -> Scores:
    return _ql(index, query, lambda tf, dl, pc, rows: (1.0 - lam) * (tf + mu * pc) / (dl + mu) + lam * pc)


# ---------------------------------------------------------------- registry and parameter grids

MODELS: Dict[str, Tuple[Callable[..., Scores], Dict[str, list]]] = {
    "TF-IDF (lnc.ltc)": (tfidf, {}),
    "BM25": (bm25, {"k1": [0.6, 0.9, 1.2, 1.5, 1.8, 2.1], "b": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}),
    "BM25L": (bm25l, {"k1": [0.6, 0.9, 1.2, 1.5, 1.8, 2.1], "b": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}),
    "BM25+": (bm25plus, {"k1": [0.6, 0.9, 1.2, 1.5, 1.8, 2.1], "b": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}),
    "PL2": (pl2, {"c": [0.5, 1, 2, 3, 5, 7, 10, 15]}),
    "QL Laplace": (ql_laplace, {"delta": [0.001, 0.01, 0.1, 0.5, 1.0]}),
    "QL Jelinek-Mercer": (ql_jm, {"lam": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}),
    "QL Dirichlet": (ql_dirichlet, {"mu": [100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000]}),
    "QL absolute discounting": (ql_absdisc, {"delta": [0.1, 0.3, 0.5, 0.7, 0.9]}),
    "QL two-stage": (ql_two_stage, {"mu": [250, 500, 1000, 2000], "lam": [0.1, 0.3, 0.5, 0.7]}),
}
