"""Model-based pseudo-relevance feedback (Zhai & Lafferty, 2001, "simple mixture model").

The top-k documents of a first Dirichlet run are treated as relevant. Every word occurrence in them is
assumed to come either from a feedback topic model theta_F or from the collection model p(w|C):

    p(w) = (1 - lam_bg) * theta_F(w) + lam_bg * p(w|C)

with lam_bg fixed. EM estimates theta_F; the background takes the common words, so theta_F keeps the
words that are frequent in the feedback documents but not in general. The query model becomes

    theta_Q' = (1 - alpha) * theta_Q + alpha * theta_F(top m terms, renormalised)

and documents are ranked by the cross entropy sum_w theta_Q'(w) log p_Dirichlet(w|d), which ranks
the same as KL divergence. With alpha = 0 this is the plain Dirichlet run.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .index import Index
from .models import Query, ql_dirichlet


def feedback_counts(index: Index, rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Term ids and their total counts in the given documents."""
    if not hasattr(index, "_csr"):
        index._csr = index.tf.tocsr()
    counts = np.asarray(index._csr[rows].sum(axis=0)).ravel()
    ids = np.flatnonzero(counts)
    return ids, counts[ids]


def em_topic_model(counts: np.ndarray, p_bg: np.ndarray, lam_bg: float, iters: int = 50,
                   tol: float = 1e-8) -> Tuple[np.ndarray, list]:
    """EM for theta_F given word counts in the feedback set and their background probabilities.

    Returns theta_F and the log-likelihood after each iteration (non-decreasing).
    """
    theta = counts / counts.sum()
    history = []
    for _ in range(iters):
        mix = (1.0 - lam_bg) * theta
        z = mix / (mix + lam_bg * p_bg)              # E-step: P(word came from theta_F)
        theta_new = counts * z
        theta_new /= theta_new.sum()                  # M-step
        ll = float(np.sum(counts * np.log((1.0 - lam_bg) * theta_new + lam_bg * p_bg)))
        history.append(ll)
        done = np.abs(theta_new - theta).sum() < tol
        theta = theta_new
        if done:
            break
    return theta, history


def expand_query(index: Index, query: Query, fb_rows: np.ndarray, alpha: float, n_terms: int,
                 lam_bg: float = 0.5) -> Query:
    ids, counts = feedback_counts(index, fb_rows)
    theta, _ = em_topic_model(counts, index.p_collection(ids), lam_bg)
    top = np.argsort(-theta, kind="stable")[:n_terms]
    theta_f = theta[top] / theta[top].sum()
    total = sum(query.values())
    new: Dict[int, float] = {t: (1.0 - alpha) * w / total for t, w in query.items()}
    for t, p in zip(ids[top].tolist(), theta_f.tolist()):
        new[t] = new.get(t, 0.0) + alpha * p
    return new


def feedback_run(index: Index, query: Query, mu: float, fb_docs: int, alpha: float, n_terms: int,
                 lam_bg: float = 0.5, first_pass=None):
    """Dirichlet run, EM feedback on its top fb_docs documents, then a second Dirichlet run."""
    rows, scores = first_pass if first_pass is not None else ql_dirichlet(index, query, mu=mu)
    order = np.lexsort((rows, -scores))[:fb_docs]
    expanded = expand_query(index, query, rows[order], alpha, n_terms, lam_bg)
    return ql_dirichlet(index, expanded, mu=mu)
