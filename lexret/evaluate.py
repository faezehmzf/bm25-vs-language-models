"""Ranking, evaluation measures, query-level cross-validation and significance testing.

Measures follow trec_eval for binary judgments: AP over the top 1,000 (divided by the number of
relevant documents), P@10, nDCG@10 with log2 discount, and recall at 1,000. Topics without any
relevant document are not evaluated. The test suite checks the measures against pytrec_eval.
"""

from __future__ import annotations

import itertools
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

DEPTH = 1000


def top_k(rows: np.ndarray, scores: np.ndarray, docnos: Sequence[str], k: int = DEPTH) -> List[str]:
    """Documents by decreasing score; ties broken by index order so runs are deterministic."""
    order = np.lexsort((rows, -scores))[:k]
    return [docnos[r] for r in rows[order]]


def average_precision(ranking: Sequence[str], relevant: set) -> float:
    hits, total = 0, 0.0
    for i, d in enumerate(ranking, 1):
        if d in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant)


def precision_at(ranking: Sequence[str], relevant: set, k: int = 10) -> float:
    return sum(d in relevant for d in ranking[:k]) / k


def ndcg_at(ranking: Sequence[str], relevant: set, k: int = 10) -> float:
    dcg = sum(1.0 / np.log2(i + 1) for i, d in enumerate(ranking[:k], 1) if d in relevant)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return dcg / idcg


def recall_at(ranking: Sequence[str], relevant: set, k: int = DEPTH) -> float:
    return sum(d in relevant for d in ranking[:k]) / len(relevant)


MEASURES: Dict[str, Callable[[Sequence[str], set], float]] = {
    "MAP": average_precision,
    "P@10": precision_at,
    "nDCG@10": ndcg_at,
    "R@1000": recall_at,
}


def evaluate_run(run: Dict[str, List[str]], relevant: Dict[str, set]) -> Dict[str, Dict[str, float]]:
    """Per-topic scores {measure: {qid: value}} for every judged topic (missing runs score 0)."""
    out = {m: {} for m in MEASURES}
    for qid, rel in relevant.items():
        ranking = run.get(qid, [])
        for m, fn in MEASURES.items():
            out[m][qid] = fn(ranking, rel)
    return out


# ---------------------------------------------------------------- tuning

def grid(params: Dict[str, list]) -> List[Dict[str, float]]:
    if not params:
        return [{}]
    keys = list(params)
    return [dict(zip(keys, values)) for values in itertools.product(*(params[k] for k in keys))]


def make_folds(qids: Sequence[str], n_folds: int = 5, seed: int = 0) -> List[List[str]]:
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(sorted(qids, key=int)))
    return [sorted(shuffled[i::n_folds], key=int) for i in range(n_folds)]


def cross_validate(ap_by_setting: Dict[Tuple, Dict[str, float]], folds: List[List[str]]):
    """Choose a setting on the training folds (highest MAP), apply it to the held-out fold.

    ap_by_setting maps a hashable setting to per-topic AP. Returns the per-topic held-out AP and
    the setting chosen for each fold. Ties between settings go to the first one in grid order.
    """
    settings = list(ap_by_setting)
    held_out: Dict[str, float] = {}
    chosen = []
    for test in folds:
        test_set = set(test)
        train = [q for q in ap_by_setting[settings[0]] if q not in test_set]
        best = max(settings, key=lambda s: np.mean([ap_by_setting[s][q] for q in train]))
        chosen.append(best)
        for q in test:
            held_out[q] = ap_by_setting[best][q]
    return held_out, chosen


# ---------------------------------------------------------------- significance

def randomization_test(a: Dict[str, float], b: Dict[str, float], n: int = 100_000, seed: int = 0) -> float:
    """Two-sided paired randomization (sign-flip) test on the difference of means."""
    qids = sorted(a)
    d = np.array([a[q] - b[q] for q in qids])
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.sum(null >= observed - 1e-12) + 1) / (n + 1))


def holm(pvalues: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0
