"""Shared experiment code: load an index with its topics, run a model over a parameter grid,
cross-validate and summarise."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from . import evaluate as ev
from .corpus import read_qrels, read_topics
from .index import Index
from .models import Query
from .text import Analyzer, load_stopwords

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
N_FOLDS, FOLD_SEED = 5, 0

ANALYZERS = {
    "stop-porter": lambda: Analyzer(load_stopwords(DATA / "stopwords.txt"), stem=True),
    "stop-nostem": lambda: Analyzer(load_stopwords(DATA / "stopwords.txt"), stem=False),
    "nostop-porter": lambda: Analyzer(None, stem=True),
    "nostop-nostem": lambda: Analyzer(None, stem=False),
}


@dataclass
class Setup:
    index: Index
    analyzer: Analyzer
    topics: Dict[str, str]
    queries: Dict[str, Query]
    relevant: Dict[str, set]
    folds: List[List[str]]


def load_setup(variant: str = "stop-porter") -> Setup:
    index = Index.load(DATA / "index" / variant)
    analyzer = ANALYZERS[variant]()
    topics = read_topics(DATA / "topics.xml")
    qrels = read_qrels(DATA / "qrels.txt", set(topics))
    relevant = {q: {d for d, r in judged.items() if r > 0} for q, judged in qrels.items()}
    relevant = {q: r for q, r in relevant.items() if r}          # topics with no relevant document are not evaluated
    queries = {}
    for qid in relevant:
        counts = Counter(analyzer(topics[qid]))
        queries[qid] = {index.vocab[t]: float(c) for t, c in counts.items() if t in index.vocab}
    folds = ev.make_folds(list(relevant), N_FOLDS, FOLD_SEED)
    return Setup(index, analyzer, topics, queries, relevant, folds)


def run_model(setup: Setup, fn: Callable, params: Dict[str, float]) -> Dict[str, List[str]]:
    run = {}
    for qid, query in setup.queries.items():
        if query:
            rows, scores = fn(setup.index, query, **params)
            run[qid] = ev.top_k(rows, scores, setup.index.docnos)
    return run


def key(params: Dict[str, float]) -> str:
    return ",".join(f"{k}={v}" for k, v in params.items()) or "-"


def sweep(setup: Setup, fn: Callable, grid: Dict[str, list]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """{setting: {measure: {qid: value}}} for every setting in the grid."""
    return {key(p): ev.evaluate_run(run_model(setup, fn, p), setup.relevant) for p in ev.grid(grid)}


def cross_validated(per_setting: Dict[str, Dict[str, Dict[str, float]]], folds) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """Tune on MAP over the training folds; return held-out per-topic values of every measure."""
    ap = {s: m["MAP"] for s, m in per_setting.items()}
    _, chosen = ev.cross_validate(ap, folds)
    held = {m: {} for m in ev.MEASURES}
    for setting, test in zip(chosen, folds):
        for m in ev.MEASURES:
            for qid in test:
                held[m][qid] = per_setting[setting][m][qid]
    return held, chosen


def summary(per_topic: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {m: ev.mean(v.values()) for m, v in per_topic.items()}


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=False))


def md_table(rows: List[List], header: List[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def bootstrap_ci(values: List[float], n: int = 10_000, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
