import math
import random

import numpy as np
import pytest

from lexret import evaluate as ev
from lexret import models as M
from lexret.corpus import iter_articles, read_qrels, read_topics
from lexret.feedback import em_topic_model, expand_query
from lexret.index import toy_index
from lexret.text import Analyzer

DOCS = [
    "the cat sat on the mat".split(),
    "the dog sat on the log".split(),
    "cats and dogs and cats".split(),
    "a long document about the cat and the dog and the mat and the log".split(),
]


@pytest.fixture
def idx():
    return toy_index(DOCS)


def q(idx, *terms):
    out = {}
    for t in terms:
        out[idx.vocab[t]] = out.get(idx.vocab[t], 0) + 1.0
    return out


def as_dict(idx, result):
    rows, scores = result
    return {idx.docnos[r]: s for r, s in zip(rows, scores)}


# ---------------------------------------------------------------- text and corpus

def test_analyzer_keeps_accents_and_removes_possessives():
    a = Analyzer(stopwords={"aren't", "the"}, stem=False)
    assert a("El Niño and the weather") == ["el", "niño", "and", "weather"]
    assert a("Clinton's plans aren't final") == ["clinton", "plans", "final"]
    assert Analyzer(stem=True)("running runners") == ["run", "runner"]


def test_readers(tmp_path):
    (tmp_path / "c.xml").write_text(
        "<DOC>\n<DOCNO>la010194</DOCNO>\n<TEXT>\n<DOC>\n<DOCNO> LA010194-0001 </DOCNO>\n"
        "<HEADLINE>\n<P>\nBig News\n</P>\n</HEADLINE>\n<BYLINE><P>By Someone</P></BYLINE>\n"
        "<TEXT>\n<P>\nSome text.\n</P>\n</TEXT>\n</DOC>\n", encoding="latin-1")
    (docno, text), = list(iter_articles(tmp_path / "c.xml"))
    assert docno == "LA010194-0001" and text.split() == ["Big", "News", "Some", "text."]
    (tmp_path / "t.xml").write_text("<DOC 41>\npesticides\nin\nbaby\nfood\n</DOC>\n")
    assert read_topics(tmp_path / "t.xml") == {"41": "pesticides in baby food"}
    (tmp_path / "q.txt").write_text("41 0 LA1 1\n41 0 LA2 0\n200 0 LA3 1\n")
    assert read_qrels(tmp_path / "q.txt", {"41"}) == {"41": {"LA1": 1, "LA2": 0}}


def test_index_statistics(idx):
    assert idx.n_docs == 4
    cat = idx.vocab["cat"]
    assert idx.df[cat] == 2 and idx.cf[cat] == 2
    assert idx.cf[idx.vocab["cats"]] == 2 and idx.df[idx.vocab["cats"]] == 1
    assert list(idx.doc_len) == [6, 6, 5, 15]
    empty = toy_index([["a"], []])
    assert empty.n_docs == 1


# ---------------------------------------------------------------- models

def test_bm25_matches_formula(idx):
    k1, b = 1.2, 0.75
    s = as_dict(idx, M.bm25(idx, q(idx, "cat", "mat"), k1=k1, b=b))
    n, avgdl = 4, idx.avgdl
    def term(tf, dl, df):
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    assert s["d0"] == pytest.approx(term(1, 6, 2) + term(1, 6, 2))
    assert s["d3"] == pytest.approx(term(1, 15, 2) + term(1, 15, 2))
    assert set(s) == {"d0", "d3"}            # only documents with a query term are scored


def test_pl2_matches_terrier_formula(idx):
    c = 1.0
    s = as_dict(idx, M.pl2(idx, q(idx, "dog"), c=c))
    tfn = 1 * math.log2(1 + c * idx.avgdl / 6)
    lam = 2 / 4
    expected = (tfn * math.log2(1 / lam) + lam / math.log(2) + 0.5 * math.log2(2 * math.pi * tfn)
                + tfn * (math.log2(tfn) - 1 / math.log(2))) / (tfn + 1)
    assert s["d1"] == pytest.approx(expected)


SMOOTHERS = [
    (M.ql_laplace, {"delta": 0.5}),
    (M.ql_jm, {"lam": 0.3}),
    (M.ql_dirichlet, {"mu": 7.0}),
    (M.ql_absdisc, {"delta": 0.7}),
    (M.ql_two_stage, {"mu": 7.0, "lam": 0.2}),
]


@pytest.mark.parametrize("fn,params", SMOOTHERS)
def test_smoothed_document_models_are_distributions(idx, fn, params):
    # The score of a one-term query is log p(t|d), so summed over the vocabulary p(t|d) must be 1.
    # A zero-weight "anchor" term from the document makes it a candidate without changing its score.
    csr = idx.tf.tocsr()
    for doc in range(idx.n_docs):
        anchor = int(csr[doc].indices[0])
        total = 0.0
        for tid in idx.vocab.values():
            query = {tid: 1.0} if tid == anchor else {tid: 1.0, anchor: 0.0}
            rows, scores = fn(idx, query, **params)
            total += math.exp(scores[list(rows).index(doc)])
        assert total == pytest.approx(1.0, abs=1e-9)


def test_query_likelihood_counts_absent_terms(idx):
    mu = 5.0
    s = as_dict(idx, M.ql_dirichlet(idx, q(idx, "cat", "log"), mu=mu))
    pc = lambda t: idx.cf[idx.vocab[t]] / idx.n_tokens
    expected = math.log((1 + mu * pc("cat")) / (6 + mu)) + math.log((0 + mu * pc("log")) / (6 + mu))
    assert s["d0"] == pytest.approx(expected)


def test_tfidf_prefers_shorter_document_for_same_match(idx):
    s = as_dict(idx, M.tfidf(idx, q(idx, "mat")))
    assert s["d0"] > s["d3"] > 0


# ---------------------------------------------------------------- feedback

def test_em_likelihood_non_decreasing_and_background_absorbs_common_words():
    counts = np.array([50.0, 10.0, 10.0])
    p_bg = np.array([0.5, 0.001, 0.001])          # word 0 is common in the collection
    theta, ll = em_topic_model(counts, p_bg, lam_bg=0.7, iters=200)
    assert all(b >= a - 1e-9 for a, b in zip(ll, ll[1:]))
    assert theta.sum() == pytest.approx(1.0)
    assert theta[0] < counts[0] / counts.sum()    # its share drops below its raw frequency
    assert theta[1] > counts[1] / counts.sum()


def test_expanded_query_is_a_distribution(idx):
    new = expand_query(idx, q(idx, "cat"), np.array([0, 3]), alpha=0.5, n_terms=3)
    assert sum(new.values()) == pytest.approx(1.0)
    assert new[idx.vocab["cat"]] >= 0.5


# ---------------------------------------------------------------- evaluation

def test_measures_match_pytrec_eval():
    pytrec_eval = pytest.importorskip("pytrec_eval")
    rng = random.Random(0)
    docs = [f"D{i}" for i in range(300)]
    run, rel, qrels, trec_run = {}, {}, {}, {}
    for qn in range(20):
        qid = str(qn)
        ranking = rng.sample(docs, 120)
        relevant = set(rng.sample(docs, rng.randint(1, 15)))
        run[qid], rel[qid] = ranking, relevant
        qrels[qid] = {d: 1 for d in relevant}
        trec_run[qid] = {d: float(len(ranking) - i) for i, d in enumerate(ranking)}
    ours = ev.evaluate_run(run, rel)
    ref = pytrec_eval.RelevanceEvaluator(qrels, {"map", "P_10", "ndcg_cut_10", "recall_1000"}).evaluate(trec_run)
    for qid in run:
        assert ours["MAP"][qid] == pytest.approx(ref[qid]["map"])
        assert ours["P@10"][qid] == pytest.approx(ref[qid]["P_10"])
        assert ours["nDCG@10"][qid] == pytest.approx(ref[qid]["ndcg_cut_10"])
        assert ours["R@1000"][qid] == pytest.approx(ref[qid]["recall_1000"])


def test_cross_validation_never_looks_at_the_test_fold():
    qids = [str(i) for i in range(10)]
    folds = ev.make_folds(qids, n_folds=5, seed=1)
    assert sorted(q for f in folds for q in f) == sorted(qids)
    base = {"A": {q: 0.5 for q in qids}, "B": {q: 0.4 for q in qids}}
    held, chosen = ev.cross_validate(base, folds)
    assert chosen == ["A"] * 5
    # make B far better on the first test fold only: fold 0 must still use A
    boosted = {"A": dict(base["A"]), "B": dict(base["B"])}
    for qid in folds[0]:
        boosted["B"][qid] = 1.0
    held2, chosen2 = ev.cross_validate(boosted, folds)
    assert chosen2[0] == "A"
    assert all(held2[qid] == 0.5 for qid in folds[0])


def test_randomization_test_and_holm():
    a = {str(i): 0.5 for i in range(30)}
    assert ev.randomization_test(a, dict(a), n=2000) == pytest.approx(1.0)
    b = {k: v - 0.2 for k, v in a.items()}
    assert ev.randomization_test(a, b, n=2000) < 0.01
    adj = ev.holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert adj["x"] == pytest.approx(0.03) and adj["z"] == pytest.approx(0.06) and adj["y"] == pytest.approx(0.06)
