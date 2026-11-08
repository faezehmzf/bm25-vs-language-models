"""Pseudo-relevance feedback with the EM mixture model, tuned by the same 5-fold cross-validation.

Grid: Dirichlet mu, number of feedback documents, number of expansion terms, interpolation weight
alpha and background weight lam_bg. Feedback is compared with the cross-validated Dirichlet and BM25
runs from scripts/run_models.py (run that first).

Outputs: results/feedback.md, results/feedback.json, results/figures/feedback_per_topic.png
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lexret import evaluate as ev  # noqa: E402
from lexret.experiment import (RESULTS, bootstrap_ci, cross_validated, key, load_setup, md_table,  # noqa: E402
                               save_json, summary)
from lexret.feedback import expand_query, feedback_run  # noqa: E402
from lexret.models import ql_dirichlet  # noqa: E402

GRID = {"mu": [500, 1000, 2000], "fb_docs": [5, 10, 20], "n_terms": [20, 50],
        "alpha": [0.2, 0.4, 0.6, 0.8], "lam_bg": [0.5, 0.8]}


def main() -> None:
    setup = load_setup("stop-porter")
    idx = setup.index
    first = {mu: {q: ql_dirichlet(idx, query, mu=mu) for q, query in setup.queries.items()} for mu in GRID["mu"]}

    per_setting = {}
    for p in ev.grid(GRID):
        run = {}
        for qid, query in setup.queries.items():
            rows, scores = feedback_run(idx, query, first_pass=first[p["mu"]][qid], **p)
            run[qid] = ev.top_k(rows, scores, idx.docnos)
        per_setting[key(p)] = ev.evaluate_run(run, setup.relevant)
    held, chosen = cross_validated(per_setting, setup.folds)
    print("feedback CV", summary(held), chosen, flush=True)

    base = json.loads((RESULTS / "per_topic_cv.json").read_text())
    dirichlet, bm25 = base["QL Dirichlet"], base["BM25"]
    p_dir = ev.randomization_test(held["MAP"], dirichlet["MAP"])
    p_bm25 = ev.randomization_test(held["MAP"], bm25["MAP"])
    diff = {q: held["MAP"][q] - dirichlet["MAP"][q] for q in setup.relevant}
    better = sum(d > 1e-9 for d in diff.values())
    worse = sum(d < -1e-9 for d in diff.values())

    rows = []
    for name, per_topic in [("QL Dirichlet", dirichlet), ("BM25", bm25), ("Dirichlet + EM feedback", held)]:
        s = summary(per_topic)
        lo, hi = bootstrap_ci(list(per_topic["MAP"].values()))
        rows.append([name, f"{s['MAP']:.4f}", f"[{lo:.3f}, {hi:.3f}]", f"{s['P@10']:.4f}", f"{s['nDCG@10']:.4f}", f"{s['R@1000']:.4f}"])

    # expansion terms for the topics feedback helps and hurts most (setting chosen on fold 0)
    p0 = dict(item.split("=") for item in chosen[0].split(","))
    p0 = {k: (int(v) if k in ("fb_docs", "n_terms") else float(v)) for k, v in p0.items()}
    inv = {i: t for t, i in idx.vocab.items()}
    examples = []
    for qid in sorted(diff, key=diff.get)[:2] + sorted(diff, key=diff.get)[-3:]:
        rows_, scores_ = first[p0["mu"]][qid] if p0["mu"] in first else ql_dirichlet(idx, setup.queries[qid], mu=p0["mu"])
        fb = rows_[np.lexsort((rows_, -scores_))[:p0["fb_docs"]]]
        new = expand_query(idx, setup.queries[qid], fb, p0["alpha"], p0["n_terms"], p0["lam_bg"])
        added = sorted(((w, inv[t]) for t, w in new.items() if t not in setup.queries[qid]), reverse=True)[:8]
        examples.append([qid, setup.topics[qid], f"{dirichlet['MAP'][qid]:.3f} → {held['MAP'][qid]:.3f}",
                         ", ".join(t for _, t in added)])

    md = ["# Pseudo-relevance feedback (EM mixture model)", "",
          "Same 5-fold cross-validation as results/models.md; every number is from held-out folds.", "",
          md_table(rows, ["run", "MAP", "MAP 95% CI", "P@10", "nDCG@10", "R@1000"]), "",
          f"- Feedback vs Dirichlet: p = {p_dir:.4f} (paired randomization test on AP); "
          f"better on {better} topics, worse on {worse}, of {len(diff)}.",
          f"- Feedback vs BM25: p = {p_bm25:.4f}.", "",
          "Settings chosen per fold: " + "; ".join(chosen), "",
          f"## Expansion terms (Porter stems) for the topics feedback hurts and helps most (fold-0 setting: {chosen[0]})", "",
          md_table(examples, ["topic", "title", "AP Dirichlet → feedback", "top added terms"])]
    (RESULTS / "feedback.md").write_text("\n".join(md) + "\n")
    save_json({"cv": summary(held), "chosen_per_fold": chosen, "p_vs_dirichlet": p_dir, "p_vs_bm25": p_bm25,
               "topics_better": better, "topics_worse": worse, "per_topic_map": held["MAP"],
               "grid": GRID, "per_setting_map": {k: ev.mean(v["MAP"].values()) for k, v in per_setting.items()}},
              RESULTS / "feedback.json")

    d = np.array(sorted(diff.values()))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(range(len(d)), d, color=np.where(d >= 0, "#2c5364", "#c0504d"), width=0.9)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xlabel("topics, sorted by change")
    ax.set_ylabel("AP(feedback) − AP(Dirichlet)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "feedback_per_topic.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
