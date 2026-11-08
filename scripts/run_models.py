"""Compare ranking models with 5-fold cross-validation over topics.

For every model, every setting of its parameter grid is run on all topics. The setting used for a
test fold is the one with the best MAP on the other four folds, so every reported number is
measured on topics that played no part in choosing the parameters. Each model is compared with
BM25 by a paired randomization test on per-topic AP, with Holm correction over the models.

Also writes the parameter-sensitivity curves (MAP on all topics for every setting; exploratory).

Outputs: results/models.md, results/models.json, results/per_setting.json, results/figures/*.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lexret import evaluate as ev  # noqa: E402
from lexret.experiment import (RESULTS, bootstrap_ci, cross_validated, load_setup, md_table,  # noqa: E402
                               run_model, save_json, summary, sweep)
from lexret.models import MODELS  # noqa: E402

FIG = RESULTS / "figures"


def main() -> None:
    setup = load_setup("stop-porter")
    per_setting, held, chosen = {}, {}, {}
    for name, (fn, grid) in MODELS.items():
        per_setting[name] = sweep(setup, fn, grid)
        held[name], chosen[name] = cross_validated(per_setting[name], setup.folds)
        print(f"{name:26s} CV MAP {summary(held[name])['MAP']:.4f}  folds: {chosen[name]}", flush=True)

    pvals = {n: ev.randomization_test(held[n]["MAP"], held["BM25"]["MAP"]) for n in MODELS if n != "BM25"}
    adj = ev.holm(pvals)

    rows, out = [], {"topics": len(setup.relevant), "folds": setup.folds, "models": {}}
    for name in MODELS:
        s = summary(held[name])
        lo, hi = bootstrap_ci(list(held[name]["MAP"].values()))
        best_all = max(per_setting[name], key=lambda k: ev.mean(per_setting[name][k]["MAP"].values()))
        oracle = ev.mean(per_setting[name][best_all]["MAP"].values())
        d = np.array([held[name]["MAP"][q] - held["BM25"]["MAP"][q] for q in setup.relevant])
        out["models"][name] = {**s, "MAP_95ci": [lo, hi], "chosen_per_fold": chosen[name],
                               "best_setting_all_topics": best_all, "MAP_best_setting_all_topics": oracle,
                               "vs_bm25_p": pvals.get(name), "vs_bm25_p_holm": adj.get(name),
                               "topics_better_than_bm25": int((d > 1e-9).sum()),
                               "topics_worse_than_bm25": int((d < -1e-9).sum())}
        sig = "" if name == "BM25" else ("< 0.001" if adj[name] < 0.001 else f"{adj[name]:.3f}")
        rows.append([name, f"{s['MAP']:.4f}", f"[{lo:.3f}, {hi:.3f}]", f"{s['P@10']:.4f}", f"{s['nDCG@10']:.4f}",
                     f"{s['R@1000']:.4f}", sig, f"{oracle:.4f} ({best_all})"])
    rows.sort(key=lambda r: -float(r[1]))

    md = ["# Ranking models (5-fold cross-validation over topics)", "",
          f"{len(setup.relevant)} topics with at least one relevant document; 1,000 documents per topic. "
          "Parameters are chosen on the training folds by MAP; every number is from held-out folds.", "",
          md_table(rows, ["model", "MAP", "MAP 95% CI", "P@10", "nDCG@10", "R@1000",
                          "p vs BM25 (Holm)", "best single setting on all topics (not held out)"]), "",
          "p: two-sided paired randomization test on per-topic AP (100,000 permutations), Holm-adjusted over the models.",
          "", "## Settings chosen per fold", ""]
    md += [f"- {n}: " + "; ".join(chosen[n]) for n in MODELS]

    # common default settings, without any tuning
    defaults = {"BM25 (k1=1.2, b=0.75)": (MODELS["BM25"][0], {"k1": 1.2, "b": 0.75}),
                "QL Dirichlet (mu=2000)": (MODELS["QL Dirichlet"][0], {"mu": 2000})}
    md += ["", "## Untuned defaults (all topics)", ""]
    out["defaults"] = {}
    for label, (fn, params) in defaults.items():
        s = summary(ev.evaluate_run(run_model(setup, fn, params), setup.relevant))
        out["defaults"][label] = s
        md.append(f"- {label}: MAP {s['MAP']:.4f}, P@10 {s['P@10']:.4f}, R@1000 {s['R@1000']:.4f}")
    (RESULTS / "models.md").write_text("\n".join(md) + "\n")
    save_json(out, RESULTS / "models.json")
    save_json(per_setting, RESULTS / "per_setting.json")
    save_json({n: held[n] for n in MODELS}, RESULTS / "per_topic_cv.json")
    plots(per_setting, out)


def map_of(per_setting, model, setting):
    return ev.mean(per_setting[model][setting]["MAP"].values())


def plots(per_setting, out) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # CV MAP per model with bootstrap intervals
    names = sorted(out["models"], key=lambda n: out["models"][n]["MAP"])
    vals = [out["models"][n]["MAP"] for n in names]
    err = np.array([[v - out["models"][n]["MAP_95ci"][0], out["models"][n]["MAP_95ci"][1] - v] for n, v in zip(names, vals)]).T
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = ["#2c5364" if n.startswith(("BM25", "PL2")) else "#7fa7b5" if n.startswith("QL") else "#b0b0b0" for n in names]
    ax.barh(names, vals, xerr=err, color=colors, error_kw={"lw": 1, "ecolor": "#555"})
    for i, (n, v) in enumerate(zip(names, vals)):
        ax.text(out["models"][n]["MAP_95ci"][1] + 0.008, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("MAP (5-fold cross-validation, 95% bootstrap interval)")
    ax.set_xlim(0, max(out["models"][n]["MAP_95ci"][1] for n in names) + 0.07)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2c5364", label="BM25 family / DFR"), Patch(color="#7fa7b5", label="query likelihood"),
                       Patch(color="#b0b0b0", label="vector space")], frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "models_map.png", dpi=150)
    plt.close(fig)

    # parameter sensitivity
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    ax = axes[0]
    for k1 in [0.6, 1.2, 1.8]:
        bs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ax.plot(bs, [map_of(per_setting, "BM25", f"k1={k1},b={b}") for b in bs], "o-", ms=3, label=f"k1 = {k1}")
    ax.set_xlabel("BM25 b (length normalisation)")
    ax.set_ylabel("MAP (all topics)")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    mus = [100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000]
    ax.plot(mus, [map_of(per_setting, "QL Dirichlet", f"mu={m}") for m in mus], "o-", ms=3, color="#2c5364")
    ax.set_xscale("log")
    ax.set_xlabel("Dirichlet μ")
    ax = axes[2]
    lams = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ax.plot(lams, [map_of(per_setting, "QL Jelinek-Mercer", f"lam={l}") for l in lams], "o-", ms=3, color="#7fa7b5")
    ax.set_xlabel("Jelinek-Mercer λ (collection weight)")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "parameter_sensitivity.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
