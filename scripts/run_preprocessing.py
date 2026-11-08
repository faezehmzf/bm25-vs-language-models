"""Effect of stop-word removal and Porter stemming on BM25 and Dirichlet query likelihood.

Each of the four analyzer variants has its own index (scripts/build_index.py --all). Parameters are
tuned by the same 5-fold cross-validation over topics, separately for every variant.

Outputs: results/preprocessing.md, results/preprocessing.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexret import evaluate as ev  # noqa: E402
from lexret.experiment import ANALYZERS, RESULTS, cross_validated, load_setup, md_table, save_json, summary, sweep  # noqa: E402
from lexret.models import MODELS  # noqa: E402

COMPARED = ["BM25", "QL Dirichlet"]
REFERENCE = "stop-porter"


def main() -> None:
    stats = json.loads((RESULTS / "index_stats.json").read_text())
    held = {}
    for variant in ANALYZERS:
        setup = load_setup(variant)
        for model in COMPARED:
            fn, grid = MODELS[model]
            held[(variant, model)], chosen = cross_validated(sweep(setup, fn, grid), setup.folds)
            print(variant, model, f"{summary(held[(variant, model)])['MAP']:.4f}", chosen, flush=True)

    rows, out = [], {}
    for model in COMPARED:
        for variant in ANALYZERS:
            s = summary(held[(variant, model)])
            p = None if variant == REFERENCE else ev.randomization_test(
                held[(variant, model)]["MAP"], held[(REFERENCE, model)]["MAP"])
            out[f"{model} | {variant}"] = {**s, "p_vs_stop_porter": p}
            rows.append([model, variant, f"{stats[variant]['vocabulary']:,}", f"{stats[variant]['tokens'] / 1e6:.1f}M",
                         f"{s['MAP']:.4f}", f"{s['P@10']:.4f}", f"{s['R@1000']:.4f}",
                         "" if p is None else ("< 0.001" if p < 0.001 else f"{p:.3f}")])
    md = ["# Preprocessing: stop words and stemming", "",
          "5-fold cross-validated scores; parameters tuned separately for every variant. "
          "p: paired randomization test on AP against stop words removed + Porter stemming.", "",
          md_table(rows, ["model", "analyzer", "vocabulary", "tokens", "MAP", "P@10", "R@1000", "p vs stop-porter"])]
    (RESULTS / "preprocessing.md").write_text("\n".join(md) + "\n")
    save_json(out, RESULTS / "preprocessing.json")


if __name__ == "__main__":
    main()
