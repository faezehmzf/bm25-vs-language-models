"""Build the inverted index for one or all analyzer variants.

    python scripts/build_index.py                 # stop words removed + Porter stemming
    python scripts/build_index.py --all           # the four stop-word / stemming combinations
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexret.corpus import iter_articles  # noqa: E402
from lexret.experiment import ANALYZERS, DATA, RESULTS  # noqa: E402
from lexret.index import Index  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="stop-porter", choices=list(ANALYZERS))
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    variants = list(ANALYZERS) if args.all else [args.variant]

    articles = list(iter_articles(DATA / "LA.xml"))
    stats = {}
    for v in variants:
        t = time.time()
        index = Index.build(articles, ANALYZERS[v]())
        index.save(DATA / "index" / v)
        stats[v] = {"articles": len(articles), "indexed_documents": index.n_docs,
                    "empty_after_analysis": len(articles) - index.n_docs,
                    "vocabulary": len(index.vocab), "tokens": index.n_tokens,
                    "mean_document_length": round(index.avgdl, 1), "seconds": round(time.time() - t, 1)}
        print(v, stats[v], flush=True)
    out = RESULTS / "index_stats.json"
    old = json.loads(out.read_text()) if out.exists() else {}
    old.update(stats)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
