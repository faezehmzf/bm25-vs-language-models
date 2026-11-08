# Data

The experiments use the *Los Angeles Times 1994* articles from the CLEF English ad-hoc test
collection, with the title queries of topics 41–140 and their binary relevance judgments. The
collection is distributed under licence by ELRA as part of the CLEF test suites and is **not
included** in this repository.

Put four files here (names as used by the scripts):

| file | content | format |
|---|---|---|
| `LA.xml` | 113,005 articles | SGML; each article has `<DOCNO> LA010194-0001 </DOCNO>`, `<HEADLINE>` and `<TEXT>` |
| `topics.xml` | 100 title queries (topics 41–140) | `<DOC 41>` one word per line `</DOC>` |
| `qrels.txt` | relevance judgments | `qid 0 docno rel`, rel ∈ {0, 1} |
| `stopwords.txt` | English stop-word list (170 words) | one word per line |

`scripts/build_index.py --all` writes four indexes to `data/index/` (about 240 MB in total).

Statistics with stop words removed and Porter stemming: 110,282 non-empty articles (2,723 have no
headline or text), 181,082 terms, 39.8 million tokens, 361 tokens per article on average. 89 of the
100 topics have at least one relevant article (1,677 relevant articles in total, 1 to 107 per topic);
the other 11 are not evaluated.
