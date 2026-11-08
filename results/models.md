# Ranking models (5-fold cross-validation over topics)

89 topics with at least one relevant document; 1,000 documents per topic. Parameters are chosen on the training folds by MAP; every number is from held-out folds.

| model | MAP | MAP 95% CI | P@10 | nDCG@10 | R@1000 | p vs BM25 (Holm) | best single setting on all topics (not held out) |
|---|---|---|---|---|---|---|---|
| BM25 | 0.4555 | [0.394, 0.518] | 0.4045 | 0.5766 | 0.9208 |  | 0.4586 (k1=0.6,b=0.4) |
| BM25+ | 0.4540 | [0.393, 0.517] | 0.4022 | 0.5730 | 0.9184 | 0.746 | 0.4566 (k1=1.8,b=0.3) |
| BM25L | 0.4509 | [0.389, 0.513] | 0.3978 | 0.5668 | 0.9188 | 0.561 | 0.4555 (k1=1.2,b=0.4) |
| PL2 | 0.4410 | [0.381, 0.502] | 0.4022 | 0.5619 | 0.9166 | 0.399 | 0.4511 (c=10) |
| QL Dirichlet | 0.4406 | [0.381, 0.501] | 0.3955 | 0.5610 | 0.9047 | 0.303 | 0.4460 (mu=500) |
| QL two-stage | 0.4403 | [0.380, 0.501] | 0.3989 | 0.5603 | 0.9205 | 0.383 | 0.4502 (mu=1000,lam=0.3) |
| QL absolute discounting | 0.4196 | [0.360, 0.480] | 0.3921 | 0.5432 | 0.9097 | 0.020 | 0.4233 (delta=0.7) |
| QL Laplace | 0.4019 | [0.342, 0.462] | 0.3910 | 0.5324 | 0.8695 | 0.006 | 0.4019 (delta=0.01) |
| QL Jelinek-Mercer | 0.3992 | [0.341, 0.459] | 0.3685 | 0.5039 | 0.9071 | < 0.001 | 0.4037 (lam=0.6) |
| TF-IDF (lnc.ltc) | 0.2552 | [0.208, 0.304] | 0.2449 | 0.3309 | 0.8858 | < 0.001 | 0.2552 (-) |

p: two-sided paired randomization test on per-topic AP (100,000 permutations), Holm-adjusted over the models.

## Settings chosen per fold

- TF-IDF (lnc.ltc): -; -; -; -; -
- BM25: k1=1.2,b=0.3; k1=0.6,b=0.4; k1=0.6,b=0.4; k1=1.2,b=0.3; k1=1.2,b=0.3
- BM25L: k1=1.8,b=0.4; k1=1.2,b=0.4; k1=1.2,b=0.3; k1=1.8,b=0.4; k1=1.2,b=0.4
- BM25+: k1=1.8,b=0.3; k1=1.8,b=0.3; k1=1.8,b=0.3; k1=2.1,b=0.4; k1=1.8,b=0.3
- PL2: c=10; c=10; c=10; c=3; c=10
- QL Laplace: delta=0.01; delta=0.01; delta=0.01; delta=0.01; delta=0.01
- QL Jelinek-Mercer: lam=0.6; lam=0.6; lam=0.6; lam=0.4; lam=0.6
- QL Dirichlet: mu=500; mu=500; mu=750; mu=500; mu=500
- QL absolute discounting: delta=0.7; delta=0.9; delta=0.7; delta=0.7; delta=0.7
- QL two-stage: mu=1000,lam=0.3; mu=1000,lam=0.3; mu=500,lam=0.1; mu=250,lam=0.5; mu=1000,lam=0.3

## Untuned defaults (all topics)

- BM25 (k1=1.2, b=0.75): MAP 0.4285, P@10 0.3933, R@1000 0.9140
- QL Dirichlet (mu=2000): MAP 0.4310, P@10 0.3944, R@1000 0.9216
