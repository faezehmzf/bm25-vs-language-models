# Pseudo-relevance feedback (EM mixture model)

Same 5-fold cross-validation as results/models.md; every number is from held-out folds.

| run | MAP | MAP 95% CI | P@10 | nDCG@10 | R@1000 |
|---|---|---|---|---|---|
| QL Dirichlet | 0.4406 | [0.381, 0.501] | 0.3955 | 0.5610 | 0.9047 |
| BM25 | 0.4555 | [0.394, 0.518] | 0.4045 | 0.5766 | 0.9208 |
| Dirichlet + EM feedback | 0.4525 | [0.392, 0.513] | 0.4225 | 0.5649 | 0.9390 |

- Feedback vs Dirichlet: p = 0.4430 (paired randomization test on AP); better on 58 topics, worse on 26, of 89.
- Feedback vs BM25: p = 0.8633.

Settings chosen per fold: mu=500,fb_docs=5,n_terms=20,alpha=0.2,lam_bg=0.8; mu=500,fb_docs=5,n_terms=20,alpha=0.2,lam_bg=0.8; mu=500,fb_docs=5,n_terms=50,alpha=0.6,lam_bg=0.8; mu=500,fb_docs=5,n_terms=50,alpha=0.4,lam_bg=0.8; mu=500,fb_docs=5,n_terms=50,alpha=0.4,lam_bg=0.8

## Expansion terms (Porter stems) for the topics feedback hurts and helps most (fold-0 setting: mu=500,fb_docs=5,n_terms=20,alpha=0.2,lam_bg=0.8)

| topic | title | AP Dirichlet → feedback | top added terms |
|---|---|---|---|
| 97 | referendum on the independence of moldova | 1.000 → 0.250 | quebec, parti, govern, canada, quebecoi, separatist, ukrain, canadian |
| 62 | northern japan earthquake | 1.000 → 0.500 | quak, magnitud, 000, structur, engin, wave, aftershock, energi |
| 103 | conflict of interests in italy | 0.360 → 0.554 | berlusconi, govern, polit, italian, minist, bossi, corrupt, prime |
| 85 | turquoise program in rwanda | 0.146 → 0.348 | french, u, n, food, aid, hutu, rebel, troop |
| 114 | civil war in afghanistan | 0.252 → 0.531 | kabul, hekmatyar, rabbani, dostum, soviet, u, 000, fight |
