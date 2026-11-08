# Preprocessing: stop words and stemming

5-fold cross-validated scores; parameters tuned separately for every variant. p: paired randomization test on AP against stop words removed + Porter stemming.

| model | analyzer | vocabulary | tokens | MAP | P@10 | R@1000 | p vs stop-porter |
|---|---|---|---|---|---|---|---|
| BM25 | stop-porter | 181,082 | 39.8M | 0.4555 | 0.4045 | 0.9208 |  |
| BM25 | stop-nostem | 234,370 | 39.8M | 0.4164 | 0.3697 | 0.8878 | 0.029 |
| BM25 | nostop-porter | 181,148 | 62.9M | 0.4565 | 0.4022 | 0.9219 | 0.543 |
| BM25 | nostop-nostem | 234,526 | 62.9M | 0.4175 | 0.3730 | 0.8876 | 0.031 |
| QL Dirichlet | stop-porter | 181,082 | 39.8M | 0.4406 | 0.3955 | 0.9047 |  |
| QL Dirichlet | stop-nostem | 234,370 | 39.8M | 0.4072 | 0.3629 | 0.8835 | 0.049 |
| QL Dirichlet | nostop-porter | 181,148 | 62.9M | 0.4387 | 0.3989 | 0.9087 | 0.942 |
| QL Dirichlet | nostop-nostem | 234,526 | 62.9M | 0.3998 | 0.3629 | 0.8854 | 0.023 |
