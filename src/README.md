| Module | Contents | Paper |
|---|---|---|
| `cgrr.py` | conformal sets, demand ranges, Gamma calibration, model (P), stochastic B3, greedy B2/B5, trigger rule, hindcast evaluation, Wilcoxon/Holm, stress, sensitivity | Sect. 6–8 |
| `routes.py` | realized road/boat travel times from flooded segments and landslide blockages | Sect. 5.3 |
| `synthetic.py` | random inputs for code testing only — never for results | — |
| `check_prop1.py` | numerical check of Proposition 1 (robust counterpart = brute-force worst case) | Sect. 6.3 |

PuLP must be 2.9.x (PuLP 4 changed its API). Solver: HiGHS via `highspy`.
