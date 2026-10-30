Each subfolder holds notes and any config overrides for one experiment group:
- baseline/          B1-B5, FU (Table 5)
- calibration/       temperature scaling, ECE/Brier/NLL, reliability diagrams (Tables 3-4)
- robust_allocation/ B4, FC vs CGRR, ablations A1-A3 (Tables 5, 7)
- digital_twin/      trigger rule, stress S1-S4, alpha/eta and demand sensitivity (Table 6)
All are run by `reproduce.py` / `notebooks/02_experiments.ipynb`.
