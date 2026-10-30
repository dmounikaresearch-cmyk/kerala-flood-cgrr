"""Numerical check of Proposition 1: the robust counterpart equals the brute-force
worst case over the budgeted uncertainty set (synthetic data)."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from itertools import combinations
import numpy as np
import synthetic, digital_twin as dt
from uncertainty import fit_conformal, conformal_sets
from demand_estimation import estimate_delta, calibrate_gamma, bridge
from common import CLASSES

d = synthetic.make(); cfg = dt.Config()
ev = dt.load_event("S", d["forecasts"], d["units"], d["exposure"], d["tau0"], d["route_times"], d["stocks"], cfg)
cal = d["calibration"]; cp = cal[[f"p_{c}" for c in CLASSES]].to_numpy(); ct = cal.true_class.map(CLASSES.index).to_numpy()
q = fit_conformal(cp, ct, 0.1); de = estimate_delta(cal, cfg); G = calibrate_gamma(cal, de, 0.95)
p = dt.plan_method(ev, "CGRR", cfg, q, de, G)
S = conformal_sets(ev.probs, q); dL, dh, dU = bridge(ev.probs, S, de, ev.P, ev.v)
kap = np.array([cfg.resources[r]["kappa"] for r in ev.res_names])
for t in (5, 20, 35):
    cov = np.einsum("r,rji->i", kap, p.x[t])
    u0 = np.maximum(dh[t] - cov, 0); a0 = np.maximum(dU[t] - cov, 0); g = ev.w * (a0 - u0)
    W = (ev.w * u0).sum() + np.sort(g)[::-1][:G].sum()
    brute = max((ev.w * u0).sum() + sum(g[list(s)]) for s in combinations(range(ev.I), min(G, ev.I)))
    print(f"t={t}: robust counterpart {W:.4f}  brute force {brute:.4f}  match={np.isclose(W, brute)}")
