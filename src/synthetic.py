"""SMOKE-TEST DATA ONLY. Generates random inputs in the exact formats the
pipeline expects, to check that the code runs. Never report these numbers."""
import numpy as np
import pandas as pd
from cgrr import CLASSES


def make(seed=0, I=14, J=5, T=40, n_cal_periods=60):
    rng = np.random.default_rng(seed)
    units = pd.DataFrame({"district": [f"D{i:02d}" for i in range(I)],
                          "total_pop": rng.integers(800_000, 3_500_000, I),
                          "w": rng.uniform(0.5, 1.5, I)})
    units["v"] = 1.0
    dates = pd.date_range("2018-08-08", periods=T, freq="6h")
    # latent severity rises then falls, differs by unit
    peak = rng.integers(10, 30, I)
    sev = np.array([[3.2 * np.exp(-((t - peak[i]) / 9.0) ** 2) for i in range(I)] for t in range(T)])
    noise = np.cumsum(rng.normal(0, 0.06, sev.shape), 0)      # slowly varying forecast error
    sev = sev + noise

    def probs_from(s):
        logits = -((np.arange(4)[None, None] - s[..., None]) ** 2)
        p = np.exp(logits); return p / p.sum(-1, keepdims=True)

    true = np.clip(np.round(sev + rng.normal(0, 0.5, sev.shape)), 0, 3).astype(int)
    pr = probs_from(sev)
    rows = []
    for t in range(T):
        for i in range(I):
            rows.append({"period": t, "datetime": dates[t], "district": units.district[i],
                         **{f"p_{c}": pr[t, i, k] for k, c in enumerate(CLASSES)},
                         "pred_depth_m": max(0, 0.6 * sev[t, i] + rng.normal(0, 0.1)),
                         "true_class": CLASSES[true[t, i]]})
    forecasts = pd.DataFrame(rows)
    # calibration periods (pre-event)
    sc = rng.uniform(0, 3.2, (n_cal_periods, I)) + rng.normal(0, 0.35, (n_cal_periods, I))
    tc = np.clip(np.round(sc + rng.normal(0, 0.5, sc.shape)), 0, 3).astype(int)
    pc = probs_from(sc)
    cal = pd.DataFrame([{"period": p, "district": units.district[i],
                         **{f"p_{c}": pc[p, i, k] for k, c in enumerate(CLASSES)},
                         "true_class": CLASSES[tc[p, i]],
                         "exposed_frac": [0.0005, 0.005, 0.02, 0.06][tc[p, i]] * rng.uniform(0.5, 1.5)}
                        for p in range(n_cal_periods) for i in range(I)])
    # observed exposure on satellite dates (every 4th period)
    frac = np.array([0.0005, 0.005, 0.02, 0.06])[true]
    exp_rows = [{"date": dates[t], "district": units.district[i],
                 "exposed_pop": units.total_pop[i] * frac[t, i] * rng.uniform(0.7, 1.3)}
                for t in range(0, T, 4) for i in range(I)]
    exposure = pd.DataFrame(exp_rows)
    bases = [f"B{j}" for j in range(J)]
    tau0 = pd.DataFrame([{"origin": b, "district": u, "tau0_hours": rng.uniform(0.5, 6)}
                         for b in bases for u in units.district])
    rt = []
    slid = rng.random((J, I)) < 0.08          # landslide blocks persist from period 12
    for t in range(T):
        for b in bases:
            for i, u in enumerate(units.district):
                base = float(tau0[(tau0.origin == b) & (tau0.district == u)].tau0_hours.iloc[0])
                j = bases.index(b)
                flooded = true[t, i] >= 3
                slide = bool(slid[j, i]) and t >= 12
                rt.append({"date": dates[t], "origin": b, "district": u, "mode": "road",
                           "tt_hours": np.inf if (flooded or slide) else base * rng.uniform(1, 1.5)})
                rt.append({"date": dates[t], "origin": b, "district": u, "mode": "boat",
                           "tt_hours": np.inf if slide else base * (2.0 if flooded else 1.0)})
    route_times = pd.DataFrame(rt)
    stocks = pd.DataFrame([{"base": b, "resource": r, "count": c}
                           for b in bases for r, c in (("team", 10), ("boat", 12), ("medical", 4))])
    return dict(forecasts=forecasts, units=units, exposure=exposure, tau0=tau0,
                route_times=route_times, stocks=stocks, calibration=cal)
