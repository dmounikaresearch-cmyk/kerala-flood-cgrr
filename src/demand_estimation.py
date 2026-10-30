"""demand_estimation -- part of the CGRR pipeline (Paper 4). Equation numbers refer to the manuscript."""
from common import *


def estimate_delta(cal, cfg):
    d = np.array(cfg.delta_default, float)
    if "exposed_frac" in cal.columns:
        need_mean = cfg.need_beta[0] / sum(cfg.need_beta)
        g = (cal["exposed_frac"] * need_mean).groupby(cal["true_class"]).mean()
        for k, c in enumerate(CLASSES):
            if c in g.index and np.isfinite(g[c]):
                d[k] = g[c]
    return np.maximum.accumulate(d)          # monotone in class


def calibrate_gamma(cal, delta, q):
    probs = cal[[f"p_{c}" for c in CLASSES]].to_numpy()
    probs = probs / probs.sum(1, keepdims=True)
    true = cal["true_class"].map(CLASSES.index).to_numpy()
    exceed = delta[true] > probs @ delta
    counts = pd.Series(exceed).groupby(cal["period"].to_numpy()).sum()
    return int(math.ceil(np.quantile(counts, q)))


def bridge(probs, sets, delta, P, v):
    """Eq. (1): demand ranges from conformal sets."""
    vP = (v * P)[None, :]
    big = np.where(sets, delta[None, None, :], -INF).max(-1)
    small = np.where(sets, delta[None, None, :], INF).min(-1)
    return vP * small, vP * (probs @ delta), vP * big   # dL, dhat, dU
