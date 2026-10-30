"""uncertainty -- split conformal prediction sets for risk classes (Sect. 6.1)."""
from common import *


def fit_conformal(cal_probs, cal_true, alpha):
    """Split conformal, LAC score 1 - p_true. Returns threshold qhat."""
    n = len(cal_true)
    scores = 1.0 - cal_probs[np.arange(n), cal_true]
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def conformal_sets(probs, qhat):
    S = (1.0 - probs) <= qhat
    top = probs.argmax(-1)
    np.put_along_axis(S, top[..., None], True, axis=-1)   # never empty
    return S
