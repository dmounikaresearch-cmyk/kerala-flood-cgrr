"""flood_forecasting -- interface to the forecasting layer.

The contribution of this paper is NOT a new flood prediction model. Any model
that issues, every decision period, class probabilities for each unit can be
plugged in (e.g. the authors' multi-hazard model driven by GPM IMERG rainfall
and GloFAS reforecast discharge). This module only validates its output.
"""
from common import *

REQUIRED = ["period", "datetime", "district"] + [f"p_{c}" for c in CLASSES]


def validate_forecasts(df, name="forecasts"):
    missing = [c for c in REQUIRED if c not in df.columns]
    assert not missing, f"{name}: missing columns {missing}"
    p = df[[f"p_{c}" for c in CLASSES]].to_numpy()
    assert (p >= 0).all(), f"{name}: negative probabilities"
    s = p.sum(1)
    assert np.allclose(s, 1, atol=1e-3), f"{name}: probabilities do not sum to 1 (min {s.min():.3f}, max {s.max():.3f})"
    if "true_class" in df.columns:
        bad = set(df["true_class"].dropna()) - set(CLASSES)
        assert not bad, f"{name}: unknown classes {bad}"
    dup = df.duplicated(["period", "district"]).sum()
    assert dup == 0, f"{name}: {dup} duplicated (period, district) rows"
    return True


def chronological_split_check(cal, test_start):
    """Calibration periods must end before the test event starts (no leakage)."""
    if "datetime" in cal.columns:
        last = pd.to_datetime(cal["datetime"]).max()
        assert last < pd.Timestamp(test_start), f"calibration data ({last}) overlaps test event ({test_start})"
    return True
