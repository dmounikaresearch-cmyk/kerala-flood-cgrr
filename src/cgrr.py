"""CGRR experiment pipeline for Paper 4 (Natural Hazards submission).

Calibration-guided robust reallocation of flood emergency resources:
conformal sets -> demand ranges -> robust MIP (P) -> forecast-triggered
re-planning, evaluated in a hindcast digital twin against baselines
B1-B5 and ablations A1-A3.

Equation numbers refer to the manuscript (Sections 6 and 7).

Key design point: plans depend only on information available at
decision time (forecasts, reported road closures), never on the random
realization, so each method is planned ONCE per event and then scored
against all Monte Carlo realizations of demand.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import pulp
from scipy.stats import wilcoxon

CLASSES = ["low", "medium", "high", "severe"]
K = len(CLASSES)
INF = np.inf


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    alpha: float = 0.10            # conformal miscoverage level
    eta: float = 0.10              # trigger threshold, Eq. (trigger)
    beta: float = 0.5              # travel-time increase per metre of predicted depth
    tau_max: float = 12.0          # h, maximum useful travel time
    lam: float = 0.01              # weight of travel time in objective
    mu: float = 0.05               # weight of relocations in objective
    rho: float = 0.5               # equity floor share of lower demand
    M: float = 1000.0              # penalty of the soft equity constraint
    vuln_quantile: float = 0.75    # units with w above this quantile form I^v
    gamma_quantile: float = 0.95   # quantile used to calibrate Gamma
    fixed_width: float = 0.30      # +-30 % ranges for B4 and A1
    b4_gamma_frac: float = 0.5     # hand-tuned Gamma of B4 as share of units
    n_runs: int = 200
    n_scen_b3: int = 30
    b5_search: int = 60            # random-search candidates for B5
    # Beta(a,b): share of EXPOSED people needing rescue/evacuation/medical help
    # in a period (mean 0.02) -- ASSUMED; calibrate from rescue/evacuation records
    need_beta: tuple = (2.0, 98.0)
    # fraction of TOTAL unit population needing help, per risk class (used if
    # calibration data has no exposed_frac column) -- ASSUMED, justify in paper
    delta_default: tuple = (0.00001, 0.0001, 0.0005, 0.002)
    depth_proxy_m: float = 1.5     # depth proxy if forecasts lack pred_depth_m
    resources: dict = field(default_factory=lambda: {
        # kappa: people served per decision period; hmax: max crossable depth (m)
        "team":    {"kappa": 150, "hmax": 0.3, "mode": "road"},
        "boat":    {"kappa": 60,  "hmax": 99.0, "mode": "boat"},
        "medical": {"kappa": 80,  "hmax": 0.3, "mode": "road"},
    })
    solver_time: int = 30
    seed: int = 42


# ---------------------------------------------------------------------------
# Event container
# ---------------------------------------------------------------------------
@dataclass
class Event:
    name: str
    units: list
    bases: list
    res_names: list
    periods: list
    P: np.ndarray          # [I] total population
    w: np.ndarray          # [I] vulnerability weight
    v: np.ndarray          # [I] vulnerability multiplier
    probs: np.ndarray      # [T,I,K] forecast class probabilities
    true_cls: np.ndarray   # [T,I] observed class index
    pred_depth: np.ndarray # [T,I] predicted inundation depth (m)
    exposed: np.ndarray    # [T,I] observed exposed population
    tau0: np.ndarray       # [J,I] normal travel time (h)
    real_tt: dict          # mode -> [T,J,I] realized travel time, inf = impassable
    stocks: np.ndarray     # [R,J]

    @property
    def T(self): return len(self.periods)
    @property
    def I(self): return len(self.units)
    @property
    def J(self): return len(self.bases)
    @property
    def R(self): return len(self.res_names)

    def known_closed(self, t, mode):
        """Closures reported by the previous period (known at decision time)."""
        if t == 0:
            return np.zeros((self.J, self.I), bool)
        return ~np.isfinite(self.real_tt[mode][t - 1])


def load_event(name, forecasts, units, exposure, tau0, route_times, stocks, cfg,
               class_thresholds=(0.01, 0.05, 0.15)):
    """Build an Event from DataFrames (formats in the notebook / README).

    forecasts   : period, datetime, district, p_low..p_severe, [pred_depth_m], [true_class]
    units       : district, total_pop, [w], [v]
    exposure    : date, district, exposed_pop   (from notebook 01; forward-filled)
    tau0        : origin, district, tau0_hours
    route_times : date, origin, district, mode, tt_hours (blank/inf = impassable)
    stocks      : base, resource, count
    """
    f = forecasts.copy()
    f["datetime"] = pd.to_datetime(f["datetime"])
    per = f[["period", "datetime"]].drop_duplicates().sort_values("period")
    periods = list(per["datetime"])
    unit_list = list(units["district"])
    I, T = len(unit_list), len(periods)
    uidx = {u: i for i, u in enumerate(unit_list)}
    pidx = {p: t for t, p in enumerate(per["period"])}

    probs = np.zeros((T, I, K))
    depth = np.full((T, I), np.nan)
    true = np.full((T, I), -1)
    for r in f.itertuples(index=False):
        t, i = pidx[r.period], uidx[r.district]
        probs[t, i] = [getattr(r, f"p_{c}") for c in CLASSES]
        if "pred_depth_m" in f.columns:
            depth[t, i] = r.pred_depth_m
        if "true_class" in f.columns and isinstance(r.true_class, str):
            true[t, i] = CLASSES.index(r.true_class)
    probs = probs / probs.sum(-1, keepdims=True)
    if np.isnan(depth).any():
        print("NOTE: pred_depth_m missing -> depth proxy = "
              f"{cfg.depth_proxy_m} m x P(high or severe). State this in the paper.")
        proxy = cfg.depth_proxy_m * (probs[..., 2] + probs[..., 3])
        depth = np.where(np.isnan(depth), proxy, depth)

    # observed exposure, forward-filled from satellite dates to periods
    ex = exposure.copy()
    ex["date"] = pd.to_datetime(ex["date"])
    exposed = np.zeros((T, I))
    for i, u in enumerate(unit_list):
        s = ex[ex["district"] == u].sort_values("date").set_index("date")["exposed_pop"]
        if len(s):
            exposed[:, i] = s.reindex(periods, method="ffill").fillna(0).to_numpy()
    P = units["total_pop"].to_numpy(float)
    if (true < 0).any():
        frac = exposed / np.maximum(P, 1)[None, :]
        derived = np.digitize(frac, class_thresholds)
        print("NOTE: true_class missing for some rows -> derived from observed "
              f"exposed fraction with thresholds {class_thresholds}.")
        true = np.where(true < 0, derived, true)

    bases = list(dict.fromkeys(stocks["base"]))
    jidx = {b: j for j, b in enumerate(bases)}
    J = len(bases)
    tau = np.full((J, I), INF)
    for r in tau0.itertuples(index=False):
        if r.origin in jidx and r.district in uidx:
            tau[jidx[r.origin], uidx[r.district]] = r.tau0_hours

    rt = route_times.copy()
    rt["date"] = pd.to_datetime(rt["date"])
    rt["tt_hours"] = pd.to_numeric(rt["tt_hours"], errors="coerce").fillna(INF)
    modes = sorted({v["mode"] for v in cfg.resources.values()})
    real = {m: np.repeat(tau[None], T, 0).copy() for m in modes}
    for m in modes:
        sub = rt[rt["mode"] == m]
        for (o, d), g in sub.groupby(["origin", "district"]):
            if o not in jidx or d not in uidx:
                continue
            s = g.sort_values("date").set_index("date")["tt_hours"]
            s = s[~s.index.duplicated(keep="last")]
            vals = s.reindex(periods, method="ffill")
            vals = vals.fillna(tau[jidx[o], uidx[d]]).to_numpy()
            real[m][:, jidx[o], uidx[d]] = vals

    res_names = list(cfg.resources)
    st = np.zeros((len(res_names), J), int)
    for r in stocks.itertuples(index=False):
        st[res_names.index(r.resource), jidx[r.base]] += int(r.count)

    w = units["w"].to_numpy(float) if "w" in units else np.ones(I)
    v = units["v"].to_numpy(float) if "v" in units else np.ones(I)
    return Event(name, unit_list, bases, res_names, periods, P, w, v, probs, true,
                 depth, exposed, tau, real, st)


# ---------------------------------------------------------------------------
# Conformal prediction, damage fractions, Gamma (Sect. 6.1)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Routes for planning (Eq. travel)
# ---------------------------------------------------------------------------
def planning_routes(ev, t, cfg, use_hazard=True, learned_closed=None):
    R, J, I = ev.R, ev.J, ev.I
    tau = np.zeros((R, J, I))
    allowed = np.ones((R, J, I), bool)
    for r, name in enumerate(ev.res_names):
        spec = cfg.resources[name]
        base = ev.tau0.copy()
        if use_hazard:
            h = np.broadcast_to(ev.pred_depth[t][None, :], (J, I))
            tau[r] = base * (1 + (cfg.beta * h if spec["mode"] == "road" else 0))
            allowed[r] &= h <= spec["hmax"]
            allowed[r] &= ~ev.known_closed(t, spec["mode"])
        else:
            tau[r] = base
            if learned_closed is not None:
                allowed[r] &= ~learned_closed[spec["mode"]]
        allowed[r] &= np.isfinite(tau[r]) & (tau[r] <= cfg.tau_max)
        if ev.stocks[r].sum() == 0:
            allowed[r] = False
        allowed[r] &= (ev.stocks[r] > 0)[:, None]
    return np.where(np.isfinite(tau), tau, cfg.tau_max * 10), allowed


# ---------------------------------------------------------------------------
# Optimization models
# ---------------------------------------------------------------------------
def _solver(cfg):
    return pulp.HiGHS(msg=False, timeLimit=cfg.solver_time)


def _xvars(allowed, stocks):
    idx = list(zip(*np.nonzero(allowed)))
    return {k: pulp.LpVariable("x_%d_%d_%d" % k, 0, int(stocks[k[0], k[1]]), cat="Integer")
            for k in idx}


def _extract(x, shape):
    out = np.zeros(shape, int)
    for k, var in x.items():
        out[k] = int(round(var.value() or 0))
    return out


def solve_robust(dhat, dU, dL, Gamma, tau, allowed, stocks, kappa, w, vul, xbar, cfg):
    """Model (P), Eqs. (3)-(9). Gamma = 0 and dU = dhat gives the nominal model."""
    R, J, I = allowed.shape
    m = pulp.LpProblem("P", pulp.LpMinimize)
    x = _xvars(allowed, stocks)
    u = [pulp.LpVariable(f"u{i}", 0) for i in range(I)]
    a = [pulp.LpVariable(f"a{i}", 0) for i in range(I)]
    pi = [pulp.LpVariable(f"pi{i}", 0) for i in range(I)]
    th = pulp.LpVariable("theta", 0)
    e = {i: pulp.LpVariable(f"e{i}", 0) for i in np.nonzero(vul)[0]}
    cov = [pulp.lpSum(kappa[r] * x[(r, j, i)] for r in range(R) for j in range(J)
                      if (r, j, i) in x) for i in range(I)]
    obj = (pulp.lpSum(w[i] * u[i] for i in range(I)) + Gamma * th + pulp.lpSum(pi)
           + cfg.lam * pulp.lpSum(tau[k] * x[k] for k in x)
           + cfg.M * pulp.lpSum(w[i] * e[i] for i in e))
    if xbar is not None:
        yp = {k: pulp.LpVariable("yp_%d_%d_%d" % k, 0) for k in x}
        ym = {k: pulp.LpVariable("ym_%d_%d_%d" % k, 0) for k in x}
        for k in x:
            m += x[k] - int(xbar[k]) == yp[k] - ym[k]
        obj += cfg.mu * pulp.lpSum(yp[k] + ym[k] for k in x)
    m += obj
    for i in range(I):
        m += u[i] >= dhat[i] - cov[i]
        m += a[i] >= dU[i] - cov[i]
        m += th + pi[i] >= w[i] * (a[i] - u[i])
    for r in range(R):
        for j in range(J):
            terms = [x[(r, j, i)] for i in range(I) if (r, j, i) in x]
            if terms:
                m += pulp.lpSum(terms) <= int(stocks[r, j])
    for i in e:
        m += cov[i] + e[i] >= cfg.rho * dL[i]
    m.solve(_solver(cfg))
    return _extract(x, (R, J, I))


def solve_stochastic(scen, tau, allowed, stocks, kappa, w, xbar, cfg):
    """B3: minimise expected weighted shortfall over sampled demand scenarios."""
    R, J, I = allowed.shape
    S = scen.shape[0]
    m = pulp.LpProblem("B3", pulp.LpMinimize)
    x = _xvars(allowed, stocks)
    z = [[pulp.LpVariable(f"z{s}_{i}", 0) for i in range(I)] for s in range(S)]
    cov = [pulp.lpSum(kappa[r] * x[(r, j, i)] for r in range(R) for j in range(J)
                      if (r, j, i) in x) for i in range(I)]
    obj = (pulp.lpSum(w[i] * z[s][i] for s in range(S) for i in range(I)) / S
           + cfg.lam * pulp.lpSum(tau[k] * x[k] for k in x))
    if xbar is not None:
        yp = {k: pulp.LpVariable("yp_%d_%d_%d" % k, 0) for k in x}
        ym = {k: pulp.LpVariable("ym_%d_%d_%d" % k, 0) for k in x}
        for k in x:
            m += x[k] - int(xbar[k]) == yp[k] - ym[k]
        obj += cfg.mu * pulp.lpSum(yp[k] + ym[k] for k in x)
    m += obj
    for s in range(S):
        for i in range(I):
            m += z[s][i] >= scen[s, i] - cov[i]
    for r in range(R):
        for j in range(J):
            terms = [x[(r, j, i)] for i in range(I) if (r, j, i) in x]
            if terms:
                m += pulp.lpSum(terms) <= int(stocks[r, j])
    m.solve(_solver(cfg))
    return _extract(x, (R, J, I))


def greedy(score, need, tau, allowed, stocks, kappa):
    """Assign resource units one at a time to the reachable unit with the best score.

    score(i_array, remaining, r, j) -> values for candidate units.
    """
    R, J, I = allowed.shape
    x = np.zeros((R, J, I), int)
    rem = need.astype(float).copy()
    for r in range(R):
        order = np.argsort(tau[r].min(1))
        for j in order:
            for _ in range(int(stocks[r, j])):
                cand = np.nonzero(allowed[r, j])[0]
                if len(cand) == 0:
                    break
                vals = score(cand, rem, r, j)
                i = cand[int(np.argmax(vals))]
                x[r, j, i] += 1
                rem[i] -= kappa[r]
    return x


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
METHOD_INFO = {
    "B1": "Static deterministic allocation",
    "B2": "Proportional-to-risk heuristic",
    "B3": "Two-stage stochastic programming",
    "B4": "Rolling budgeted robust (fixed budget)",
    "B5": "Learned score-based dispatcher",
    "CGRR": "Proposed",
    "A1": "Fixed-width ranges",
    "A2a": "Re-plan every period",
    "A2b": "Never re-plan",
    "A3": "No inundation/landslide routing",
}


@dataclass
class Plan:
    x: np.ndarray          # [T,R,J,I]
    replans: np.ndarray    # [T] bool
    dU: np.ndarray         # [T,I] upper demand used for planning (for exceedance)
    reason: list           # [T] trigger reason or ''


def _sample_scenarios(probs_t, delta, P, v, S, rng):
    cum = probs_t.cumsum(-1)
    u = rng.random((S, probs_t.shape[0], 1))
    cls = (u > cum[None]).sum(-1).clip(max=K - 1)
    return (v * P)[None, :] * delta[cls]


def train_b5(ev, delta, qhat, cfg, rng):
    """B5: fit greedy score weights by random search on outcomes simulated
    from the forecasts only (never sees realized demand or roads)."""
    kappa = np.array([cfg.resources[r]["kappa"] for r in ev.res_names])
    best, best_val = None, INF
    cands = [np.array([1.0, 0.0, 0.0])] + [rng.uniform([0, 0, 0], [1, 1, 0.5])
                                            for _ in range(cfg.b5_search - 1)]
    T_train = list(range(0, ev.T, max(1, ev.T // 12)))
    sims = {t: _sample_scenarios(ev.probs[t], delta, ev.P, ev.v, 10, rng) for t in T_train}
    for th in cands:
        val = 0.0
        for t in T_train:
            tau, allowed = planning_routes(ev, t, cfg)
            _, dh, du = bridge(ev.probs[t:t + 1], conformal_sets(ev.probs[t:t + 1], qhat),
                               delta, ev.P, ev.v)
            dhat, spread = dh[0], np.maximum(du[0] - dh[0], 0)
            sc = lambda c, rem, r, j: (th[0] * ev.w[c] * np.maximum(rem[c], 0)
                                       + th[1] * ev.w[c] * np.maximum(spread[c], 0)
                                       - th[2] * tau[r, j, c] * ev.w[c] * dhat[c] / cfg.tau_max)
            x = greedy(sc, dhat, tau, allowed, ev.stocks, kappa)
            cov = np.einsum("r,rji->i", kappa, x)
            val += (ev.w[None] * np.maximum(sims[t] - cov[None], 0)).sum(1).mean()
        if val < best_val:
            best, best_val = th, val
    return best


def plan_method(ev, method, cfg, qhat, delta, Gamma, probs=None, b5_theta=None, rng=None):
    probs = ev.probs if probs is None else probs
    T, R, J, I = ev.T, ev.R, ev.J, ev.I
    kappa = np.array([cfg.resources[r]["kappa"] for r in ev.res_names])
    cap = float((kappa[:, None] * ev.stocks).sum())
    sets = conformal_sets(probs, qhat)
    dL, dhat, dU = bridge(probs, sets, delta, ev.P, ev.v)
    if method in ("B4", "A1"):
        dU = dhat * (1 + cfg.fixed_width)
        dL = dhat * (1 - cfg.fixed_width)
    G = {"B4": int(math.ceil(cfg.b4_gamma_frac * I))}.get(method, Gamma)
    vul = ev.w >= np.quantile(ev.w, cfg.vuln_quantile)
    if np.allclose(ev.w, ev.w[0]):
        vul = np.zeros(I, bool)
    # upper class used by the escalation trigger
    upper = sets.cumsum(-1).argmax(-1) if method != "A1" else probs.argmax(-1)

    xs = np.zeros((T, R, J, I), int)
    rep = np.zeros(T, bool)
    reasons = [""] * T
    xbar, ref = None, None
    learned = {m: np.zeros((J, I), bool) for m in ev.real_tt}
    every = method in ("B2", "B3", "B4", "B5", "A2a")
    for t in range(T):
        use_hazard = method != "A3"
        if method == "A3" and t > 0:          # learn closures only from own failures
            for r, name in enumerate(ev.res_names):
                mode = cfg.resources[name]["mode"]
                failed = (xs[t - 1, r] > 0) & ~np.isfinite(ev.real_tt[mode][t - 1])
                learned[mode] |= failed
        tau, allowed = planning_routes(ev, t, cfg, use_hazard,
                                       learned if method == "A3" else None)
        # ---- decide whether to (re)plan
        if xbar is None:
            fire, why = True, "initial"
        elif method in ("B1", "A2b"):
            fire, why = False, ""
        elif every:
            fire, why = True, "every period"
        else:
            d_ref = ref["dhat"]
            delta_t = (ev.w * np.abs(dhat[t] - d_ref)).sum() / max((ev.w * d_ref).sum(), cap)
            esc = bool((upper[t] > ref["upper"]).any())
            fail = bool(((xbar > 0) & ~allowed).any())
            fire = delta_t > cfg.eta or esc or fail
            why = ",".join(n for n, f in (("shift", delta_t > cfg.eta), ("escalation", esc),
                                          ("route", fail)) if f)
        if fire:
            if method == "B2":
                sc = lambda c, rem, r, j: ev.w[c] * np.maximum(rem[c], 0) - 1e-6 * tau[r, j, c]
                x = greedy(sc, dhat[t], tau, allowed, ev.stocks, kappa)
            elif method == "B5":
                th = b5_theta
                spread = np.maximum(dU[t] - dhat[t], 0)
                sc = lambda c, rem, r, j: (th[0] * ev.w[c] * np.maximum(rem[c], 0)
                                           + th[1] * ev.w[c] * spread[c]
                                           - th[2] * tau[r, j, c] * ev.w[c] * dhat[t][c] / cfg.tau_max)
                x = greedy(sc, dhat[t], tau, allowed, ev.stocks, kappa)
            elif method == "B3":
                scen = _sample_scenarios(probs[t], delta, ev.P, ev.v, cfg.n_scen_b3, rng)
                x = solve_stochastic(scen, tau, allowed, ev.stocks, kappa, ev.w, xbar, cfg)
            elif method == "B1":
                x = solve_robust(dhat[t], dhat[t], dL[t], 0, tau, allowed, ev.stocks,
                                 kappa, ev.w, vul, None, cfg)
            else:  # B4, CGRR, A1, A2a, A2b, A3
                x = solve_robust(dhat[t], dU[t], dL[t], G, tau, allowed, ev.stocks,
                                 kappa, ev.w, vul, xbar, cfg)
            xbar = x
            ref = {"dhat": dhat[t].copy(), "upper": upper[t].copy()}
            rep[t], reasons[t] = True, why
        xs[t] = xbar
    return Plan(xs, rep, dU, reasons)


# ---------------------------------------------------------------------------
# Hindcast evaluation (Sect. 7)
# ---------------------------------------------------------------------------
def draw_need(ev, cfg, n_runs, seed):
    rng = np.random.default_rng(seed)
    return rng.beta(*cfg.need_beta, size=(n_runs, ev.I))


def evaluate(ev, plan, cfg, need, demand_mult=1.0, sets=None):
    T, R, J, I = plan.x.shape
    N = need.shape[0]
    kappa = np.array([cfg.resources[r]["kappa"] for r in ev.res_names])
    d = (ev.v * ev.exposed)[None] * need[:, None, :] * demand_mult      # [N,T,I]
    cov = np.zeros((T, I))
    cov_plan = np.einsum("r,trji->ti", kappa, plan.x)
    rt = np.full((T, I), cfg.tau_max)
    failed = np.zeros(T)
    for r, name in enumerate(ev.res_names):
        tt = ev.real_tt[cfg.resources[name]["mode"]]                    # [T,J,I]
        ok = np.isfinite(tt) & (tt <= cfg.tau_max)
        xr = plan.x[:, r]
        cov += kappa[r] * (xr * ok).sum(1)
        failed += (xr * ~ok).sum((1, 2))
        cand = np.where((xr > 0) & ok, tt, INF).min(1)
        rt = np.minimum(rt, np.where(np.isfinite(cand), cand, cfg.tau_max))
    s = np.maximum(d - cov[None], 0)                                    # [N,T,I]
    D = d.sum((1, 2))
    out = {
        "worst_unit_weighted_unmet": (ev.w[None] * s.sum(1)).max(1),
        "total_weighted_unmet": (ev.w[None, None] * s).sum((1, 2)),
        "total_unmet_people": s.sum((1, 2)),
        "coverage_pct": 100 * (1 - s.sum((1, 2)) / np.maximum(D, 1e-9)),
        "mean_response_h": (d * rt[None]).sum((1, 2)) / np.maximum(D, 1e-9),
    }
    flat_rt = np.broadcast_to(rt[None], d.shape).reshape(N, -1)
    flat_d = d.reshape(N, -1)
    p95, w6 = np.zeros(N), np.zeros(N)
    order = np.argsort(flat_rt[0])
    for n in range(N):
        cw = np.cumsum(flat_d[n, order]) / max(flat_d[n].sum(), 1e-9)
        p95[n] = flat_rt[0, order][min(np.searchsorted(cw, 0.95), len(cw) - 1)]
        w6[n] = 100 * flat_d[n][flat_rt[0] <= 6].sum() / max(flat_d[n].sum(), 1e-9)
    out["p95_response_h"] = p95
    out["share_within_6h_pct"] = w6
    severe = (ev.true_cls == K - 1)[None]
    out["missed_high_risk"] = (severe & (s > 0.5 * d) & (d > 0)).sum((1, 2)).astype(float)
    reloc = np.maximum(np.diff(plan.x, axis=0), 0).sum() if T > 1 else 0
    out["relocations"] = np.full(N, float(reloc))
    out["replans"] = np.full(N, float(plan.replans.sum()))
    out["failed_dispatches"] = np.full(N, float(failed.sum()))
    a = np.maximum(plan.dU - cov_plan, 0)
    out["exceedance_rate"] = (s > a[None] + 1e-9).mean((1, 2))
    if sets is not None:
        hit = np.take_along_axis(sets, ev.true_cls[..., None], -1)[..., 0]
        out["set_coverage"] = np.full(N, hit.mean())
        out["mean_set_size"] = np.full(N, sets.sum(-1).mean())
    return pd.DataFrame(out)


METRICS_LOWER_BETTER = {
    "worst_unit_weighted_unmet": True, "total_weighted_unmet": True,
    "total_unmet_people": True, "coverage_pct": False, "mean_response_h": True,
    "p95_response_h": True, "share_within_6h_pct": False, "missed_high_risk": True,
    "relocations": True, "replans": True, "failed_dispatches": True,
}


def summarize(results, ref="CGRR"):
    """Median [IQR] per method, plus Wilcoxon signed-rank vs ref with Holm correction."""
    rows, tests = [], []
    for m, df in results.items():
        row = {"method": m}
        for c in METRICS_LOWER_BETTER:
            q1, med, q3 = np.percentile(df[c], [25, 50, 75])
            row[c] = med
            row[c + "_iqr"] = f"[{q1:.3g}, {q3:.3g}]"
        rows.append(row)
        if m == ref:
            continue
        for c in METRICS_LOWER_BETTER:
            diff = results[ref][c].to_numpy() - df[c].to_numpy()
            if np.allclose(diff, 0):
                p = 1.0
            elif np.allclose(diff, diff[0]):
                p = 0.0   # deterministic, constant non-zero difference across runs
            else:
                p = wilcoxon(results[ref][c], df[c]).pvalue
            tests.append({"method": m, "metric": c, "p": p,
                          "median_diff_ref_minus_method": float(np.median(diff))})
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["p_holm"] = np.nan
        for c, g in tests.groupby("metric"):
            order = g["p"].sort_values().index
            k = len(order)
            adj, run = [], 0.0
            for rank, idx in enumerate(order):
                run = max(run, min(1.0, (k - rank) * tests.loc[idx, "p"]))
                adj.append((idx, run))
            for idx, val in adj:
                tests.loc[idx, "p_holm"] = val
    return pd.DataFrame(rows), tests


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------
def degrade_probs(probs, true_cls):
    """Stress S3: double the forecast error vector (one-hot(true) - p)."""
    onehot = np.eye(K)[true_cls]
    p = np.clip(2 * probs - onehot, 1e-6, None)
    return p / p.sum(-1, keepdims=True)


def run_event(ev, cal, cfg, methods=None, log=print):
    methods = methods or list(METHOD_INFO)
    rng = np.random.default_rng(cfg.seed)
    cal_probs = cal[[f"p_{c}" for c in CLASSES]].to_numpy()
    cal_probs = cal_probs / cal_probs.sum(1, keepdims=True)
    cal_true = cal["true_class"].map(CLASSES.index).to_numpy()
    qhat = fit_conformal(cal_probs, cal_true, cfg.alpha)
    delta = estimate_delta(cal, cfg)
    Gamma = calibrate_gamma(cal, delta, cfg.gamma_quantile)
    log(f"[{ev.name}] qhat={qhat:.3f}  delta={np.round(delta, 5)}  Gamma={Gamma}")
    b5 = train_b5(ev, delta, qhat, cfg, rng) if "B5" in methods else None
    need = draw_need(ev, cfg, cfg.n_runs, cfg.seed)
    sets = conformal_sets(ev.probs, qhat)
    plans, res = {}, {}
    for m in methods:
        plans[m] = plan_method(ev, m, cfg, qhat, delta, Gamma, b5_theta=b5,
                               rng=np.random.default_rng(cfg.seed + 1))
        res[m] = evaluate(ev, plans[m], cfg, need, sets=sets)
        log(f"  {m:5s} replans={int(plans[m].replans.sum()):3d}  "
            f"coverage={res[m]['coverage_pct'].median():.1f}%")
    main, tests = summarize({m: res[m] for m in methods})
    info = {"qhat": qhat, "delta": delta, "Gamma": Gamma, "b5_theta": b5,
            "set_coverage": float(res[methods[0]]["set_coverage"].iloc[0]) if "set_coverage" in res[methods[0]] else None,
            "mean_set_size": float(sets.sum(-1).mean())}
    return {"plans": plans, "results": res, "summary": main, "tests": tests,
            "info": info, "need": need, "qhat": qhat, "delta": delta, "Gamma": Gamma, "b5": b5}


def run_stress(ev, cal, cfg, base, methods=("B1", "B2", "B3", "B4", "B5", "CGRR"), log=print):
    out = []
    qhat, delta, Gamma, b5, need = base["qhat"], base["delta"], base["Gamma"], base["b5"], base["need"]
    sets = conformal_sets(ev.probs, qhat)
    scen = {}
    # S1: demand 30 % higher than implied (realization only)
    scen["S1"] = {m: evaluate(ev, base["plans"][m], cfg, need, 1.3, sets) for m in methods}
    # S2: 25 % fewer resources
    ev2 = replace(ev, stocks=np.floor(ev.stocks * 0.75).astype(int))
    scen["S2"] = {m: evaluate(ev2, plan_method(ev2, m, cfg, qhat, delta, Gamma, b5_theta=b5,
                                               rng=np.random.default_rng(cfg.seed + 1)),
                              cfg, need, sets=sets) for m in methods}
    # S3: forecast errors doubled, no recalibration
    p3 = degrade_probs(ev.probs, ev.true_cls)
    ev3 = replace(ev, probs=p3)
    scen["S3"] = {m: evaluate(ev3, plan_method(ev3, m, cfg, qhat, delta, Gamma, b5_theta=b5,
                                               rng=np.random.default_rng(cfg.seed + 1)),
                              cfg, need, sets=conformal_sets(p3, qhat)) for m in methods}
    for s, res in scen.items():
        base_m = [m for m in methods if m != "CGRR"]
        best = min(base_m, key=lambda m: res[m]["total_weighted_unmet"].median())
        c, b = res["CGRR"], res[best]
        out.append({"scenario": s, "best_baseline": best,
                    "unmet_change_pct": 100 * (c["total_unmet_people"].median() / max(b["total_unmet_people"].median(), 1e-9) - 1),
                    "p95_change_pct": 100 * (c["p95_response_h"].median() / max(b["p95_response_h"].median(), 1e-9) - 1),
                    "coverage_change_pp": c["coverage_pct"].median() - b["coverage_pct"].median()})
        log(f"  {s}: best baseline {best}")
    return pd.DataFrame(out), scen


def run_sensitivity(ev, cal, cfg, alphas=(0.05, 0.10, 0.20, 0.30),
                    etas=(0.0, 0.05, 0.10, 0.20, 0.30), log=print):
    rows = []
    need = draw_need(ev, cfg, cfg.n_runs, cfg.seed)
    cal_probs = cal[[f"p_{c}" for c in CLASSES]].to_numpy()
    cal_probs = cal_probs / cal_probs.sum(1, keepdims=True)
    cal_true = cal["true_class"].map(CLASSES.index).to_numpy()
    delta = estimate_delta(cal, cfg)
    Gamma = calibrate_gamma(cal, delta, cfg.gamma_quantile)
    for a in alphas:
        qhat = fit_conformal(cal_probs, cal_true, a)
        sets = conformal_sets(ev.probs, qhat)
        for e in etas:
            c2 = replace(cfg, alpha=a, eta=e)
            p = plan_method(ev, "CGRR", c2, qhat, delta, Gamma)
            r = evaluate(ev, p, c2, need, sets=sets)
            rows.append({"alpha": a, "eta": e,
                         "set_coverage": r["set_coverage"].iloc[0],
                         "mean_set_size": r["mean_set_size"].iloc[0],
                         "total_weighted_unmet": r["total_weighted_unmet"].median(),
                         "worst_unit_weighted_unmet": r["worst_unit_weighted_unmet"].median(),
                         "coverage_pct": r["coverage_pct"].median(),
                         "relocations": r["relocations"].iloc[0],
                         "replans": r["replans"].iloc[0],
                         "exceedance_rate": r["exceedance_rate"].median()})
        log(f"  alpha={a} done")
    return pd.DataFrame(rows)


def latex_main_table(summary, methods):
    cols = ["worst_unit_weighted_unmet", "total_unmet_people", "mean_response_h",
            "p95_response_h", "coverage_pct", "missed_high_risk", "relocations"]
    names = {"B1": "B1 Static", "B2": "B2 Proportional", "B3": "B3 Stochastic",
             "B4": "B4 Fixed robust", "B5": "B5 Learned", "CGRR": "CGRR"}
    s = summary.set_index("method")
    lines = []
    for m in methods:
        vals = []
        for c in cols:
            v = s.loc[m, c]
            vals.append(f"{v:,.0f}" if abs(v) >= 100 else f"{v:.2f}")
        lines.append(f"{names.get(m, m)} & " + " & ".join(vals) + r" \\")
    return "\n".join(lines)
