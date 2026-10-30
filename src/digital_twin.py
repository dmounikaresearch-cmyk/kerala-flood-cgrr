"""digital_twin -- part of the CGRR pipeline (Paper 4). Equation numbers refer to the manuscript."""
from common import *
from uncertainty import conformal_sets
from demand_estimation import bridge
from resource_optimizer import planning_routes, solve_robust, solve_stochastic, greedy


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
    "FU": "Forecast-based allocation (uncalibrated)",
    "FC": "Calibration-guided allocation (calibrated, nominal)",
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
    every = method in ("B2", "B3", "B4", "B5", "A2a", "FU", "FC")
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
            elif method in ("FU", "FC"):   # nominal model on point forecasts, with relocation cost
                x = solve_robust(dhat[t], dhat[t], dL[t], 0, tau, allowed, ev.stocks,
                                 kappa, ev.w, vul, xbar, cfg)
            else:  # B4, CGRR, A1, A2a, A2b, A3
                x = solve_robust(dhat[t], dU[t], dL[t], G, tau, allowed, ev.stocks,
                                 kappa, ev.w, vul, xbar, cfg)
            xbar = x
            ref = {"dhat": dhat[t].copy(), "upper": upper[t].copy()}
            rep[t], reasons[t] = True, why
        xs[t] = xbar
    return Plan(xs, rep, dU, reasons)
