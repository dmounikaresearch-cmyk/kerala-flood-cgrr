"""evaluation -- part of the CGRR pipeline (Paper 4). Equation numbers refer to the manuscript."""
from common import *
from dataclasses import replace
from scipy.stats import wilcoxon
from calibration import fit_conformal, conformal_sets, fit_temperature, temperature_scale, calibration_metrics, forecast_skill
from demand_estimation import bridge
from demand_estimation import estimate_delta, calibrate_gamma
from digital_twin import plan_method, train_b5, METHOD_INFO


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
    # fairness: Gini coefficient of event-level served share across units
    served = np.minimum(d, cov[None]).sum(1)                             # [N,I]
    share = served / np.maximum(d.sum(1), 1e-9)
    srt = np.sort(share, 1)
    n_u = share.shape[1]
    gini = (2 * (np.arange(1, n_u + 1)[None] * srt).sum(1) / (n_u * np.maximum(srt.sum(1), 1e-9))
            - (n_u + 1) / n_u)
    out["fairness_gini"] = gini
    dispatched = float(np.einsum("r,trji->", kappa, plan.x))
    out["utilization_pct"] = 100 * np.minimum(d, cov[None]).sum((1, 2)) / max(dispatched, 1e-9)
    inadequate = (cov[None] < 0.5 * d) & (d > 0)
    out["uncovered_exposed_pop"] = (inadequate * ev.exposed[None]).sum((1, 2))
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
    "fairness_gini": True, "utilization_pct": False, "uncovered_exposed_pop": True,
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


def degrade_probs(probs, true_cls):
    """Stress S3: double the forecast error vector (one-hot(true) - p)."""
    onehot = np.eye(K)[true_cls]
    p = np.clip(2 * probs - onehot, 1e-6, None)
    return p / p.sum(-1, keepdims=True)


def run_event(ev, cal, cfg, methods=None, log=print):
    """Main experiment for one event.

    Raw forecast probabilities -> temperature scaling fitted on the calibration
    periods -> conformal sets on the calibrated probabilities. FU plans on the
    raw probabilities; every other method uses the calibrated ones.
    """
    methods = methods or list(METHOD_INFO)
    rng = np.random.default_rng(cfg.seed)
    raw_cal = cal[[f"p_{c}" for c in CLASSES]].to_numpy()
    raw_cal = raw_cal / raw_cal.sum(1, keepdims=True)
    cal_true = cal["true_class"].map(CLASSES.index).to_numpy()
    Tmp = fit_temperature(raw_cal, cal_true)
    q_cal = temperature_scale(raw_cal, Tmp)
    cal_q = cal.copy()
    for k, c in enumerate(CLASSES):
        cal_q[f"p_{c}"] = q_cal[:, k]
    qhat = fit_conformal(q_cal, cal_true, cfg.alpha)
    delta = estimate_delta(cal_q, cfg)
    Gamma = calibrate_gamma(cal_q, delta, cfg.gamma_quantile)
    ev_raw = ev
    ev = replace(ev_raw, probs=temperature_scale(ev_raw.probs, Tmp))
    calib = {}
    for name, p, tr in (("calibration_raw", raw_cal, cal_true), ("calibration_scaled", q_cal, cal_true),
                        ("test_raw", ev_raw.probs, ev.true_cls), ("test_scaled", ev.probs, ev.true_cls)):
        calib[name], bins = calibration_metrics(p, tr)
        calib[name + "_bins"] = bins
    log(f"[{ev.name}] temperature T={Tmp:.3f}  qhat={qhat:.3f}  delta={np.round(delta, 6)}  Gamma={Gamma}")
    log(f"[{ev.name}] test ECE raw={calib['test_raw']['ECE']:.3f} -> scaled={calib['test_scaled']['ECE']:.3f}; "
        f"Brier {calib['test_raw']['Brier']:.3f} -> {calib['test_scaled']['Brier']:.3f}")
    skill = {"raw": forecast_skill(ev_raw.probs, ev.true_cls),
             "scaled": forecast_skill(ev.probs, ev.true_cls)}
    b5 = train_b5(ev, delta, qhat, cfg, rng) if "B5" in methods else None
    need = draw_need(ev, cfg, cfg.n_runs, cfg.seed)
    sets = conformal_sets(ev.probs, qhat)
    # demand-estimation error: forecast point demand vs realized demand (mean over runs)
    realized = ((ev.v * ev.exposed)[None] * need[:, None, :]).mean(0)
    demand_err = {}
    for name, e_ in (("raw", ev_raw), ("scaled", ev)):
        dh = bridge(e_.probs, conformal_sets(e_.probs, qhat), delta, e_.P, e_.v)[1]
        demand_err[name] = {"MAE": float(np.abs(dh - realized).mean()),
                            "RMSE": float(np.sqrt(((dh - realized) ** 2).mean()))}
    plans, res = {}, {}
    for m in methods:
        e_m = ev_raw if m == "FU" else ev
        if m == "FU":   # uncalibrated: conformal threshold refitted on raw probabilities
            q_m = fit_conformal(raw_cal, cal_true, cfg.alpha)
        else:
            q_m = qhat
        plans[m] = plan_method(e_m, m, cfg, q_m, delta, Gamma, b5_theta=b5,
                               rng=np.random.default_rng(cfg.seed + 1))
        res[m] = evaluate(ev, plans[m], cfg, need, sets=sets)
        log(f"  {m:5s} replans={int(plans[m].replans.sum()):3d}  "
            f"coverage={res[m]['coverage_pct'].median():.1f}%")
    main, tests = summarize({m: res[m] for m in methods})
    info = {"temperature": Tmp, "qhat": qhat, "delta": delta, "Gamma": Gamma, "b5_theta": b5,
            "set_coverage": float(res[methods[0]]["set_coverage"].iloc[0]),
            "mean_set_size": float(sets.sum(-1).mean())}
    return {"plans": plans, "results": res, "summary": main, "tests": tests, "info": info,
            "calibration": calib, "skill": skill, "demand_error": demand_err, "need": need, "qhat": qhat, "delta": delta, "Gamma": Gamma,
            "b5": b5, "ev_cal": ev, "temperature": Tmp}


def run_stress(ev, cal, cfg, base, methods=("B1", "B2", "B3", "B4", "B5", "FU", "FC", "CGRR"), log=print):
    out = []
    qhat, delta, Gamma, b5, need = base["qhat"], base["delta"], base["Gamma"], base["b5"], base["need"]
    ev = base["ev_cal"]
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
    # S4: sudden escalation -- demand in the 3 most exposed units doubles from mid-event,
    # unforeseen by the forecasts (realization only)
    top = np.argsort(-ev.exposed.max(0))[:3]
    mult = np.ones((ev.T, ev.I)); mult[ev.T // 2:, top] = 2.0
    scen["S4"] = {m: evaluate(ev, base["plans"][m], cfg, need, mult[None], sets) for m in methods}
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
    Tmp = fit_temperature(cal_probs, cal_true)
    cal_probs = temperature_scale(cal_probs, Tmp)
    cal = cal.copy()
    for k, c in enumerate(CLASSES):
        cal[f"p_{c}"] = cal_probs[:, k]
    ev = replace(ev, probs=temperature_scale(ev.probs, Tmp))
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
             "B4": "B4 Fixed robust", "B5": "B5 Learned", "FU": "FU Uncalibrated",
             "FC": "FC Calibrated", "CGRR": "CGRR"}
    s = summary.set_index("method")
    lines = []
    for m in methods:
        vals = []
        for c in cols:
            v = s.loc[m, c]
            vals.append(f"{v:,.0f}" if abs(v) >= 100 else f"{v:.2f}")
        lines.append(f"{names.get(m, m)} & " + " & ".join(vals) + r" \\")
    return "\n".join(lines)


def run_demand_sensitivity(ev, cfg, base, means=(0.01, 0.02, 0.05),
                           methods=("B1", "B2", "B3", "B4", "B5", "FU", "FC", "CGRR"), conc=100.0):
    """Alternative demand assumptions: mean share of exposed people needing help.
    Plans are unchanged; only the realized-demand model varies."""
    rows = []
    evc = base["ev_cal"]
    for mval in means:
        c2 = replace(cfg, need_beta=(mval * conc, (1 - mval) * conc))
        need = draw_need(evc, c2, cfg.n_runs, cfg.seed)
        for m in methods:
            r = evaluate(evc, base["plans"][m], c2, need)
            rows.append({"need_mean": mval, "method": m,
                         "total_unmet_people": r["total_unmet_people"].median(),
                         "coverage_pct": r["coverage_pct"].median(),
                         "worst_unit_weighted_unmet": r["worst_unit_weighted_unmet"].median()})
    return pd.DataFrame(rows)
