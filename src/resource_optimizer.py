"""resource_optimizer -- part of the CGRR pipeline (Paper 4). Equation numbers refer to the manuscript."""
from common import *


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
