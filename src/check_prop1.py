import sys; sys.path.insert(0, sys.argv[1])
import collections, numpy as np, cgrr, synthetic
d = synthetic.make(); cfg = cgrr.Config(n_runs=20)
ev = cgrr.load_event("S", d["forecasts"], d["units"], d["exposure"], d["tau0"], d["route_times"], d["stocks"], cfg)
cal=d["calibration"]; cp=cal[[f"p_{c}" for c in cgrr.CLASSES]].to_numpy(); ct=cal.true_class.map(cgrr.CLASSES.index).to_numpy()
q=cgrr.fit_conformal(cp,ct,0.1); de=cgrr.estimate_delta(cal,cfg); G=cgrr.calibrate_gamma(cal,de,0.95)
p=cgrr.plan_method(ev,"CGRR",cfg,q,de,G)
c=collections.Counter(r2 for r in p.reason for r2 in r.split(",") if r2); print("trigger reasons:",c)
# sanity: robust counterpart exactness on one period (Prop. 1) via brute force
from itertools import combinations
t=20; S=cgrr.conformal_sets(ev.probs,q); dL,dh,dU=cgrr.bridge(ev.probs,S,de,ev.P,ev.v)
kap=np.array([cfg.resources[r]["kappa"] for r in ev.res_names]); cov=np.einsum("r,rji->i",kap,p.x[t])
u0=np.maximum(dh[t]-cov,0); a0=np.maximum(dU[t]-cov,0); g=ev.w*(a0-u0)
W=(ev.w*u0).sum()+np.sort(g)[::-1][:G].sum()
best=max((ev.w*u0).sum()+sum(g[list(s)]) for s in combinations(range(ev.I),min(G,ev.I)))
print("Prop1 check worst-case:",round(W,4),round(best,4))
print("set coverage %.3f (target 0.90)"%np.take_along_axis(S,ev.true_cls[...,None],-1).mean())
