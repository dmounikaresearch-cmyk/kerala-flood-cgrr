"""Reproduce all experiments: python reproduce.py [configs/experiment.yaml]

Writes Tables 3-7 (CSV + LaTeX rows), figures and a run log with input
checksums to results/. With smoke_test: true it uses random data (code check).
"""
import sys, os, json, time, hashlib, platform
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import yaml, numpy as np, pandas as pd
import digital_twin as dt, evaluation as evm, synthetic
from flood_forecasting import validate_forecasts

cfg_file = sys.argv[1] if len(sys.argv) > 1 else "configs/experiment.yaml"
C = yaml.safe_load(open(cfg_file))
m, x = C["model"], C["experiment"]
cfg = dt.Config(alpha=m["alpha"], eta=m["eta"], beta=m["beta"], tau_max=m["tau_max"], lam=m["lam"],
                mu=m["mu"], rho=m["rho"], fixed_width=m["fixed_width"], b4_gamma_frac=m["b4_gamma_frac"],
                need_beta=tuple(m["need_beta"]), resources=m["resources"], n_runs=x["n_runs"],
                n_scen_b3=x["n_scen_b3"], b5_search=x["b5_search"], seed=x["seed"])
OUT = C["results_dir"] + ("/SMOKE_TEST" if C["smoke_test"] else "")
for sub in ("tables", "figures", "metrics"):
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)
LOG = []
def log(s):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + str(s); print(line); LOG.append(line)
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16] if os.path.exists(p) else "missing"

log(f"START smoke_test={C['smoke_test']} python={platform.python_version()} numpy={np.__version__} pandas={pd.__version__}")
log("CONFIG " + json.dumps(C))
D, hashes = C["data_dir"], {}
for k, ev_name in enumerate(C["events"]):
    if C["smoke_test"]:
        d = synthetic.make(seed=k)
        ev = dt.load_event("SMOKE-" + ev_name, d["forecasts"], d["units"], d["exposure"], d["tau0"],
                           d["route_times"], d["stocks"], cfg)
        cal = d["calibration"]
    else:
        files = {"forecasts": f"{D}/forecasts_{ev_name}.csv", "stocks": f"{D}/stocks_{ev_name}.csv",
                 "route_times": f"{D}/route_times_{ev_name}.csv", "exposure": f"{D}/kerala_flood_exposure_all.csv",
                 "tau0": f"{D}/kerala_travel_time_tau0.csv", "calibration": f"{D}/calibration.csv"}
        hashes[ev_name] = {k2: sha(p) for k2, p in files.items()}
        fc = pd.read_csv(files["forecasts"]); validate_forecasts(fc)
        ex = pd.read_csv(files["exposure"]); ex = ex[ex["event"].astype(str) == ev_name]
        ex = ex[ex["imaged_frac"] >= 0.5]   # dates on which Sentinel-1 did not image the district are unobserved
        units = ex.groupby("district", as_index=False)["total_pop"].max()
        vf = f"{D}/vulnerability.csv"
        if os.path.exists(vf):
            units = units.merge(pd.read_csv(vf), on="district", how="left").fillna({"w": 1.0, "v": 1.0})
        ev = dt.load_event(ev_name, fc, units, ex, pd.read_csv(files["tau0"]), pd.read_csv(files["route_times"]),
                           pd.read_csv(files["stocks"]), cfg)
        cal = pd.read_csv(files["calibration"])
    res = evm.run_event(ev, cal, cfg, log=log)
    T = f"{OUT}/tables"
    pd.DataFrame({"raw": res["skill"]["raw"], "temperature_scaled": res["skill"]["scaled"]}).to_csv(f"{T}/table3_forecast_skill_{ev_name}.csv")
    pd.DataFrame({k2: v for k2, v in res["calibration"].items() if not k2.endswith("_bins")}).to_csv(f"{T}/table4_calibration_{ev_name}.csv")
    res["calibration"]["test_raw_bins"].to_csv(f"{OUT}/metrics/reliability_raw_{ev_name}.csv", index=False)
    res["calibration"]["test_scaled_bins"].to_csv(f"{OUT}/metrics/reliability_scaled_{ev_name}.csv", index=False)
    pd.DataFrame(res["demand_error"]).to_csv(f"{T}/table4b_demand_error_{ev_name}.csv")
    main_methods = ["B1", "B2", "B3", "B4", "B5", "FU", "FC", "CGRR"]
    s = res["summary"]
    s[s.method.isin(main_methods)].to_csv(f"{T}/table5_allocation_{ev_name}.csv", index=False)
    s[s.method.isin(["CGRR", "A1", "A2a", "A2b", "A3", "FC"])].to_csv(f"{T}/table7_ablation_{ev_name}.csv", index=False)
    res["tests"].to_csv(f"{OUT}/metrics/wilcoxon_holm_{ev_name}.csv", index=False)
    open(f"{T}/table5_latex_rows_{ev_name}.tex", "w").write(evm.latex_main_table(s, main_methods))
    st, _ = evm.run_stress(ev, cal, cfg, res, log=log)
    se = evm.run_sensitivity(ev, cal, cfg, alphas=x["sensitivity_alpha"], etas=x["sensitivity_eta"], log=log)
    dm = evm.run_demand_sensitivity(ev, cfg, res, means=x["demand_means"])
    st.to_csv(f"{T}/table6_stress_{ev_name}.csv", index=False)
    se.to_csv(f"{T}/table6b_sensitivity_alpha_eta_{ev_name}.csv", index=False)
    dm.to_csv(f"{T}/table6c_demand_assumptions_{ev_name}.csv", index=False)
    p = res["plans"]["CGRR"]
    pd.DataFrame({"period": range(ev.T), "datetime": ev.periods, "replanned": p.replans,
                  "reason": p.reason}).to_csv(f"{OUT}/metrics/trigger_log_{ev_name}.csv", index=False)
    log(f"[{ev_name}] done")
log("END")
with open(f"{OUT}/run_log.txt", "w") as f:
    f.write("\n".join(LOG) + "\nINPUT_SHA256_16 " + json.dumps(hashes) + "\n")
if C["smoke_test"]:
    print("\n*** SMOKE TEST: random data. Do NOT use these numbers in the paper. ***")
