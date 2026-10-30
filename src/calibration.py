"""calibration -- temperature scaling, calibration metrics (ECE, Brier, NLL, reliability bins) and forecast skill."""
from common import *
from uncertainty import fit_conformal, conformal_sets  # re-exported for convenience


def temperature_scale(probs, T):
    """q = softmax(log p / T). T > 1 softens over-confident forecasts."""
    z = np.log(np.clip(probs, 1e-12, 1.0)) / T
    z = z - z.max(-1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(-1, keepdims=True)


def fit_temperature(cal_probs, cal_true):
    """Fit T by minimising negative log-likelihood on the calibration periods."""
    from scipy.optimize import minimize_scalar
    idx = np.arange(len(cal_true))
    nll = lambda T: -np.log(np.clip(temperature_scale(cal_probs, T)[idx, cal_true], 1e-12, 1)).mean()
    return float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)


def calibration_metrics(probs, true, n_bins=10):
    """ECE (top-label), multi-class Brier score, NLL and reliability-diagram bins."""
    probs = probs.reshape(-1, probs.shape[-1])
    true = np.asarray(true).reshape(-1)
    n = len(true)
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    for b in range(n_bins):
        m = (conf > edges[b]) & (conf <= edges[b + 1]) if b else (conf >= 0) & (conf <= edges[1])
        if m.sum():
            ece += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
            rows.append({"bin_low": edges[b], "bin_high": edges[b + 1], "count": int(m.sum()),
                         "mean_confidence": conf[m].mean(), "accuracy": correct[m].mean()})
    onehot = np.eye(probs.shape[1])[true]
    return {"ECE": float(ece),
            "Brier": float(((probs - onehot) ** 2).sum(1).mean()),
            "NLL": float(-np.log(np.clip(probs[np.arange(n), true], 1e-12, 1)).mean()),
            "accuracy": float(correct.mean())}, pd.DataFrame(rows)


def forecast_skill(probs, true):
    """Forecasting performance on observed classes: accuracy, macro precision/recall/F1,
    macro one-vs-rest ROC-AUC (classes absent from `true` are skipped)."""
    probs = probs.reshape(-1, probs.shape[-1]); true = np.asarray(true).reshape(-1)
    pred = probs.argmax(1)
    P, R, F, A = [], [], [], []
    for k in range(probs.shape[1]):
        tp = ((pred == k) & (true == k)).sum(); fp = ((pred == k) & (true != k)).sum()
        fn = ((pred != k) & (true == k)).sum()
        if (true == k).sum() == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        P.append(p); R.append(r); F.append(2 * p * r / (p + r) if p + r else 0.0)
        pos, neg = probs[true == k, k], probs[true != k, k]
        if len(neg):
            # Mann-Whitney formulation of AUC
            allv = np.concatenate([pos, neg]); ranks = pd.Series(allv).rank().to_numpy()
            A.append((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    return {"accuracy": float((pred == true).mean()), "precision_macro": float(np.mean(P)),
            "recall_macro": float(np.mean(R)), "f1_macro": float(np.mean(F)),
            "roc_auc_ovr_macro": float(np.mean(A)) if A else float("nan")}
