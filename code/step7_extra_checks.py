"""
Step 7: additional checks (calibration size, baselines, bootstrap). Uses only the Step 3 outputs
(no retraining), so it runs in a few minutes on a laptop CPU.

  1. Calibration size: how many local patients are needed for the threshold?
     The adaptation pool stays at 6 patients (k = 6, and k = 0 for reference);
     the calibration set uses 2, 3, 4 or 5 of the remaining patients, or all
     of them (6 patients; 7 in Malawi).
  2. Baselines for saying "not sure", on the same adjusted model:
       - plain cut-off 0.9 on the top probability (common practice)
       - cut-off tuned on Spanish validation images (keeps 90% of correct
         Spanish answers), applied unchanged in Africa
       - APS conformal sets (Romano et al. 2020), local threshold
       - our LAC conformal sets, Spanish and local threshold
  3. Patient-level bootstrap: 95% intervals over TEST patients for the main
     numbers (zero-shot accuracy; coverage, answered and wrong answers at k = 6;
     the full system with the local unknown-view gate).

Run:  python step7_extra_checks.py            (expects results/step3/)
Output: results/step7/
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.covariance import LedoitWolf

ap = argparse.ArgumentParser()
ap.add_argument("--step3", default="results/step3")
ap.add_argument("--out", default="results/step7")
ap.add_argument("--draws", type=int, default=20)
ap.add_argument("--boot", type=int, default=2000)
args = ap.parse_args()
S3, OUT = Path(args.step3), Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
ALPHA, LAMBDA = 0.10, 1e-3

T = pd.read_csv(S3 / "image_table.csv")
F = np.load(S3 / "outputs.npz")["feats"].astype(np.float64)
sd = torch.load(S3 / "model_spain.pt", map_location="cpu")
W0, b0 = sd["fc.weight"].double().numpy(), sd["fc.bias"].double().numpy()
lab = T.label.values
log = []
def say(s=""):
    print(s, flush=True); log.append(str(s))

def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def adapt(idx):
    if len(idx) == 0:
        return W0.copy(), b0.copy()
    X = torch.tensor(F[idx]); y = torch.tensor(lab[idx]); W0t, b0t = torch.tensor(W0), torch.tensor(b0)
    W = W0t.clone().requires_grad_(True); b = b0t.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([W, b], max_iter=200, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        l = torch.nn.functional.cross_entropy(X @ W.T + b, y) + LAMBDA * ((W - W0t) ** 2).sum()
        l.backward(); return l
    opt.step(closure)
    return W.detach().numpy(), b.detach().numpy()

def q_level(s):
    n = len(s); lev = np.ceil((n + 1) * (1 - ALPHA)) / n
    return np.inf if lev > 1 else np.quantile(s, lev, method="higher")

def lac_sets(p, q):
    return (1 - p) <= q

def aps_scores(p):
    """Score of every class = total probability of classes ranked at or above it."""
    order = np.argsort(-p, 1); cum = np.cumsum(np.take_along_axis(p, order, 1), 1)
    S = np.empty_like(p); np.put_along_axis(S, order, cum, 1); return S

def per_image(sets, y):
    single = sets.sum(1) == 1; cov = sets[np.arange(len(y)), y]
    return cov.astype(float), single.astype(float), (single & ~cov).astype(float)

def cutoff(p, y, t):
    ans = p.max(1) >= t; wrong = ans & (p.argmax(1) != y)
    return np.full(len(y), np.nan), ans.astype(float), wrong.astype(float)

val = np.where(T.role == "val")[0]; tr = np.where(T.role == "train")[0]
p_val = softmax(F[val] @ W0.T + b0)
q_sp_lac = q_level(1 - p_val[np.arange(len(val)), lab[val]])
corr = p_val.argmax(1) == lab[val]
t_spain = np.quantile(p_val.max(1)[corr], 0.10)
say(f"Spanish LAC threshold q = {q_sp_lac:.4f};  Spanish-tuned cut-off t = {t_spain:.4f}")

# unknown-view gate (same as Step 5)
mus = np.stack([F[tr][lab[tr] == j].mean(0) for j in range(4)])
P = LedoitWolf().fit(F[tr] - mus[lab[tr]]).precision_
FP = F @ P; xPx = (F * FP).sum(1); centre = F[tr].mean(0)
def maha(idx, shift):
    C = mus + shift
    return -(xPx[idx, None] - 2 * FP[idx] @ C.T + np.einsum("cj,jk,ck->c", C, P, C)[None, :]).min(1)

countries = sorted(T[T.dataset == "africa"].site.unique())
sweep, base, img = [], [], []
for c in countries:
    tr_pat = np.array(sorted(T[(T.site == c) & (T.split == "train")].patient.unique()))
    te = np.where((T.site == c) & (T.split == "test"))[0]; y = lab[te]
    for d in range(args.draws):
        perm = np.random.default_rng(d).permutation(tr_pat); pool, calp = perm[:6], perm[6:]
        for k in [0, 2, 6]:
            ad = np.where(T.patient.isin(pool[:k]))[0]
            W, b = adapt(ad); pt = softmax(F[te] @ W.T + b)
            # ---- 1. calibration size sweep
            if k in (0, 6):
                for m in [2, 3, 4, 5, "all"]:
                    use = calp if m == "all" else calp[:m]
                    cal = np.where(T.patient.isin(use))[0]; pc = softmax(F[cal] @ W.T + b)
                    q = q_level(1 - pc[np.arange(len(cal)), lab[cal]])
                    cv, an, wr = per_image(lac_sets(pt, q), y)
                    sweep.append(dict(country=c, draw=d, k=k, cal_patients=str(m), n_cal_patients=len(use),
                                      n_cal=len(cal), finite_threshold=np.isfinite(q),
                                      coverage=cv.mean(), answered=an.mean(), wrong_answer=wr.mean()))
            # ---- 2. baselines
            cal = np.where(T.patient.isin(calp))[0]; pc = softmax(F[cal] @ W.T + b)
            q_loc = q_level(1 - pc[np.arange(len(cal)), lab[cal]])
            q_aps = q_level(aps_scores(pc)[np.arange(len(cal)), lab[cal]])
            meth = {
                "cutoff_0.9": cutoff(pt, y, 0.9),
                "cutoff_spain_tuned": cutoff(pt, y, t_spain),
                "LAC_spain": per_image(lac_sets(pt, q_sp_lac), y),
                "LAC_local": per_image(lac_sets(pt, q_loc), y),
                "APS_local": per_image(aps_scores(pt) <= q_aps, y),
            }
            for name, (cv, an, wr) in meth.items():
                base.append(dict(country=c, draw=d, k=k, method=name, coverage=np.nanmean(cv) if name[:3] != "cut" else np.nan,
                                 answered=an.mean(), wrong_answer=wr.mean(),
                                 wrong_among_answered=wr.sum() / an.sum() if an.sum() else np.nan))
            # ---- per-image records for the bootstrap
            if k in (0, 6):
                cv, an, wr = meth["LAC_local"]
                rec = dict(correct=(pt.argmax(1) == y).astype(float), cov=cv, ans=an, wrong=wr)
                if k == 6:
                    shift = F[ad].mean(0) - centre
                    keep = maha(te, shift) >= np.quantile(maha(cal, shift), 0.05)
                    rec["sys_ans"] = an * keep; rec["sys_wrong"] = wr * keep
                for key, v in rec.items():
                    img.append(pd.DataFrame(dict(country=c, draw=d, k=k, idx=te, patient=T.patient.values[te],
                                                 metric=key, value=v)))

SW = pd.DataFrame(sweep); B = pd.DataFrame(base); I = pd.concat(img, ignore_index=True)
SW.to_csv(OUT / "calibration_sweep_all.csv", index=False); B.to_csv(OUT / "baselines_all.csv", index=False)

say("\n=== 1. Calibration size (LAC, local threshold), mean over draws ===")
SW["cal_patients"] = SW.cal_patients.astype(str)
g = SW.groupby(["k", "cal_patients"]).agg(n_cal=("n_cal", "mean"), finite=("finite_threshold", "mean"),
        coverage=("coverage", "mean"), coverage_p05=("coverage", lambda v: v.quantile(.05)),
        answered=("answered", "mean"), wrong=("wrong_answer", "mean"), wrong_max=("wrong_answer", "max"))
say(g.round(3).to_string())
g.round(4).to_csv(OUT / "calibration_sweep_summary.csv")
say("\nPer country, k = 6:")
gc = SW[SW.k == 6].groupby(["country", "cal_patients"])[["coverage", "answered", "wrong_answer"]].mean().unstack(0)
say(gc.round(3).to_string())

say("\n=== 2. Baselines on the same adjusted model, mean over countries and draws ===")
gb = B.groupby(["k", "method"])[["coverage", "answered", "wrong_answer", "wrong_among_answered"]].mean()
gb["wrong_answer_max_country"] = B.groupby(["k", "method", "country"]).wrong_answer.mean().groupby(["k", "method"]).max()
say(gb.round(3).to_string()); gb.round(4).to_csv(OUT / "baselines_summary.csv")
say("\nPer country wrong answers, k = 0:")
say(B[B.k == 0].groupby(["method", "country"]).wrong_answer.mean().unstack().round(3).to_string())
B.groupby(["k", "method", "country"])[["coverage", "answered", "wrong_answer"]].mean().round(4).to_csv(OUT / "baselines_by_country.csv")

say(f"\n=== 3. Patient bootstrap ({args.boot} resamples of test patients), 95% intervals ===")
rng = np.random.default_rng(123); rows = []
per = I.groupby(["country", "k", "metric", "idx", "patient"]).value.mean().reset_index()   # average over draws
for (k, metric), sub in per.groupby(["k", "metric"]):
    for c in countries + ["All"]:
        s = sub if c == "All" else sub[sub.country == c]
        pats = s.patient.unique(); by = s.groupby("patient").value.agg(["sum", "count"]).loc[pats]
        sums, cnts = by["sum"].values, by["count"].values
        pick = rng.integers(0, len(pats), (args.boot, len(pats)))
        bs = sums[pick].sum(1) / cnts[pick].sum(1)
        rows.append(dict(k=k, metric=metric, country=c, n_patients=len(pats), estimate=sums.sum() / cnts.sum(),
                         lo=np.quantile(bs, .025), hi=np.quantile(bs, .975)))
CI = pd.DataFrame(rows); CI.to_csv(OUT / "bootstrap_ci.csv", index=False)
for (k, metric), sub in CI.groupby(["k", "metric"]):
    say(f"\n{metric}  (k = {k})")
    for _, r in sub.iterrows():
        say(f"  {r.country:8s} {r.estimate:.3f}  [{r.lo:.3f}, {r.hi:.3f}]  ({r.n_patients} patients)")
(OUT / "report.txt").write_text("\n".join(log))
