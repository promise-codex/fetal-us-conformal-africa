"""
Step 4: the core experiment of the paper.

Question: after a model trained in Spain is adapted with a FEW local patients,
can it (1) refuse to answer when it is not sure, with a known error rate,
and (2) refuse views it was never taught? And how many local patients does
that take?

Nothing is retrained from scratch here. We reuse the network from Step 3 and
only adjust its last layer, which takes seconds. That lets us repeat every
experiment many times with different random patients (seeds), which matters
because each country has only 25 to 27 patients.

For each African country and each random seed:
  * The country's TRAINING patients (official split) are shuffled and cut in two:
      - adaptation pool (up to 6 patients): k of them are used to adjust the
        last layer, k = 0, 1, 2, 4, 6
      - calibration pool (the rest, about 6 patients): used only to set the
        conformal threshold
  * The country's TEST patients (official split) are used only for scoring.

Conformal prediction (split conformal, score = 1 - probability of the true plane):
  Instead of one answer, the model gives a SET of planes that contains the
  true plane at least 90% of the time, IF the calibration images come from the
  same kind of data as the test images. If the set has exactly one plane, the
  model answers. Otherwise it says "not sure" and passes the image to a person.
  We compare two ways of choosing the threshold:
      spain_cal : threshold from Spanish validation images (no local work)
      local_cal : threshold from the country's own calibration patients

Unknown views: the Spanish "Other" and "Maternal cervix" test images. No African
country has such images, so this part is a proxy and is reported as such.

Run:  python step4_abstain.py      (needs results/step3 from Step 3, about 2 to 5 minutes)
Output: results/step4/
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S3, OUT = Path("results/step3"), Path("results/step4")
OUT.mkdir(parents=True, exist_ok=True)
ALPHA = 0.10                      # target: true plane inside the set at least 90% of the time
KS = [0, 1, 2, 4, 6]
N_SEEDS = 20
LAMBDA = 1e-3                     # how strongly the adjusted last layer is pulled back to the Spanish one (fixed in advance)

T = pd.read_csv(S3 / "image_table.csv")
o = np.load(S3 / "outputs.npz")
F = o["feats"].astype(np.float64)
sd = torch.load(S3 / "model_spain.pt", map_location="cpu")
W0, b0 = sd["fc.weight"].double(), sd["fc.bias"].double()
lab = T.label.values
log = []
def say(s=""):
    print(s); log.append(str(s))

def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def head_probs(W, b, idx):
    return softmax(F[idx] @ W.numpy().T + b.numpy())

def adapt(idx_local):
    """Adjust the last layer on local images, staying close to the Spanish layer (L2-SP)."""
    if len(idx_local) == 0:
        return W0.clone(), b0.clone()
    X = torch.tensor(F[idx_local]); y = torch.tensor(lab[idx_local])
    W = W0.clone().requires_grad_(True); b = b0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([W, b], max_iter=200, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X @ W.T + b, y) + LAMBDA * ((W - W0) ** 2).sum()
        loss.backward(); return loss
    opt.step(closure)
    return W.detach(), b.detach()

def conformal_q(p_cal, y_cal, alpha=ALPHA):
    s = 1 - p_cal[np.arange(len(y_cal)), y_cal]
    n = len(s)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    return np.inf if level > 1 else np.quantile(s, level, method="higher")

def score_sets(p, y, q):
    sets = (1 - p) <= q
    size = sets.sum(1)
    covered = sets[np.arange(len(y)), y]
    single = size == 1
    wrong_single = single & ~covered
    return dict(coverage=covered.mean(), set_size=size.mean(), answered=single.mean(),
                wrong_answer=wrong_single.mean(),
                acc_when_answered=(covered[single].mean() if single.any() else np.nan))

# ---------------- Spanish reference ----------------
val = np.where(T.role == "val")[0]
q_spain = conformal_q(head_probs(W0, b0, val), lab[val])
te_es = np.where(T.role == "test_known")[0]
say("=== Spain, test patients (same data the model was trained on) ===")
say({k: round(v, 3) for k, v in score_sets(head_probs(W0, b0, te_es), lab[te_es], q_spain).items()})

# ---------------- Africa: k local patients x calibration source ----------------
rows = []
for c in sorted(T[T.dataset == "africa"].site.unique()):
    tr_pat = np.array(sorted(T[(T.site == c) & (T.split == "train")].patient.unique()))
    test = np.where((T.site == c) & (T.split == "test"))[0]
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(tr_pat)
        pool, calp = perm[:6], perm[6:]
        cal = np.where(T.patient.isin(calp))[0]
        for k in KS:
            adapt_idx = np.where(T.patient.isin(pool[:k]))[0]
            W, b = adapt(adapt_idx)
            p_test = head_probs(W, b, test)
            acc = (p_test.argmax(1) == lab[test]).mean()
            for src, q in [("spain_cal", q_spain), ("local_cal", conformal_q(head_probs(W, b, cal), lab[cal]))]:
                r = score_sets(p_test, lab[test], q)
                rows.append(dict(country=c, seed=seed, k=k, calibration=src, accuracy=acc,
                                 n_cal=len(cal), n_test=len(test), **r))
R = pd.DataFrame(rows)
R.to_csv(OUT / "all_runs.csv", index=False)

def summ(g):
    out = {}
    for m in ["accuracy", "coverage", "set_size", "answered", "wrong_answer", "acc_when_answered"]:
        out[m] = f"{g[m].mean():.3f} [{g[m].quantile(.05):.2f}-{g[m].quantile(.95):.2f}]"
    return pd.Series(out)
S = R.groupby(["country", "calibration", "k"]).apply(summ)
S.to_csv(OUT / "summary.csv")
say("\n=== Africa (mean over seeds, [5th-95th percentile across seeds]) ===")
say("k = local patients used to adjust the last layer; target coverage = 0.90")
with pd.option_context("display.width", 250, "display.max_rows", 500):
    say(S.to_string())

# ---------------- unknown views (out-of-distribution) ----------------
say("\n=== Unknown views: can the scores separate taught planes from untaught views? ===")
tr_es = np.where(T.role == "train")[0]
mus = np.stack([F[tr_es][lab[tr_es] == j].mean(0) for j in range(4)])
cov = LedoitWolf().fit(F[tr_es] - mus[lab[tr_es]])
P = cov.precision_
def maha(idx):
    d = np.stack([np.einsum("ij,jk,ik->i", F[idx] - m, P, F[idx] - m) for m in mus], 1)
    return d.min(1)                                  # distance to the nearest taught plane
def msp(idx):
    return head_probs(W0, b0, idx).max(1)            # top probability
def energy(idx):
    z = F[idx] @ W0.numpy().T + b0.numpy()
    return np.log(np.exp(z - z.max(1, keepdims=True)).sum(1)) + z.max(1)

unk = np.where((T.role == "spain_unknown") & (T.split == "test"))[0]
af_test = np.where((T.dataset == "africa") & (T.split == "test"))[0]
ood = []
for name, f, higher_is_known in [("max probability", msp, True), ("energy", energy, True), ("Mahalanobis", maha, False)]:
    sk, su, sa = f(te_es), f(unk), f(af_test)
    sgn = 1 if higher_is_known else -1
    auc_es = roc_auc_score(np.r_[np.ones(len(sk)), np.zeros(len(su))], sgn * np.r_[sk, su])
    auc_af = roc_auc_score(np.r_[np.ones(len(sa)), np.zeros(len(su))], sgn * np.r_[sa, su])
    thr = np.quantile(sgn * f(val), 0.05)           # keep 95% of Spanish validation images
    rej_af = (sgn * sa < thr).mean()
    acc_unk = (sgn * su >= thr).mean()
    # Most Spanish "Other" images come from the Aloka machine, so a score could
    # learn "Aloka" instead of "untaught view". Repeat on Voluson E6 images only.
    e6k = te_es[T.site.values[te_es] == "Spain_Voluson E6"]
    e6u = unk[T.site.values[unk] == "Spain_Voluson E6"]
    auc_e6 = roc_auc_score(np.r_[np.ones(len(e6k)), np.zeros(len(e6u))], sgn * np.r_[f(e6k), f(e6u)])
    ood.append(dict(score=name, AUROC_spain_known_vs_unknown=auc_es, AUROC_same_machine_E6=auc_e6,
                    AUROC_africa_known_vs_spain_unknown=auc_af,
                    africa_known_rejected=rej_af, spain_unknown_accepted=acc_unk))
    for c in sorted(T[T.dataset == "africa"].site.unique()):
        m = (T.site.values[af_test] == c)
        ood[-1][f"rejected_{c}"] = (sgn * sa[m] < thr).mean()
O = pd.DataFrame(ood); O.to_csv(OUT / "unknown_views.csv", index=False)
with pd.option_context("display.width", 250):
    say(O.round(3).to_string(index=False))
say("Threshold set to accept 95% of Spanish validation images. 'rejected_<country>' = share of real")
say("African plane images wrongly refused. AUROC Africa column is a proxy: the unknown views come from Spain.")
say(f"Unknown test images: {len(unk)}; their types: {T.plane.values[unk].tolist().count('other')} other, "
    f"{T.plane.values[unk].tolist().count('maternal cervix')} maternal cervix")

# ---------------- figures ----------------
countries = sorted(R.country.unique())
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for c in countries:
    for src, ls in [("spain_cal", ":"), ("local_cal", "-")]:
        g = R[(R.country == c) & (R.calibration == src)].groupby("k")
        ax[0].plot(KS, g.coverage.mean(), ls, marker="o", label=f"{c} ({src.replace('_', ' ')})" if src == "local_cal" else None)
        if src == "local_cal":
            ax[1].plot(KS, g.wrong_answer.mean(), "-", marker="o", label=c)
            ax[2].plot(KS, g.answered.mean(), "-", marker="o", label=c)
ax[0].axhline(1 - ALPHA, color="k", lw=1, ls="--")
ax[0].set_title("Coverage (dotted: Spanish threshold,\nsolid: local threshold)")
ax[1].set_title("Wrong single answers (local threshold)")
ax[2].set_title("Images answered without a person (local threshold)")
for a in ax:
    a.set_xlabel("Local patients used to adjust the model (k)"); a.set_xticks(KS)
ax[0].set_ylabel("Share of test images"); ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "local_patients_curves.png", dpi=200)

(OUT / "report.txt").write_text("\n".join(log))
print(f"\nSaved results in {OUT}/")
