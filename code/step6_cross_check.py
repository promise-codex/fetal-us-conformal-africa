"""
Step 6: does the conclusion hold for a different network and for repeated training?

PART A  Train the Spanish model again with two architectures and three training
        seeds each:  ResNet-18 (seeds 0, 1, 2) and DenseNet-169 (seeds 0, 1, 2).
        DenseNet-169 is the network used in the original African study.
        Same data, same split, same recipe as Step 3; only the network and the
        random seed change. Each finished model is saved, so if the run stops,
        start it again and finished models are skipped.

PART B  For every one of the 6 models, repeat the key analyses of Steps 4 and 5:
          - zero-shot accuracy per African country
          - conformal coverage / answered / wrong answers, Spanish vs local
            threshold, k = 0, 2, 6 local patients, 20 random draws
          - unknown-view AUROC at home, and the locally tuned Mahalanobis gate

PART C  Baseline: instead of adjusting only the last layer, fine-tune the WHOLE
        ResNet-18 (seed 0) on k = 2 and k = 6 local patients (5 draws each),
        then apply the same local conformal threshold.

Run (from the project folder):
    python step6_cross_check.py
Time: roughly 2.5 to 3 hours on an Apple M-series laptop, mostly DenseNet training.
Finished models are skipped on restart. Output: results/step6/
"""
import argparse, copy, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torchvision
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from common import KNOWN, load_tables, cached_images, to_tensor, device

ap = argparse.ArgumentParser()
ap.add_argument("--archs", nargs="+", default=["resnet18", "densenet169"])
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--no-pretrained", action="store_true", help="testing only")
ap.add_argument("--limit", type=int, default=0, help="testing only")
ap.add_argument("--draws", type=int, default=20)
args = ap.parse_args()

OUT = Path("results/step6"); OUT.mkdir(parents=True, exist_ok=True)
dev = device()
ALPHA, LAMBDA, KS = 0.10, 1e-3, [0, 2, 6]
log = []
def say(s=""):
    print(s, flush=True); log.append(str(s))

# ---------------- data and split: identical to Step 3 ----------------
T = load_tables()
if args.limit:
    es = T[T.dataset == "spain"].sample(args.limit, random_state=0)
    T = pd.concat([es, T[T.dataset == "africa"]], ignore_index=True)
X = cached_images(T)
rng0 = np.random.default_rng(0)
es_train_pat = T[(T.dataset == "spain") & (T.split == "train")].patient.unique()
val_pat = set(rng0.choice(es_train_pat, size=max(1, len(es_train_pat) // 10), replace=False))
sp = T.dataset == "spain"
T["role"] = "other"
T.loc[sp & T.known & (T.split == "train") & ~T.patient.isin(val_pat), "role"] = "train"
T.loc[sp & T.known & (T.split == "train") & T.patient.isin(val_pat), "role"] = "val"
T.loc[sp & T.known & (T.split == "test"), "role"] = "test_known"
T.loc[sp & ~T.known, "role"] = "spain_unknown"
T.loc[T.dataset == "africa", "role"] = "africa"
lab = T.label.values
y_all = torch.tensor(lab)
tr_idx, va_idx = np.where(T.role == "train")[0], np.where(T.role == "val")[0]
say(f"Images {len(T)}, train {len(tr_idx)}, val {len(va_idx)}, device {dev}")

# ---------------- networks ----------------
def build(arch):
    if arch == "resnet18":
        w = None if args.no_pretrained else torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        m = torchvision.models.resnet18(weights=w); m.fc = nn.Linear(m.fc.in_features, 4)
    elif arch == "densenet169":
        w = None if args.no_pretrained else torchvision.models.DenseNet169_Weights.IMAGENET1K_V1
        m = torchvision.models.densenet169(weights=w); m.classifier = nn.Linear(m.classifier.in_features, 4)
    else:
        raise SystemExit(f"unknown architecture {arch}")
    return m

def split(m):
    """Return (feature extractor, last layer) for either network."""
    if isinstance(m, torchvision.models.ResNet):
        return nn.Sequential(*list(m.children())[:-1], nn.Flatten()), m.fc
    return nn.Sequential(m.features, nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten()), m.classifier

aug = nn.Sequential(
    torchvision.transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
    torchvision.transforms.RandomRotation(10))

def run(m, idx, bs=64):
    body, head = split(m); m.eval(); fs, ls = [], []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            f = body(to_tensor(X[idx[i:i + bs]]).to(dev))
            fs.append(f.cpu().numpy()); ls.append(head(f).cpu().numpy())
    return np.concatenate(ls), np.concatenate(fs)

def train(arch, seed, folder):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = build(arch).to(dev)
    counts = np.bincount(lab[tr_idx], minlength=4)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (4 * counts), dtype=torch.float32).to(dev))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bs = 32 if arch == "resnet18" else 16
    best = -1; tmp = folder / "model_unfinished.pt"
    for ep in range(args.epochs):
        m.train(); t0 = time.time()
        perm = rng.permutation(tr_idx)
        for i0 in range(0, len(perm), bs):
            b = np.sort(perm[i0:i0 + bs])
            opt.zero_grad()
            loss = lossf(m(aug(to_tensor(X[b]).to(dev))), y_all[b].to(dev))
            loss.backward(); opt.step()
        sched.step()
        lv, _ = run(m, va_idx)
        acc = (lv.argmax(1) == lab[va_idx]).mean()
        say(f"  {arch} seed {seed} epoch {ep+1}/{args.epochs} val acc {acc:.3f} ({time.time()-t0:.0f}s)")
        if acc > best:
            best = acc; torch.save(m.state_dict(), tmp)
    m.load_state_dict(torch.load(tmp, map_location=dev))
    tmp.rename(folder / "model.pt")
    return m

# ---------------- analysis helpers (same rules as Steps 4 and 5) ----------------
def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def adapt(F, W0, b0, idx):
    if len(idx) == 0:
        return W0.copy(), b0.copy()
    Xf = torch.tensor(F[idx]); yy = torch.tensor(lab[idx])
    W0t, b0t = torch.tensor(W0), torch.tensor(b0)
    W = W0t.clone().requires_grad_(True); b = b0t.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([W, b], max_iter=200, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        l = nn.functional.cross_entropy(Xf @ W.T + b, yy) + LAMBDA * ((W - W0t) ** 2).sum()
        l.backward(); return l
    opt.step(closure)
    return W.detach().numpy(), b.detach().numpy()

def conformal_q(p, y):
    s = 1 - p[np.arange(len(y)), y]; n = len(s)
    lev = np.ceil((n + 1) * (1 - ALPHA)) / n
    return np.inf if lev > 1 else np.quantile(s, lev, method="higher")

def set_stats(p, y, q):
    sets = (1 - p) <= q; single = sets.sum(1) == 1; cov = sets[np.arange(len(y)), y]
    return cov.mean(), single.mean(), (single & ~cov).mean()

def draws_for(c, d):
    pats = np.array(sorted(T[(T.site == c) & (T.split == "train")].patient.unique()))
    perm = np.random.default_rng(d).permutation(pats)                     # same rule as Steps 4 and 5
    return perm[:6], perm[6:]

COUNTRIES = sorted(T[T.dataset == "africa"].site.unique())
TEST = {c: np.where((T.site == c) & (T.split == "test"))[0] for c in COUNTRIES}
te_es = np.where(T.role == "test_known")[0]
unk = np.where((T.role == "spain_unknown") & (T.split == "test"))[0]

def analyse(name, F, logits, W0, b0):
    F = F.astype(np.float64); W0 = W0.astype(np.float64); b0 = b0.astype(np.float64)
    rows = []
    def add(metric, value, country="all", k=-1, cal="-"):
        rows.append(dict(model=name, metric=metric, country=country, k=k, calibration=cal, value=float(value)))
    pred = logits.argmax(1)
    add("spain_test_accuracy", (pred[te_es] == lab[te_es]).mean())
    for c in COUNTRIES:
        add("zero_shot_accuracy", (pred[TEST[c]] == lab[TEST[c]]).mean(), c)
    p_val = softmax(F[va_idx] @ W0.T + b0); q_sp = conformal_q(p_val, lab[va_idx])
    # unknown views at home
    tr = np.where(T.role == "train")[0]
    mus = np.stack([F[tr][lab[tr] == j].mean(0) for j in range(4)])
    P = LedoitWolf().fit(F[tr] - mus[lab[tr]]).precision_
    FP = F @ P; xPx = (F * FP).sum(1)
    def maha(idx, shift):
        C = mus + shift
        d = xPx[idx, None] - 2 * FP[idx] @ C.T + np.einsum("cj,jk,ck->c", C, P, C)[None, :]
        return -d.min(1)
    zero = np.zeros(F.shape[1])
    lab_ood = np.r_[np.ones(len(te_es)), np.zeros(len(unk))]
    def energy(z):
        return np.log(np.exp(z - z.max(1, keepdims=True)).sum(1)) + z.max(1)
    z_home = F[np.r_[te_es, unk]] @ W0.T + b0
    add("AUROC_home_maxprob", roc_auc_score(lab_ood, softmax(z_home).max(1)))
    add("AUROC_home_energy", roc_auc_score(lab_ood, energy(z_home)))
    add("AUROC_home_mahalanobis", roc_auc_score(lab_ood, maha(np.r_[te_es, unk], zero)))
    centre = F[tr].mean(0)
    for c in COUNTRIES:
        te = TEST[c]
        for k in KS:
            acc_l, res = [], {("local", m): [] for m in ["cov", "ans", "wrong"]}
            res.update({("spain", m): [] for m in ["cov", "ans", "wrong"]})
            gate = {m: [] for m in ["refused", "auc", "ans", "wrong"]}
            for d in range(args.draws):
                pool, calp = draws_for(c, d)
                ad = np.where(T.patient.isin(pool[:k]))[0]; cal = np.where(T.patient.isin(calp))[0]
                W, b = adapt(F, W0, b0, ad)
                pt = softmax(F[te] @ W.T + b); acc_l.append((pt.argmax(1) == lab[te]).mean())
                for src, q in [("spain", q_sp), ("local", conformal_q(softmax(F[cal] @ W.T + b), lab[cal]))]:
                    cv, an, wr = set_stats(pt, lab[te], q)
                    res[(src, "cov")].append(cv); res[(src, "ans")].append(an); res[(src, "wrong")].append(wr)
                if k == 6:
                    shift = F[ad].mean(0) - centre
                    thr = np.quantile(maha(cal, shift), 0.05)
                    acc_g = maha(te, shift) >= thr
                    qloc = conformal_q(softmax(F[cal] @ W.T + b), lab[cal])
                    single = ((1 - pt) <= qloc).sum(1) == 1
                    ans = acc_g & single
                    gate["refused"].append(1 - acc_g.mean())
                    gate["auc"].append(roc_auc_score(lab_ood, maha(np.r_[te_es, unk], shift)))
                    gate["ans"].append(ans.mean()); gate["wrong"].append((ans & (pt.argmax(1) != lab[te])).mean())
            add("accuracy_after_adaptation", np.mean(acc_l), c, k)
            for (src, m), v in res.items():
                add({"cov": "coverage", "ans": "answered", "wrong": "wrong_answer"}[m], np.mean(v), c, k, src)
                if m == "cov":
                    add("coverage_p05", np.quantile(v, 0.05), c, k, src)
            if k == 6:
                for m, v in gate.items():
                    add(f"gate_{m}", np.mean(v), c, k, "local")
    return rows

# ---------------- PART A + B ----------------
all_rows = []
for arch in args.archs:
    for seed in args.seeds:
        name = f"{arch}_s{seed}"; folder = OUT / name; folder.mkdir(exist_ok=True)
        if (folder / "outputs.npz").exists():
            say(f"{name}: already trained, loading")
            o = np.load(folder / "outputs.npz"); logits, F, W0, b0 = o["logits"], o["feats"], o["W"], o["b"]
        else:
            say(f"{name}: training")
            m = train(arch, seed, folder)
            logits, F = run(m, np.arange(len(T)))
            _, head = split(m)
            W0, b0 = head.weight.detach().cpu().numpy(), head.bias.detach().cpu().numpy()
            np.savez_compressed(folder / "outputs.npz", logits=logits, feats=F, W=W0, b=b0)
            del m
            if dev == "mps":
                torch.mps.empty_cache()
        say(f"{name}: analysing")
        rows = analyse(name, F, logits, W0, b0)
        pd.DataFrame(rows).to_csv(folder / "analysis.csv", index=False)
        all_rows += rows

L = pd.DataFrame(all_rows)
L["arch"] = L.model.str.rsplit("_s", n=1).str[0]
L.to_csv(OUT / "analysis_all_models.csv", index=False)
S = (L.groupby(["arch", "metric", "country", "k", "calibration"]).value
       .agg(["mean", "std", "min", "max", "count"]).reset_index())
S.to_csv(OUT / "summary_by_model.csv", index=False)

def show(metric, k=-1, cal="-"):
    sub = S[(S.metric == metric) & (S.k == k) & (S.calibration == cal)]
    t = sub.pivot(index="country", columns="arch", values="mean").round(3)
    sd = sub.pivot(index="country", columns="arch", values="std").round(3)
    say(f"\n{metric}  k={k if k >= 0 else '-'}  threshold={cal}   (mean over training seeds; sd in brackets)")
    for c in t.index:
        say(f"  {c:9s} " + "   ".join(f"{a}: {t.loc[c, a]:.3f} ({sd.loc[c, a]:.3f})" for a in t.columns))

say("\n================ SUMMARY ACROSS ARCHITECTURES AND SEEDS ================")
show("spain_test_accuracy")
show("zero_shot_accuracy")
for cal in ["spain", "local"]:
    show("coverage", 0, cal)
show("coverage", 6, "local"); show("coverage_p05", 6, "local")
show("answered", 0, "local"); show("answered", 6, "local")
show("wrong_answer", 6, "local"); show("wrong_answer", 0, "spain")
show("AUROC_home_maxprob"); show("AUROC_home_energy"); show("AUROC_home_mahalanobis")
show("gate_refused", 6, "local"); show("gate_auc", 6, "local")
show("gate_ans", 6, "local"); show("gate_wrong", 6, "local")

# ---------------- PART C: full fine-tuning baseline ----------------
base = OUT / "resnet18_s0" / "model.pt"
if base.exists():
    say("\n================ PART C: full fine-tuning on local patients (ResNet-18 seed 0) ================")
    m0 = build("resnet18"); m0.load_state_dict(torch.load(base, map_location="cpu"))
    rows = []
    for c in COUNTRIES:
        te = TEST[c]
        for k in [2, 6]:
            for d in range(5):
                pool, calp = draws_for(c, d)
                ad = np.where(T.patient.isin(pool[:k]))[0]; cal = np.where(T.patient.isin(calp))[0]
                m = copy.deepcopy(m0).to(dev); torch.manual_seed(d)
                opt = torch.optim.AdamW(m.parameters(), lr=1e-5, weight_decay=1e-4)
                for step in range(30):
                    m.train()
                    for mod in m.modules():          # keep Spanish batch-norm statistics (batches are tiny)
                        if isinstance(mod, nn.BatchNorm2d):
                            mod.eval()
                    opt.zero_grad()
                    loss = nn.functional.cross_entropy(m(aug(to_tensor(X[np.sort(ad)]).to(dev))), y_all[np.sort(ad)].to(dev))
                    loss.backward(); opt.step()
                lt, _ = run(m, te); lc, _ = run(m, cal)
                pt, pc = softmax(lt), softmax(lc)
                cv, an, wr = set_stats(pt, lab[te], conformal_q(pc, lab[cal]))
                rows.append(dict(country=c, k=k, draw=d, accuracy=(pt.argmax(1) == lab[te]).mean(),
                                 coverage=cv, answered=an, wrong_answer=wr))
                del m
    Ff = pd.DataFrame(rows); Ff.to_csv(OUT / "full_finetune.csv", index=False)
    say(Ff.groupby(["country", "k"])[["accuracy", "coverage", "answered", "wrong_answer"]].mean().round(3).to_string())
    say("Compare with the last-layer rows for resnet18 in summary_by_model.csv (accuracy_after_adaptation, local threshold).")

(OUT / "report.txt").write_text("\n".join(log))
print(f"\nSaved results in {OUT}/")
