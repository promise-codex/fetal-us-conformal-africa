"""
Step 5: two last checks before writing the results.

PART A. Setting the unknown-view gate on local patients
  Step 4 showed that a gate tuned in Spain wrongly refuses many real African
  images. Here we tune it on the same 6 to 7 local calibration patients used
  for conformal prediction, after adapting with k = 6 local patients.
    * max probability and energy: computed with the locally adjusted last layer
    * Mahalanobis: the Spanish class centres are shifted by the average
      difference between local and Spanish features ("mean shift"), so the
      score measures distance from the local look of each plane
  The threshold keeps 95% of the local calibration images.

  How we judge it, honestly:
    - wrongly_refused: share of real African plane images (test patients) refused
    - AUROC_spain_semantic: Spanish known vs Spanish unknown under the SAME
      locally tuned score. A threshold does not change this ranking, so it tells
      us whether the score still separates taught from untaught views after
      being moved to the local site. No African unknown views exist, so this is
      the cleanest check we can make.
    - full system on African test images: answer only if the gate accepts AND
      the conformal set has one plane. Share answered and share answered wrongly.

PART B. Saliency maps (Grad-CAM) for medical review
  For 8 test images per country we show where the network looked.
  Ghana and Uganda images carry measurement marks; if the heat sits on the
  marks rather than the anatomy, the model may be using a shortcut.
  A review sheet (review_sheet.csv) is written for clinical review.

Run:  python step5_local_gate_and_saliency.py      (a few minutes)
Output: results/step5/
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torchvision
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import KNOWN, to_tensor

S3, OUT = Path("results/step3"), Path("results/step5")
(OUT / "saliency").mkdir(parents=True, exist_ok=True)
ALPHA, K, N_SEEDS, LAMBDA = 0.10, 6, 20, 1e-3

T = pd.read_csv(S3 / "image_table.csv")
F = np.load(S3 / "outputs.npz")["feats"].astype(np.float64)
sd = torch.load(S3 / "model_spain.pt", map_location="cpu")
W0, b0 = sd["fc.weight"].double(), sd["fc.bias"].double()
lab = T.label.values
log = []
def say(s=""):
    print(s); log.append(str(s))

def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def adapt(idx):
    if len(idx) == 0:
        return W0.clone(), b0.clone()
    X = torch.tensor(F[idx]); y = torch.tensor(lab[idx])
    W = W0.clone().requires_grad_(True); b = b0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([W, b], max_iter=200, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        l = nn.functional.cross_entropy(X @ W.T + b, y) + LAMBDA * ((W - W0) ** 2).sum()
        l.backward(); return l
    opt.step(closure)
    return W.detach(), b.detach()

def conformal_q(p, y):
    s = 1 - p[np.arange(len(y)), y]; n = len(s)
    lev = np.ceil((n + 1) * (1 - ALPHA)) / n
    return np.inf if lev > 1 else np.quantile(s, lev, method="higher")

tr_es = np.where(T.role == "train")[0]
te_es = np.where(T.role == "test_known")[0]
unk = np.where((T.role == "spain_unknown") & (T.split == "test"))[0]
mus = np.stack([F[tr_es][lab[tr_es] == j].mean(0) for j in range(4)])
P = LedoitWolf().fit(F[tr_es] - mus[lab[tr_es]]).precision_
spain_centre = F[tr_es].mean(0)

# Fast Mahalanobis: (x-c)'P(x-c) = x'Px - 2 x'Pc + c'Pc, with x'Px computed once for all images
FP = F @ P
xPx = (F * FP).sum(1)

def maha(idx, shift):
    C = mus + shift                                        # 4 shifted class centres
    d = xPx[idx, None] - 2 * FP[idx] @ C.T + np.einsum("cj,jk,ck->c", C, P, C)[None, :]
    return -d.min(1)                          # higher = more familiar

# ===================== PART A =====================
rows = []
for c in sorted(T[T.dataset == "africa"].site.unique()):
    tr_pat = np.array(sorted(T[(T.site == c) & (T.split == "train")].patient.unique()))
    test = np.where((T.site == c) & (T.split == "test"))[0]
    for seed in range(N_SEEDS):
        perm = np.random.default_rng(seed).permutation(tr_pat)     # same split rule as Step 4
        ad = np.where(T.patient.isin(perm[:K]))[0]
        cal = np.where(T.patient.isin(perm[K:]))[0]
        W, b = adapt(ad)
        logit = lambda i: F[i] @ W.numpy().T + b.numpy()
        shift = F[ad].mean(0) - spain_centre
        scores = {
            "max probability": lambda i: softmax(logit(i)).max(1),
            "energy": lambda i: (lambda z: np.log(np.exp(z - z.max(1, keepdims=True)).sum(1)) + z.max(1))(logit(i)),
            "Mahalanobis (mean shift)": lambda i: maha(i, shift),
        }
        p_test = softmax(logit(test))
        q = conformal_q(softmax(logit(cal)), lab[cal])
        single = ((1 - p_test) <= q).sum(1) == 1
        correct = p_test.argmax(1) == lab[test]
        for name, f in scores.items():
            thr = np.quantile(f(cal), 0.05)
            acc_gate = f(test) >= thr
            auc = roc_auc_score(np.r_[np.ones(len(te_es)), np.zeros(len(unk))], np.r_[f(te_es), f(unk)])
            ans = acc_gate & single
            rows.append(dict(country=c, seed=seed, score=name,
                             wrongly_refused=1 - acc_gate.mean(),
                             AUROC_spain_semantic=auc,
                             answered=ans.mean(), wrong_answer=(ans & ~correct).mean(),
                             conformal_only_answered=single.mean()))
A = pd.DataFrame(rows); A.to_csv(OUT / "local_gate_all_runs.csv", index=False)
G = A.drop(columns="seed").groupby(["score", "country"])
S = pd.concat({"mean": G.mean(), "p05": G.quantile(.05), "p95": G.quantile(.95)}, axis=1)
S.to_csv(OUT / "local_gate_summary.csv")
say(f"=== PART A: unknown-view gate tuned on local patients (k = {K}, {N_SEEDS} seeds, means) ===")
with pd.option_context("display.width", 250, "display.max_columns", 50):
    say(A.groupby(["score", "country"])[["wrongly_refused", "AUROC_spain_semantic", "conformal_only_answered",
                                          "answered", "wrong_answer"]].mean().round(3).to_string())
say("\nwrongly_refused: real African plane images refused by the gate (target about 0.05)")
say("AUROC_spain_semantic: Spanish known vs unknown views under the locally tuned score")
say("answered / wrong_answer: full system (gate AND single conformal answer) on African test images")

# ===================== PART B =====================
say("\n=== PART B: Grad-CAM saliency maps ===")
X = np.load("results/cache/images_224.npy", mmap_mode="r")
net = torchvision.models.resnet18(weights=None); net.fc = nn.Linear(512, 4)
net.load_state_dict(sd); net.eval()
acts, grads = {}, {}
net.layer4.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
net.layer4.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))

def gradcam(i):
    x = to_tensor(X[i:i + 1]).requires_grad_(True)
    out = net(x); cls = int(out.argmax(1))
    net.zero_grad(); out[0, cls].backward()
    w = grads["g"].mean((2, 3), keepdim=True)
    cam = torch.relu((w * acts["a"]).sum(1)).detach()[0]
    cam = torch.nn.functional.interpolate(cam[None, None], size=X.shape[1:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam / (cam.max() + 1e-8)
    return cam.numpy(), cls, torch.softmax(out, 1)[0, cls].item()

review = []
rng = np.random.default_rng(0)
for c in ["Ghana", "Uganda", "Algeria", "Egypt", "Malawi"]:
    idx = np.where((T.site == c) & (T.split == "test"))[0]
    pick = []
    for pl in KNOWN:                                  # two per plane where available
        cand = idx[T.plane.values[idx] == pl]
        pick += list(rng.choice(cand, size=min(2, len(cand)), replace=False))
    fig, ax = plt.subplots(2, len(pick), figsize=(2.4 * len(pick), 5))
    for j, i in enumerate(pick):
        cam, cls, conf = gradcam(i)
        ax[0, j].imshow(X[i], cmap="gray"); ax[0, j].axis("off")
        ax[0, j].set_title(f"#{i}\ntrue: {T.plane[i]}", fontsize=9)
        ax[1, j].imshow(X[i], cmap="gray"); ax[1, j].imshow(cam, cmap="jet", alpha=0.4); ax[1, j].axis("off")
        ax[1, j].set_title(f"model: {KNOWN[cls]} ({conf:.2f})", fontsize=9,
                           color="green" if KNOWN[cls] == T.plane[i] else "red")
        review.append(dict(image_id=int(i), country=c, file=Path(T.path[i]).name, true_plane=T.plane[i],
                           model_plane=KNOWN[cls], confidence=round(conf, 3),
                           looks_at_anatomy="", looks_at_marks_or_text="", comment=""))
    fig.suptitle(f"{c}: top = image, bottom = where the model looked (Grad-CAM)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "saliency" / f"gradcam_{c}.png", dpi=200); plt.close(fig)
    say(f"saved saliency/gradcam_{c}.png")
pd.DataFrame(review).to_csv(OUT / "review_sheet.csv", index=False)
say("saved review_sheet.csv: for each image, a reviewer fills looks_at_anatomy (yes/no),")
say("looks_at_marks_or_text (yes/no) and a comment.")

(OUT / "report.txt").write_text("\n".join(log))
print(f"\nSaved results in {OUT}/")
