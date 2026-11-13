"""
Step 3: teach a network the four planes using the Spanish data only,
then test it, untouched, on every African country.

Why this design
  * The model learns ONLY the four planes (abdomen, brain, femur, thorax).
    The Spanish "Other" and "Maternal cervix" images are kept aside and never
    shown in training. Later (Step 4) they play the role of "views the model
    was never taught", which is exactly what a clinic produces all day.
  * Images are cropped to the ultrasound fan first (see common.py), so the
    black frame and image size of each country cannot give the country away.
    We repeat the country test from Step 2 on the cropped images to check.
  * Splits follow the original Train flag in each CSV, which is by patient.
    10% of the Spanish training patients are kept for validation.

Outputs in results/step3/
  model_spain.pt              the trained network (re-used by Step 4)
  outputs.npz                 logits and 512 features for EVERY image
  report.txt                  accuracies, confusion matrices, country check
  zero_shot_by_country.png    accuracy per African country, no local training

Run:  python step3_train_spain.py        (about 10 to 25 minutes on a laptop GPU)
If it stops halfway, run it again: the image preprocessing is kept,
training restarts from the beginning.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torchvision
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import KNOWN, load_tables, cached_images, to_tensor, device

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--no-pretrained", action="store_true", help="testing only")
ap.add_argument("--limit", type=int, default=0, help="testing only: use N Spanish images")
args = ap.parse_args()

OUT = Path("results/step3"); OUT.mkdir(parents=True, exist_ok=True)
SEED = 0
rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
dev = device()
log = []
def say(s=""):
    print(s); log.append(str(s))

# ---------------- data ----------------
T = load_tables()
if args.limit:
    es = T[T.dataset == "spain"].sample(args.limit, random_state=SEED)
    T = pd.concat([es, T[T.dataset == "africa"]], ignore_index=True)
say(f"Images: {len(T)}  (Spain {sum(T.dataset=='spain')}, Africa {sum(T.dataset=='africa')})")
say(f"Spanish planes:\n{T[T.dataset=='spain'].plane.value_counts().to_string()}")
X = cached_images(T)

es_train_pat = T[(T.dataset == "spain") & (T.split == "train")].patient.unique()
val_pat = set(rng.choice(es_train_pat, size=max(1, len(es_train_pat) // 10), replace=False))
T["role"] = "other"
sp = T.dataset == "spain"
T.loc[sp & T.known & (T.split == "train") & ~T.patient.isin(val_pat), "role"] = "train"
T.loc[sp & T.known & (T.split == "train") & T.patient.isin(val_pat), "role"] = "val"
T.loc[sp & T.known & (T.split == "test"), "role"] = "test_known"
T.loc[sp & ~T.known, "role"] = "spain_unknown"     # Other + Maternal cervix, never trained on
T.loc[T.dataset == "africa", "role"] = "africa"
say(f"\nRoles:\n{T.role.value_counts().to_string()}")
T.to_csv(OUT / "image_table.csv", index=False)

# ---------------- model ----------------
weights = None if args.no_pretrained else torchvision.models.ResNet18_Weights.IMAGENET1K_V1
def new_model():
    m = torchvision.models.resnet18(weights=weights)
    m.fc = nn.Linear(512, len(KNOWN))
    return m

aug = torch.nn.Sequential(          # light, ultrasound-safe augmentation, applied on the GPU
    torchvision.transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
    torchvision.transforms.RandomRotation(10),
)

def batches(idx, bs, shuffle):
    idx = np.array(idx)
    if shuffle:
        idx = rng.permutation(idx)
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]

def run(model, idx, bs=64):
    """Return logits and 512-d features for the given rows."""
    model.eval()
    feats, logits = [], []
    body = nn.Sequential(*list(model.children())[:-1])
    with torch.no_grad():
        for b in batches(idx, bs, False):
            x = to_tensor(X[b]).to(dev)
            f = body(x).flatten(1)
            feats.append(f.cpu().numpy()); logits.append(model.fc(f).cpu().numpy())
    return np.concatenate(logits), np.concatenate(feats)

ckpt = OUT / "model_spain.pt"
partial = OUT / "model_spain_unfinished.pt"
model = new_model().to(dev)
tr_idx = np.where(T.role == "train")[0]
va_idx = np.where(T.role == "val")[0]
y = torch.tensor(T.label.values)

if ckpt.exists():
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    say("\nLoaded trained model from a previous run.")
else:
    counts = np.bincount(T.label.values[tr_idx], minlength=len(KNOWN))
    w = torch.tensor(counts.sum() / (len(KNOWN) * counts), dtype=torch.float32).to(dev)
    lossf = nn.CrossEntropyLoss(weight=w)            # rarer planes count more
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best, hist = -1, []
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0
        for b in batches(tr_idx, 32, True):
            x = aug(to_tensor(X[np.sort(b)]).to(dev))
            yy = y[np.sort(b)].to(dev)
            opt.zero_grad(); loss = lossf(model(x), yy); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        sched.step()
        lv, _ = run(model, va_idx)
        acc = accuracy_score(T.label.values[va_idx], lv.argmax(1))
        hist.append((ep + 1, tot / len(tr_idx), acc))
        say(f"epoch {ep+1}/{args.epochs}  train loss {tot/len(tr_idx):.3f}  val acc {acc:.3f}  ({time.time()-t0:.0f}s)")
        if acc > best:
            best = acc; torch.save(model.state_dict(), partial)
    model.load_state_dict(torch.load(partial, map_location=dev))
    partial.rename(ckpt)              # only a fully finished training counts as done
    pd.DataFrame(hist, columns=["epoch", "train_loss", "val_acc"]).to_csv(OUT / "training_history.csv", index=False)

# ---------------- outputs for every image ----------------
logits, feats = run(model, np.arange(len(T)))
np.savez_compressed(OUT / "outputs.npz", logits=logits, feats=feats)
pred = logits.argmax(1)

say("\n=== Spanish test set (4 known planes) ===")
m = T.role == "test_known"
say(f"accuracy {accuracy_score(T.label[m], pred[m]):.3f}   n={m.sum()}")

say("\n=== Africa, zero-shot (no African image used in training) ===")
rows = []
for c in sorted(T[T.dataset == "africa"].site.unique()):
    m = ((T.site == c) & (T.split == "test")).values      # African TEST patients only
    acc = accuracy_score(T.label[m], pred[m])
    rows.append((c, int(m.sum()), acc))
    cm = pd.DataFrame(confusion_matrix(T.label[m], pred[m], labels=range(4)), index=KNOWN, columns=KNOWN)
    say(f"\n{c}: accuracy {acc:.3f}  n={m.sum()}  (rows = truth, columns = model)\n{cm.to_string()}")
zs = pd.DataFrame(rows, columns=["country", "n_test", "accuracy"])
zs.to_csv(OUT / "zero_shot_by_country.csv", index=False)

say("\n=== What does the model call the views it was never taught? (Spain Other + Maternal cervix) ===")
m = (T.role == "spain_unknown").values
p = torch.softmax(torch.tensor(logits[m]), 1).numpy()
say(pd.Series(np.array(KNOWN)[pred[m]]).value_counts().to_string())
say(f"share of these unknown images given >90% confidence: {(p.max(1) > 0.9).mean():.3f}")

# ---------------- repeat the country check on cropped images ----------------
say("\n=== Country check again, after cropping to the fan (ImageNet features, as in Step 2) ===")
base = torchvision.models.resnet18(weights=weights); base.fc = nn.Identity(); base.eval().to(dev)
ai = np.where(T.dataset == "africa")[0]
with torch.no_grad():
    F = np.concatenate([base(to_tensor(X[b]).to(dev)).cpu().numpy() for b in np.array_split(ai, max(1, len(ai) // 32))])
yc, g = T.site.values[ai], T.patient.values[ai]
pc = np.empty_like(yc)
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(F, yc, g):
    pc[te] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1)).fit(F[tr], yc[tr]).predict(F[te])
say(f"country accuracy after cropping: {accuracy_score(yc, pc):.3f}  (Step 2, padded only: 0.998)")

# ---------------- figure ----------------
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.bar(zs.country, zs.accuracy, color="#4C72B0")
sp_acc = accuracy_score(T.label[T.role == "test_known"], pred[(T.role == "test_known").values])
ax.axhline(sp_acc, color="k", ls="--", lw=1.2, label=f"Spanish test ({sp_acc:.2f})")
for i, v in enumerate(zs.accuracy):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
ax.set_ylim(0, 1.05); ax.set_ylabel("Plane accuracy"); ax.set_xlabel("African country (test patients)")
ax.set_title("Trained in Spain only, tested in Africa without local data")
ax.legend(loc="lower right", fontsize=9); fig.tight_layout()
fig.savefig(OUT / "zero_shot_by_country.png", dpi=200)

(OUT / "report.txt").write_text("\n".join(log))
print(f"\nSaved results in {OUT}/")
