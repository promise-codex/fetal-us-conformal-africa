"""
Step 2: a first look at how different the five African countries are.
Needs only the African set.

Idea (no training of a deep network yet):
  1. Pass every African image through a ResNet-18 that was pre-trained on
     ImageNet (everyday photos). We do not change its weights. We just take
     the 512 numbers it produces for each image (its "features").
  2. Fit a simple logistic regression on those features to answer two questions:
     (a) Can the model tell WHICH COUNTRY an image came from?
         If yes, every country really is its own "domain", so a model trained
         in one place cannot be assumed to work in another.
     (b) Train on four countries, test on the fifth (leave-one-country-out).
         How well does plane recognition carry over to a country it never saw?

Every split is done by PATIENT, so images of the same patient never sit in
both train and test. Mixing them would inflate the scores (data leakage).

Images are first padded to a square with black and then resized to 224 x 224,
so that the original image size (which differs by country) cannot leak.

Run:   python step2_african_probe.py
Output: results/step2/
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import torchvision
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--no-pretrained", action="store_true", help="only for testing the code without internet")
args = ap.parse_args()

DATA = Path("data")
OUT = Path("results/step2")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------- load the table ----------
csv = sorted(DATA.rglob("African_planes_database.csv"))[0]
af = pd.read_csv(csv)
af["path"] = [csv.parent / c / f"{f}.png" for c, f in zip(af.Center, af.Filename)]
af["patient_id"] = af.Center + "_" + af.Patient_num.astype(str)   # patient numbers repeat across countries
af["plane"] = af.Plane.str.replace("Fetal ", "", regex=False)

# ---------- same preprocessing for every image ----------
def pad_to_square(im):
    w, h = im.size
    s = max(w, h)
    return ImageOps.pad(im, (s, s), color=0)   # black borders, keeps aspect ratio

prep = transforms.Compose([
    transforms.Lambda(lambda im: pad_to_square(im.convert("L"))),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),   # ResNet expects 3 channels
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---------- feature extractor ----------
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
weights = None if args.no_pretrained else torchvision.models.ResNet18_Weights.IMAGENET1K_V1
net = torchvision.models.resnet18(weights=weights)
net.fc = torch.nn.Identity()          # drop the ImageNet classifier, keep the 512 features
net.eval().to(device)
print(f"Device: {device}   pretrained: {weights is not None}")

feats = []
with torch.no_grad():
    for i in range(0, len(af), 32):
        batch = torch.stack([prep(Image.open(p)) for p in af.path.iloc[i:i + 32]]).to(device)
        feats.append(net(batch).cpu().numpy())
X = np.concatenate(feats)
np.savez(OUT / "african_features.npz", X=X, filename=af.Filename.values)
print(f"Features: {X.shape}")

def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1))

lines = []
def say(s=""):
    print(s)
    lines.append(s)

# ---------- (a) can we tell the country? ----------
y_c = af.Center.values
groups = af.patient_id.values
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
pred_c = np.empty_like(y_c)
for tr, te in cv.split(X, y_c, groups):
    m = clf().fit(X[tr], y_c[tr])
    pred_c[te] = m.predict(X[te])
say("=== (a) Predicting the COUNTRY from the image (5-fold, split by patient) ===")
say(f"Accuracy: {accuracy_score(y_c, pred_c):.3f}   (chance = about {1/5:.2f})")
labs_c = sorted(set(y_c))
say(pd.DataFrame(confusion_matrix(y_c, pred_c, labels=labs_c), index=labs_c, columns=labs_c).to_string())

# ---------- (b) leave one country out, predict the PLANE ----------
say("\n=== (b) Predicting the PLANE, train on 4 countries, test on the 5th ===")
planes = sorted(af.plane.unique())
rows = []
for c in labs_c:
    tr = (af.Center != c).values
    te = ~tr
    m = clf().fit(X[tr], af.plane.values[tr])
    p = m.predict(X[te])
    t = af.plane.values[te]
    acc = accuracy_score(t, p)
    bacc = balanced_accuracy_score(t, p)
    rows.append((c, int(te.sum()), acc, bacc))
    cm = pd.DataFrame(confusion_matrix(t, p, labels=planes), index=planes, columns=planes)
    say(f"\nHeld-out country: {c}   n={te.sum()}   accuracy={acc:.3f}   balanced accuracy={bacc:.3f}")
    say(cm.to_string())
res = pd.DataFrame(rows, columns=["held_out", "n", "accuracy", "balanced_accuracy"])
res.to_csv(OUT / "leave_one_country_out.csv", index=False)

# within-country reference: 5-fold by patient, all countries pooled
pred_p = np.empty(len(af), dtype=object)
for tr, te in cv.split(X, af.plane.values, groups):
    pred_p[te] = clf().fit(X[tr], af.plane.values[tr]).predict(X[te])
pooled = accuracy_score(af.plane.values, pred_p)
say(f"\nReference, all countries mixed (5-fold by patient): plane accuracy = {pooled:.3f}")

# ---------- figure ----------
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.bar(res.held_out, res.accuracy, color="#4C72B0", label="held-out country")
ax.axhline(pooled, color="k", ls="--", lw=1.2, label=f"countries mixed ({pooled:.2f})")
ax.set_ylim(0, 1)
ax.set_ylabel("Plane accuracy")
ax.set_xlabel("Country left out of training")
ax.set_title("Frozen ImageNet features, logistic regression")
for i, v in enumerate(res.accuracy):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "leave_one_country_out.png", dpi=200)

(OUT / "report.txt").write_text("\n".join(lines))
print(f"\nSaved results in {OUT}/")
