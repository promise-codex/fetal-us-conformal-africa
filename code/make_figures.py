"""Paper figures from the saved result tables (run from the project folder).
Inputs: results/step4/all_runs.csv, results/step5/local_gate_all_runs.csv,
        results/step5/saliency/gradcam_Ghana.png and gradcam_Uganda.png
The Spanish-gate refusal rates and home AUROCs are copied from results/step4/report.txt."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

R4 = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/step4/all_runs.csv")
R5 = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/step5/local_gate_all_runs.csv")
GH = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("results/step5/saliency/gradcam_Ghana.png")
UG = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("results/step5/saliency/gradcam_Uganda.png")
OUT = Path(sys.argv[5]) if len(sys.argv) > 5 else Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})
COUNTRIES = ["Algeria", "Egypt", "Ghana", "Malawi", "Uganda"]
COL = dict(zip(COUNTRIES, ["#1b6ca8", "#d1495b", "#2e933c", "#edae49", "#7b4b94"]))
KS = [0, 1, 2, 4, 6]

# ---------------- Figure 1 ----------------
R = pd.read_csv(R4)
fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.6))
for c in COUNTRIES:
    for cal, ls, mk in [("spain_cal", ":", "s"), ("local_cal", "-", "o")]:
        g = R[(R.country == c) & (R.calibration == cal)].groupby("k").mean(numeric_only=True).loc[KS]
        ax[0].plot(KS, g.coverage, ls, marker=mk, ms=3, color=COL[c], lw=1.2)
        ax[1].plot(KS, g.answered, ls, marker=mk, ms=3, color=COL[c], lw=1.2)
        ax[2].plot(KS, 100 * g.wrong_answer, ls, marker=mk, ms=3, color=COL[c], lw=1.2)
ax[0].axhline(0.9, color="k", lw=0.8, ls="--")
ax[0].set_title("(a) Coverage (target 0.90)")
ax[1].set_title("(b) Answered without a person")
ax[2].set_title("(c) Wrong single answers (%)")
ax[0].set_ylabel("Share of test images")
for a in ax:
    a.set_xticks(KS); a.set_xlabel("Local patients for adaptation, $k$"); a.grid(alpha=0.25, lw=0.5)
h = [plt.Line2D([], [], color=COL[c], lw=1.5, label=c) for c in COUNTRIES]
h += [plt.Line2D([], [], color="k", ls="-", marker="o", ms=3, label="local threshold"),
      plt.Line2D([], [], color="k", ls=":", marker="s", ms=3, label="Spanish threshold")]
fig.legend(handles=h, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1))
for ext in ["pdf", "png"]:
    fig.savefig(OUT / f"fig1_local_patients.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------- Figure 2 ----------------
A = pd.read_csv(R5)
SCORES = ["max probability", "energy", "Mahalanobis (mean shift)"]
LABEL = {"max probability": "Max. probability", "energy": "Energy", "Mahalanobis (mean shift)": "Mahalanobis"}
# From results/step4/report.txt (gate tuned on Spanish validation images, Spanish class centres)
spain_refused = {"max probability": [0.269, 0.404, 0.231, 0.255, 0.128],
                 "energy": [0.269, 0.365, 0.282, 0.255, 0.103],
                 "Mahalanobis (mean shift)": [0.327, 1.000, 0.769, 0.314, 0.410]}
home_auc = {"max probability": 0.905, "energy": 0.917, "Mahalanobis (mean shift)": 0.916}
e6_auc = {"max probability": 0.849, "energy": 0.868, "Mahalanobis (mean shift)": 0.908}

fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.9), gridspec_kw={"width_ratios": [1.6, 1]})
x = np.arange(len(COUNTRIES)); w = 0.13
for j, s in enumerate(SCORES):
    loc = A[A.score == s].groupby("country").wrongly_refused.mean().loc[COUNTRIES].values
    ax[0].bar(x + (j - 1) * 2 * w - w / 2, spain_refused[s], w, color=f"C{j}", alpha=0.35, edgecolor=f"C{j}")
    ax[0].bar(x + (j - 1) * 2 * w + w / 2, loc, w, color=f"C{j}", label=LABEL[s])
ax[0].axhline(0.05, color="k", lw=0.8, ls="--")
ax[0].set_xticks(x); ax[0].set_xticklabels(COUNTRIES)
ax[0].set_ylabel("Share refused")
ax[0].set_title("(a) Real images refused: Spanish gate (pale), local gate (solid)", fontsize=8.5)
ax[0].legend(frameon=False, loc="upper right"); ax[0].grid(axis="y", alpha=0.25, lw=0.5)
xx = np.arange(3)
ax[1].bar(xx - 0.25, [home_auc[s] for s in SCORES], 0.25, color="0.75", label="Spain, all machines")
ax[1].bar(xx, [e6_auc[s] for s in SCORES], 0.25, color="0.45", label="Spain, one machine (E6)")
loc_auc = [A[A.score == s].AUROC_spain_semantic.mean() for s in SCORES]
lo = [A[A.score == s].groupby("country").AUROC_spain_semantic.mean().min() for s in SCORES]
hi = [A[A.score == s].groupby("country").AUROC_spain_semantic.mean().max() for s in SCORES]
ax[1].bar(xx + 0.25, loc_auc, 0.25, color="C3", label="after local tuning")
ax[1].errorbar(xx + 0.25, loc_auc, yerr=[np.subtract(loc_auc, lo), np.subtract(hi, loc_auc)], fmt="none", ecolor="k", lw=0.8, capsize=2)
ax[1].set_xticks(xx); ax[1].set_xticklabels([LABEL[s] for s in SCORES])
ax[1].set_ylim(0.6, 1.0); ax[1].set_ylabel("AUROC")
ax[1].set_title("(b) Separating untaught views")
ax[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, fontsize=7); ax[1].grid(axis="y", alpha=0.25, lw=0.5)
fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(OUT / f"fig2_unknown_views.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------- Figure 3 ----------------
ims = [Image.open(p).convert("RGB") for p in (GH, UG)]
W = max(i.width for i in ims)
canvas = Image.new("RGB", (W, sum(i.height for i in ims)), "white")
y = 0
for i in ims:
    canvas.paste(i, (0, y)); y += i.height
canvas.save(OUT / "fig3_gradcam.png", dpi=(300, 300))
print("figures saved in", OUT)
