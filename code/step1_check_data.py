"""
Step 1: check both datasets before any training.

What it does
  1. Finds the African set and the Spanish set inside the data folder.
  2. Counts images per country (or per machine for Spain) and per plane.
  3. Records image size and colour mode per country. If sizes differ by
     country, a model could learn "which country" from size alone, so we
     must resize everything to one size later.
  4. Saves a grid of example images for visual inspection.

Run from the project folder:
    python step1_check_data.py
Outputs go to results/step1/
"""
from pathlib import Path
import collections
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("data")
OUT = Path("results/step1")
OUT.mkdir(parents=True, exist_ok=True)


def find_one(pattern):
    hits = sorted(DATA.rglob(pattern))
    return hits[0] if hits else None


report = []
def say(line=""):
    print(line)
    report.append(line)


# ---------------- African set ----------------
af_csv = find_one("African_planes_database.csv")
if af_csv is None:
    raise SystemExit("African CSV not found. Put the extracted Zenodo_dataset folder inside data/")
af_root = af_csv.parent
af = pd.read_csv(af_csv)
af["path"] = [af_root / c / f"{f}.png" for c, f in zip(af.Center, af.Filename)]
missing = [p for p in af.path if not p.exists()]

say("=== AFRICAN SET ===")
say(f"CSV: {af_csv}")
say(f"Rows: {len(af)}   missing image files: {len(missing)}")
say(f"Patients per country:\n{af.groupby('Center').Patient_num.nunique().to_string()}")
say(f"\nImages per country and plane:\n{pd.crosstab(af.Center, af.Plane).to_string()}")
say(f"\nTrain flag (1 = train, 0 = test) per country:\n{pd.crosstab(af.Center, af.Train).to_string()}")

sizes = collections.defaultdict(collections.Counter)
for c, p in zip(af.Center, af.path):
    if p.exists():
        im = Image.open(p)
        sizes[c][(im.size, im.mode)] += 1
say("\nImage size (width, height) and colour mode per country:")
for c, cnt in sizes.items():
    say(f"  {c}: {dict(cnt)}")

# example grid: one row per country, one column per plane
planes = sorted(af.Plane.unique())
countries = sorted(af.Center.unique())
fig, ax = plt.subplots(len(countries), len(planes), figsize=(3 * len(planes), 2.6 * len(countries)))
for i, c in enumerate(countries):
    for j, pl in enumerate(planes):
        a = ax[i, j]
        a.axis("off")
        sub = af[(af.Center == c) & (af.Plane == pl)]
        if len(sub):
            a.imshow(Image.open(sub.path.iloc[0]).convert("L"), cmap="gray")
        else:
            a.text(0.5, 0.5, "none", ha="center", va="center", fontsize=12)
        if i == 0:
            a.set_title(pl.replace("Fetal ", ""), fontsize=12)
        if j == 0:
            a.text(-0.08, 0.5, c, transform=a.transAxes, rotation=90, ha="right", va="center", fontsize=12)
fig.suptitle("African set: one example per country and plane", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "african_examples.png", dpi=150)
plt.close(fig)

# ---------------- Spanish set ----------------
es_csv = find_one("FETAL_PLANES_DB_data.csv")
say("\n=== SPANISH SET ===")
if es_csv is None:
    say("Not found yet. Download FETAL_PLANES_ZENODO.zip (Zenodo record 3904280), unzip it into data/, and run this again.")
else:
    es = pd.read_csv(es_csv, sep=None, engine="python")
    es.columns = [c.strip() for c in es.columns]
    img_dir = es_csv.parent / "Images"
    es["path"] = [img_dir / f"{n}.png" for n in es.Image_name]
    miss = sum(not p.exists() for p in es.path)
    say(f"CSV: {es_csv}")
    say(f"Columns: {list(es.columns)}")
    say(f"Rows: {len(es)}   patients: {es.Patient_num.nunique()}   missing image files: {miss}")
    say(f"\nImages per plane:\n{es.Plane.value_counts().to_string()}")
    if "US_Machine" in es.columns:
        say(f"\nImages per machine and plane:\n{pd.crosstab(es.US_Machine, es.Plane).to_string()}")
    say(f"\nTrain flag per plane:\n{pd.crosstab(es.Plane, es.Train).to_string()}")
    s = collections.Counter()
    for p in es.path.sample(min(300, len(es)), random_state=0):
        if p.exists():
            im = Image.open(p)
            s[(im.size, im.mode)] += 1
    say(f"\nSize and mode (random 300): {s.most_common(6)}")

(OUT / "data_report.txt").write_text("\n".join(report))
print(f"\nSaved {OUT/'data_report.txt'} and {OUT/'african_examples.png'}")
