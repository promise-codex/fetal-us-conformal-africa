"""Shared helpers used by every step from Step 3 onward."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch

DATA = Path("data")
SIZE = 224
KNOWN = ["abdomen", "brain", "femur", "thorax"]          # the four planes the model is taught
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def crop_to_fan(im):
    """Cut away the black frame around the ultrasound fan.

    A row (or column) is kept if more than 5% of its pixels are brighter than 15.
    Small text and logos in the margin do not pass this rule, the fan does.
    After cropping, the image is padded to a square and resized, so that the
    original image size of each country no longer shows.
    """
    g = np.asarray(im.convert("L"))
    m = g > 15
    rows = np.where(m.mean(1) > 0.05)[0]
    cols = np.where(m.mean(0) > 0.05)[0]
    if len(rows) > 10 and len(cols) > 10:
        g = g[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    im = Image.fromarray(g)
    s = max(im.size)
    im = ImageOps.pad(im, (s, s), color=0)
    return im.resize((SIZE, SIZE), Image.BILINEAR)


def load_tables():
    """Return one table with every image from both datasets."""
    af_csv = sorted(DATA.rglob("African_planes_database.csv"))[0]
    af = pd.read_csv(af_csv)
    af = pd.DataFrame({
        "path": [str(af_csv.parent / c / f"{f}.png") for c, f in zip(af.Center, af.Filename)],
        "dataset": "africa",
        "site": af.Center,
        "patient": af.Center + "_" + af.Patient_num.astype(str),
        "plane": af.Plane.str.replace("Fetal ", "", regex=False).str.lower(),
        "split": np.where(af.Train == 1, "train", "test"),
    })

    es_csv = sorted(DATA.rglob("FETAL_PLANES_DB_data.csv"))
    if not es_csv:
        raise SystemExit("Spanish set not found in data/. Finish the download and unzip first.")
    es_csv = es_csv[0]
    es = pd.read_csv(es_csv, sep=None, engine="python")
    es.columns = [c.strip() for c in es.columns]
    plane = es.Plane.str.strip().str.replace("Fetal ", "", regex=False).str.lower()
    es = pd.DataFrame({
        "path": [str(es_csv.parent / "Images" / f"{n.strip()}.png") for n in es.Image_name],
        "dataset": "spain",
        "site": "Spain_" + es.US_Machine.astype(str).str.strip(),
        "patient": "ES_" + es.Patient_num.astype(str),
        "plane": plane,
        "split": np.where(es.Train.astype(str).str.strip() == "1", "train", "test"),
    })
    t = pd.concat([es, af], ignore_index=True)
    t["known"] = t.plane.isin(KNOWN)
    t["label"] = t.plane.map({p: i for i, p in enumerate(KNOWN)}).fillna(-1).astype(int)
    return t


def cached_images(table, cache=Path("results/cache/images_224.npy")):
    """Preprocess every image once and keep them as one uint8 array (about 700 MB)."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    key = cache.with_suffix(".paths.txt")
    if cache.exists() and key.exists() and key.read_text().splitlines() == list(table.path):
        return np.load(cache, mmap_mode="r")
    arr = np.zeros((len(table), SIZE, SIZE), dtype=np.uint8)
    for i, p in enumerate(table.path):
        arr[i] = np.asarray(crop_to_fan(Image.open(p)))
        if i % 1000 == 0:
            print(f"  preprocessing {i}/{len(table)}")
    np.save(cache, arr)
    key.write_text("\n".join(table.path))
    return np.load(cache, mmap_mode="r")


def to_tensor(batch_uint8):
    """uint8 (N,H,W) -> normalised float tensor (N,3,H,W)."""
    x = torch.from_numpy(np.asarray(batch_uint8, dtype=np.float32) / 255.0)
    x = x.unsqueeze(1).repeat(1, 3, 1, 1)
    m = torch.tensor(MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    s = torch.tensor(STD, dtype=torch.float32).view(1, 3, 1, 1)
    return (x - m) / s
