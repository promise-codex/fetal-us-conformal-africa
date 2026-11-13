# Local conformal abstention for fetal ultrasound plane classification in five African countries

This repository holds the code and result tables for a study on whether a fetal ultrasound plane classifier trained in Spain can be made to answer only when it is likely to be right, and to refuse views it was never taught, in Algeria, Egypt, Ghana, Malawi and Uganda, using only a few local patients.

## The question

A plane classifier trained on four standard fetal planes (brain, abdomen, femur, thorax) always returns one of those four answers, even for views it was never taught and even when it is unsure. Split conformal prediction can turn it into a model that answers only when its prediction set holds a single plane, with a target coverage of 90%. That guarantee needs calibration data from the same source as the test data. This study measures, per African country:

1. whether a threshold set on Spanish data keeps its coverage in Africa (it does not);
2. how many local patients are needed to restore it, separately for calibration and for adapting the last layer;
3. whether an unknown-view gate (maximum softmax probability, energy, Mahalanobis distance) can be moved to a new site with the same few patients;
4. whether the findings hold across two networks (ResNet-18, DenseNet-169), several training runs, full fine-tuning and simpler baselines.

## Data

Both datasets are public under CC BY 4.0 and are not included here.

| Dataset | Source | Content |
|---|---|---|
| FETAL_PLANES_DB | Zenodo record 3904280 (Burgos-Artizzu et al., Sci. Rep. 2020) | 12,400 images, 1,792 patients, Barcelona |
| African fetal planes | Zenodo record 7540448 (Sendra-Balcells et al., Sci. Rep. 2023) | 450 images, 127 patients, five countries |

Download and unzip both into `code/data/`. The loader searches `code/data/` for `FETAL_PLANES_DB_data.csv` and `African_planes_database.csv`, so the exact sub-folder names do not matter.

## Setup

```
cd code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.13 and PyTorch on Apple silicon (MPS). CUDA and CPU also work; CPU training is slow.

## Steps

Run each step from inside `code/`. Every step writes to `code/results/stepN/`.

| Step | Script | What it does | Time (Apple M-series) |
|---|---|---|---|
| 1 | `step1_check_data.py` | Counts images per country and plane, image sizes and modes, example grid | 1 min |
| 2 | `step2_african_probe.py` | Frozen ImageNet ResNet-18 features: can the country be predicted from the image, and leave-one-country-out plane accuracy | 5 min |
| 3 | `step3_train_spain.py` | Fine-tunes ResNet-18 on the four Spanish planes (Spanish "other" and "maternal cervix" held out as untaught views), zero-shot test in Africa, saves logits and features | 10 to 25 min |
| 4 | `step4_abstain.py` | Last-layer L2-SP adaptation with k = 0, 1, 2, 4, 6 local patients, split conformal (LAC, alpha = 0.1) with Spanish vs local threshold, 20 random draws; unknown-view AUROC including a same-machine (Voluson E6) check | 5 min |
| 5 | `step5_local_gate_and_saliency.py` | Unknown-view gate tuned on local patients (mean-shifted Mahalanobis, energy, max probability), full system; Grad-CAM for Ghana and Uganda | 5 min |
| 6 | `step6_cross_check.py` | Retrains ResNet-18 and DenseNet-169 with seeds 0, 1, 2 and repeats the main analyses; full fine-tuning baseline | 2.5 to 3 h |
| 7 | `step7_extra_checks.py` | Calibration-size sweep, baselines (fixed 0.9 cut-off, Spain-tuned cut-off, APS), patient-level bootstrap intervals | 5 min |
| | `make_figures.py` | Paper figures from the Step 4 and Step 5 tables | under 1 min |

`common.py` holds the shared preprocessing (greyscale, crop to the ultrasound fan, pad to square, resize to 224) and the image cache.

## Main results

Results from the runs in `results/` (first ResNet-18 run unless stated).

- Spanish test accuracy 99.1%. Zero-shot accuracy in Africa 63.5% (Egypt) to 100% (Uganda); six repeat runs gave 84.6% to 92.3% in Egypt.
- With a threshold set on Spanish data, coverage in Africa was 0.29 to 0.85 across all runs, against a 0.90 target.
- With a threshold set on 6 or 7 local patients, mean coverage was 0.88 to 1.00 at k = 0 in every country and run. After adaptation with 6 patients, Uganda fell to 0.82 to 0.88 in five of six repeat runs, entirely through refusals (no wrong answers).
- Two calibration patients (about 7 images) are too few to set a threshold at alpha = 0.1; three is the minimum.
- A fixed 0.9 confidence cut-off gave wrong answers to 21.2% of Egyptian images without adaptation; the local conformal threshold had a worst-country rate of 2.9%.
- The mean-shifted Mahalanobis gate kept an AUROC of 0.88 to 0.94 for taught against untaught views after moving to each site.
- Per-country test sets have only 13 or 14 patients, so per-country intervals are wide (see `results/step7/bootstrap_ci.csv`).

## Results folder

| Folder | Files |
|---|---|
| `results/step4/` | `all_runs.csv`: every country, draw, k and threshold source |
| `results/step5/` | `local_gate_all_runs.csv`, Grad-CAM images for Ghana and Uganda |
| `results/step6/` | `analysis_all_models.csv`, `summary_by_model.csv`, `full_finetune.csv` |
| `results/step7/` | calibration sweep, baselines, bootstrap intervals, `report.txt` |
| `figures/` | paper figures |

Trained weights and cached features are not included because of their size; Steps 3 and 6 recreate them.

## Limitations

- No African dataset contains untaught views, so the unknown-view tests use Spanish "other" and "maternal cervix" images as a proxy.
- Each country has 39 to 52 test images and 18 to 28 calibration images; the conformal guarantee holds on average over calibration draws and varies between draws.
- Grad-CAM works on a 7 x 7 grid and cannot resolve thin measurement marks, so it is not proof that the marks in Ghana and Uganda are ignored.

## Licence

Code: MIT (see `LICENSE`). Result tables and figures: CC BY 4.0. They are derived from the two CC BY 4.0 datasets above, which should be cited when they are used.

## Citation

See `CITATION.cff`.
