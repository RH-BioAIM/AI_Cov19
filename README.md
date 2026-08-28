# Predicting COVID-19 Severity Using a Multimodal AI Model with Cross-Institutional Evaluation of Imaging Performance

**Journal:** IEEE Open Journal of Engineering in Medicine and Biology (OJEMB)
**Manuscript:** OJEMB-00026-2026
**Authors:** Hunter Lau, Ryan Lang, Hamed Akbari — Santa Clara University

**Citation:** [citation to be added upon publication]

---

## Overview

This repository contains the analysis code for a dual-branch multimodal framework that predicts COVID-19 severity from two inputs: chest radiographs, processed by a Swin Transformer imaging branch, and structured clinical variables (labs, vitals, comorbidities, presenting symptoms), processed by an XGBoost clinical branch. The two branches are combined via feature-level fusion — the imaging branch's predicted length of stay is used as a single engineered feature alongside the clinical variables in the fused (integrated) model.

Length of hospital stay (LOS) is the training target for both the imaging and integrated models, thresholded post hoc into a binary severity label. The resulting severity score is additionally validated against an outcome the model was never trained on (in-hospital mortality) and evaluated on two external imaging datasets independent of the primary training cohort.

## Repository structure

```
github_staging/
├── preprocessing/            data acquisition
├── imaging/                  Swin Transformer imaging branch (training)
├── integrated/                clinical (XGBoost) + integrated (clinical+imaging) models
├── evaluation/                DeLong tests, threshold sweep, mortality validation,
│   ├── severity_eval/         metrics tables, and external evaluations
│   └── binary_external_eval/
├── fusion_experiments/        alternative imaging-fusion representations (ablation)
│   └── replace_variants/
├── explainability/             Grad-CAM analysis of the imaging model
└── plotting/                  manuscript figure generation
```

### `preprocessing/`
| File | Description |
|---|---|
| `importkaggle.py` | Downloads the Kaggle COVID-19 Radiography Database (`tawsifurrahman/covid19-radiography-database`) via `kagglehub` and copies it to a local directory. |
| `pos_neg_database.sbatch` | SLURM launcher for `importkaggle.py`. |

### `imaging/`
| File | Description |
|---|---|
| `train_swin.py` | Trains the Swin Transformer imaging model (`swin_base_patch4_window7_224` backbone + a `Dropout`→`Linear` regression head) to predict length of stay from chest radiographs. Includes NIfTI loading and preprocessing (background removal, CLAHE contrast enhancement), stratified 5-fold cross-validation, and per-fold checkpoint/metric logging. |
| `train_swin.sbatch` | SLURM launcher for `train_swin.py`. |

### `integrated/`
| File | Description |
|---|---|
| `build_integrated_model.py` | Core shared module. Loads the matched clinical+imaging cohort (using the imaging model's own cross-validation fold assignments as the shared ground-truth partition), and builds clinical-only and integrated (clinical+imaging) feature matrices for both the full and leakage-restricted feature sets. Imported by most other scripts in this repository. |
| `build_integrated_model.sbatch` | SLURM launcher. |
| `build_nested_tuned_models.py` | Runs nested-cross-validation hyperparameter tuning for the restricted clinical-only and integrated models, producing the final out-of-fold predictions behind the paper's headline numbers. |
| `build_final_models.py` | Finalization pass: drops the high-missingness `kidney_transplant` feature and applies a consistent tuned XGBoost configuration to both the clinical-only-restricted and integrated-restricted models. |
| `hyperparam_tuning.py` | Shared nested-CV hyperparameter search (outer 5-fold / inner 4-fold grid search) used by `build_nested_tuned_models.py`. |
| `step7_feature_table.py` | Builds the candidate clinical-variable / missingness table (Supplementary Table S1) from the TCIA data dictionary and the modeled cohort. |

### `evaluation/`
| File | Description |
|---|---|
| `delong.py` | From-scratch implementation of the fast DeLong algorithm (Sun & Xu, 2014) for comparing correlated ROC AUCs; cross-checked against scikit-learn's AUC computation. |
| `step4_delong.py` | Trains the tuned clinical-only-restricted baseline and runs the initial paired DeLong significance tests. |
| `delong_nested_tuned.py` | Paired DeLong tests (integrated vs. clinical-only, integrated vs. imaging-only) on the final nested-CV-tuned restricted predictions — the paper's headline significance results. |
| `step6_mortality_nested_tuned.py` | Validates the LOS-trained severity score against in-hospital mortality — an outcome never used as a training feature — on the final nested-tuned predictions. |
| `step6_threshold_sweep_nested_tuned.py` | Regenerates **Supplementary Table S2**: classification metrics (accuracy, sensitivity, specificity, precision, F1, ROC AUC) swept across LOS decision thresholds 3–10 days, for the final nested-tuned integrated model. |
| `metrics_by_threshold_5_67_7.py` | Full classification-metrics table at three decision thresholds (5-day fixed, 6.71-day Youden-optimal, 7-day) for all three final restricted models (imaging-only, clinical-only, integrated). |
| `table2_comparison.py` | Builds the manuscript's Table II: original (full/leaky feature set) vs. new restricted-feature-set model metrics, side by side. |
| `diagnose_specificity.py` | Diagnostic analysis of the integrated model's specificity at the 5-day decision threshold (calibration curve, false-positive characterization, alternative operating points, fold-honest recalibration check). |
| `check_specificity_stability.py` | Tests whether the model's specificity is a property of the restricted feature set or an artifact of the hyperparameter-tuning procedure, by comparing configurations on the identical feature set and cross-validation partition. |
| `severity_eval/sev_eval.py` | External evaluation of the imaging model on the MIDRC-RICORD-1c severity-stratified dataset (mild vs. severe COVID-19); reports F1, ROC AUC, and a threshold-sweep classification-metrics table. |
| `severity_eval/sev_eval.sbatch` | SLURM launcher. |
| `binary_external_eval/pos_neg_analysis.py` | External binary (COVID-positive vs. negative) evaluation of the imaging model on the Kaggle COVID-19 Radiography Database. |
| `binary_external_eval/pos_neg_analysis.sbatch` | SLURM launcher. |

### `fusion_experiments/`
| File | Description |
|---|---|
| `extract_embeddings.py` | Extracts the Swin backbone's 1024-dimensional penultimate-layer embeddings, out-of-fold, for every patient. |
| `extract_embeddings.sbatch` | SLURM launcher. |
| `compare_fusion_variants.py` | Ablation study comparing the committed scalar (predicted-LOS) imaging fusion against variants that *add* PCA-reduced or raw 1024-dim imaging embeddings, or a probability-recast imaging feature, on top of the scalar. |
| `fusion_variant_comparison.csv` | Saved results of the above comparison. |
| `replace_variants/compare_replace_variants.py` | Extends the above: tests *replacing* (rather than augmenting) the scalar imaging feature with embeddings at several PCA dimensionalities (8/16/32/64) and the raw 1024-dim vector. |
| `replace_variants/full_comparison_with_references.csv` | Saved results, including reference rows for the committed model and the "added-embedding" variants. |

### `explainability/`
| File | Description |
|---|---|
| `swin_explain2.py` | Grad-CAM implementation adapted for the Swin Transformer's hierarchical token stages (the refined second of three attempts, after an initial attempt's failure modes). Also provides the model-loading/preprocessing helper functions reused by `fusion_experiments/extract_embeddings.py`. |
| `run_explainability2.py` | Driver script for the Grad-CAM analysis above and its diagnostics (corner-artifact checks, in-lung mass-fraction quantification). |
| `lung_segmentation.py` | Wraps a pretrained `torchxrayvision` lung-segmentation model, used to quantify how much Grad-CAM attention actually falls inside the lungs versus a chance baseline. |
| `preprocess_soft.py` | Two preprocessing variants — the model's actual "hard" background-masking preprocessing, and a "soft" feathered-edge variant — used to test whether a preprocessing artifact was biasing the Grad-CAM attention maps. |
| `run_explainability3.py` | Final Grad-CAM attempt: re-runs the analysis under the corrected ("soft") preprocessing and reports the lung-localization quantification and conclusion. |

### `plotting/`
| File | Description |
|---|---|
| `figure4_roc_confusion.py` | Generates Figure 4 (ROC curve + confusion matrix) from the final nested-tuned integrated model. |
| `figure6_feature_association.py` | Generates Figure 6 (top-12 clinical-feature bar chart, ranked by concordance-index association with length of stay). |

---

## Data availability

**No patient data, chest radiographs, NIfTI/DICOM files, or trained model checkpoints are included in this repository.** All data-loading paths in the scripts are placeholders (see *Configuration / paths* below) that must be pointed at your own local copies of the following datasets, obtained independently from their original sources:

1. **Stony Brook University COVID-19 dataset (COVID-19-NY-SBU)** — the primary cohort's chest radiographs and structured clinical variables (labs, vitals, comorbidities, outcomes). Available via **The Cancer Imaging Archive (TCIA)**.
2. **MIDRC-RICORD-1c** — a severity-stratified (mild/severe) chest radiograph dataset, used for the external severity evaluation (`evaluation/severity_eval/sev_eval.py`). Available via the **Medical Imaging and Data Resource Center (MIDRC)** / TCIA.
3. **COVID-19 Radiography Database** — used for the external binary (positive/negative) evaluation (`evaluation/binary_external_eval/pos_neg_analysis.py`). Publicly available on **Kaggle** (`tawsifurrahman/covid19-radiography-database`); `preprocessing/importkaggle.py` downloads it automatically via `kagglehub` once you have Kaggle API credentials configured.

Users must obtain each dataset independently under its own data-use terms and set their own local paths before running any script.

## Setup

- Python 3.11
- Install dependencies: `pip install -r requirements.txt`

This code was developed and run on an HPC cluster using the SLURM scheduler. The `.sbatch` files included alongside their corresponding Python scripts are the actual launcher scripts used, provided as examples — they contain placeholder cluster paths, a placeholder output-log path, and a placeholder `--mail-user` email that you will need to edit for your own environment (or ignore entirely and invoke the Python scripts directly).

## Configuration / paths

The data, checkpoint, and intermediate-output paths hardcoded in these scripts (e.g. `ALLDATA_PATH`, `IMAGING_PRED_PATH`, `PATIENT_DICT_PATH`, checkpoint directories, and the `.sbatch` files' `--output`/`--mail-user` lines) are placeholders left over from the original development environment. **Before running anything, edit these to point at your own local copies of the datasets described above and your own working/output directories.**

## Reproducing key results

The pipeline runs in the following order; each stage's outputs feed the next:

1. **Preprocess / acquire data** (`preprocessing/`) — obtain the three datasets listed under *Data availability* and set local paths.
2. **Train the imaging model** (`imaging/train_swin.py`) — trains the Swin Transformer on chest radiographs to predict length of stay, producing per-fold checkpoints and out-of-fold predictions.
3. **Build the clinical and integrated models** (`integrated/`) — `build_integrated_model.py` builds the shared cohort and baseline models; `hyperparam_tuning.py` + `build_nested_tuned_models.py` produce the final nested-CV-tuned models; `build_final_models.py` applies the finalized feature set and configuration.
4. **Evaluation** (`evaluation/`) — DeLong significance tests (`step4_delong.py`, `delong_nested_tuned.py`), the length-of-stay threshold sweep behind **Supplementary Table S2** (`step6_threshold_sweep_nested_tuned.py`), mortality-endpoint validation (`step6_mortality_nested_tuned.py`), and the classification-metrics tables (`metrics_by_threshold_5_67_7.py`, `table2_comparison.py`, `diagnose_specificity.py`, `check_specificity_stability.py`).
5. **External evaluations** — severity-stratified evaluation on MIDRC-RICORD-1c (`evaluation/severity_eval/`) and binary positive/negative evaluation on the Kaggle radiography dataset (`evaluation/binary_external_eval/`).
6. **Fusion experiments** (`fusion_experiments/`) — ablation study of alternative imaging-fusion representations (embeddings vs. the scalar predicted-LOS feature).
7. **Figures** (`plotting/`) — regenerates the manuscript's Figure 4 and Figure 6 from the final model outputs.

## License

[license to be determined]

## Contact

For questions about this repository or the underlying study, see the corresponding author listed in the paper.
