# Predicting COVID-19 Severity Using a Multimodal AI Model with Cross-Institutional Evaluation of Imaging Performance

**Journal:** IEEE Open Journal of Engineering in Medicine and Biology (OJEMB)
**Manuscript:** OJEMB-00026-2026
**Authors:** Hunter Lau, Ryan Lang, Hamed Akbari (Santa Clara University)

---

## Overview

This repository contains the analysis code for a dual-branch multimodal framework that predicts COVID-19 severity from two inputs. Chest radiographs are processed by a Swin Transformer imaging branch, and structured clinical variables (labs, vitals, comorbidities, presenting symptoms) are processed by an XGBoost clinical branch. The two branches are combined via feature-level fusion: the imaging branch's predicted length of stay is used as a single engineered feature alongside the clinical variables in the fused (integrated) model.

Length of hospital stay (LOS) is the training target for both the imaging and integrated models, thresholded post hoc into a binary severity label.

## Repository structure

```
AI_Cov19/
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
| `importkaggle.py` | Downloads the Kaggle COVID-19 Radiography Database and copies it to a local directory. |
| `pos_neg_database.sbatch` | SLURM launcher for `importkaggle.py`. |

### `imaging/`
| File | Description |
|---|---|
| `train_swin.py` | Trains a Swin Transformer to predict length of stay from chest radiographs, with stratified 5-fold cross-validation. |
| `train_swin.sbatch` | SLURM launcher for `train_swin.py`. |

### `integrated/`
| File | Description |
|---|---|
| `build_integrated_model.py` | Core shared module. Builds the clinical-only and integrated feature matrices for the full and restricted feature sets. Used by most other scripts here. |
| `build_integrated_model.sbatch` | SLURM launcher. |
| `build_tuned_models.py` | Runs nested cross-validation hyperparameter tuning for the restricted clinical-only and integrated models. |
| `build_final_models.py` | Trains the restricted clinical-only and integrated models with a tuned XGBoost configuration. |
| `hyperparam_tuning.py` | Nested cross-validation hyperparameter search used by `build_tuned_models.py`. |
| `feature_table.py` | Builds the candidate clinical-variable and missingness table (Supplementary Table S1). |

### `evaluation/`
| File | Description |
|---|---|
| `delong.py` | Implements the fast DeLong algorithm for comparing correlated ROC AUCs. |
| `delong_analysis.py` | Trains the tuned clinical-only-restricted baseline and runs paired DeLong significance tests. |
| `delong_comparison.py` | Runs paired DeLong tests on the nested-CV-tuned restricted models, the paper's headline significance results. |
| `mortality_validation.py` | Validates the LOS-trained severity score against in-hospital mortality. |
| `threshold_sweep.py` | Regenerates **Supplementary Table S2**: classification metrics across LOS decision thresholds 3 to 10 days. |
| `classification_metrics.py` | Classification-metrics table at three decision thresholds for all three restricted models. |
| `table2_comparison.py` | Builds the manuscript's Table II: full feature set vs. restricted feature set model metrics. |
| `specificity_analysis.py` | Diagnostic analysis of the integrated model's specificity at the 5-day decision threshold. |
| `specificity_check.py` | Tests whether the model's specificity depends on the feature set or the hyperparameter-tuning procedure. |
| `severity_eval/sev_eval.py` | External evaluation of the imaging model on the MIDRC-RICORD-1c severity-stratified dataset. |
| `severity_eval/sev_eval.sbatch` | SLURM launcher. |
| `binary_external_eval/pos_neg_analysis.py` | External binary (COVID-positive vs. negative) evaluation of the imaging model on the Kaggle COVID-19 Radiography Database. |
| `binary_external_eval/pos_neg_analysis.sbatch` | SLURM launcher. |

### `fusion_experiments/`
| File | Description |
|---|---|
| `extract_embeddings.py` | Extracts the Swin backbone's image embeddings for every patient, out-of-fold. |
| `extract_embeddings.sbatch` | SLURM launcher. |
| `compare_fusion_variants.py` | Ablation study comparing the scalar imaging fusion against variants that add PCA-reduced or raw imaging embeddings. |
| `fusion_variant_comparison.csv` | Saved results of the above comparison. |
| `replace_variants/compare_replace_variants.py` | Tests replacing (rather than adding to) the scalar imaging feature with embeddings at several PCA dimensionalities. |
| `replace_variants/full_comparison_with_references.csv` | Saved results, with reference rows from the added-embedding comparison. |


## Data availability

**No patient data, chest radiographs, NIfTI/DICOM files, or trained model checkpoints are included in this repository.** All data-loading paths in the scripts are placeholders (see *Configuration / paths* below) that must be pointed at your own local copies of the following datasets, obtained independently from their original sources:

1. **Stony Brook University COVID-19 dataset (COVID-19-NY-SBU).** The primary cohort's chest radiographs and structured clinical variables (labs, vitals, comorbidities, outcomes). Available via **The Cancer Imaging Archive (TCIA)**.
2. **MIDRC-RICORD-1c.** A severity-stratified (mild/severe) chest radiograph dataset, used for the external severity evaluation (`evaluation/severity_eval/sev_eval.py`). Available via the **Medical Imaging and Data Resource Center (MIDRC)** and TCIA.
3. **COVID-19 Radiography Database.** Used for the external binary (positive/negative) evaluation (`evaluation/binary_external_eval/pos_neg_analysis.py`). Publicly available on **Kaggle** (`tawsifurrahman/covid19-radiography-database`). `preprocessing/importkaggle.py` downloads it automatically via `kagglehub` once you have Kaggle API credentials configured.

Users must obtain each dataset independently under its own data-use terms and set their own local paths before running any script.

## Setup

- Python 3.11
- Install dependencies: `pip install -r requirements.txt`

This code was developed and run on an HPC cluster using the SLURM scheduler. The `.sbatch` files included alongside their corresponding Python scripts are the actual launcher scripts used, provided as examples. 

## Configuration / paths

The data, checkpoint, and intermediate-output paths hardcoded in these scripts (for example `ALLDATA_PATH`, `IMAGING_PRED_PATH`, `PATIENT_DICT_PATH`, checkpoint directories, and the `.sbatch` files' `--output`/`--mail-user` lines) are placeholders left over from the original development environment.



## Contact

For questions about this repository or the underlying study, see the corresponding author listed in the paper.
