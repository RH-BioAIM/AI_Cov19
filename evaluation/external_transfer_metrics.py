"""
Computes ROC AUC with bootstrap confidence intervals and threshold metrics
for each external dataset, using the fold model and cutoff chosen by
external_transfer_selection.py. Patient-clustered resampling is used for
MIDRC-RICORD-1c and image-level resampling for the COVID-19 Radiography
Database.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REVISION_DIR = os.path.dirname(os.path.abspath(__file__))
SELECTION_PATH = os.path.join(REVISION_DIR, "external_transfer_selection.csv")

N_BOOT = 2000
SEED = 42

# Per-dataset prediction file, bootstrap unit, and the five-fold ensemble AUC
# quoted alongside the selected fold for reference.
DATASETS = {
    "MIDRC-RICORD-1c": {
        "pred_path": os.path.join(REVISION_DIR, "midrc_predictions_all_folds.csv"),
        "resample": "patient", "ensemble_auc": 0.807,
    },
    "COVID-19 Radiography Database": {
        "pred_path": os.path.join(REVISION_DIR, "covid19_radiography_predictions_all_folds.csv"),
        "resample": "image", "ensemble_auc": 0.739,
    },
}


def threshold_metrics(y_true, y_score, cutoff, inclusive):
    """Confusion-matrix metrics at a cutoff. The selection procedure treats
    the cutoff as inclusive for MIDRC-RICORD-1c and exclusive for the
    COVID-19 Radiography Database; both are reproduced here."""
    y_pred = ((y_score >= cutoff) if inclusive else (y_score > cutoff)).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    return {
        "accuracy": (tp + tn) / len(y_true), "precision": precision, "recall": recall,
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"), "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def bootstrap_ci_image(y_true, y_score):
    """Percentile CI from resampling images with replacement."""
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    stats = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(roc_auc_score(y_true[idx], y_score[idx]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)), len(stats)


def bootstrap_ci_patient(y_true, y_score, patients):
    """Percentile CI from resampling patients with replacement, carrying all
    of a sampled patient's images."""
    rng = np.random.default_rng(SEED)
    unique_patients = np.unique(patients)
    rows_by_patient = {p: np.where(patients == p)[0] for p in unique_patients}
    stats = []
    for _ in range(N_BOOT):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        idx = np.concatenate([rows_by_patient[p] for p in sampled])
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(roc_auc_score(y_true[idx], y_score[idx]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)), len(stats)


def main():
    selection = pd.read_csv(SELECTION_PATH).set_index("dataset")

    rows = []
    for dataset, cfg in DATASETS.items():
        chosen = selection.loc[dataset]
        fold = int(chosen["selected_fold"])
        cutoff = float(chosen["selected_cutoff_days"])
        inclusive = bool(chosen["cutoff_inclusive"])

        frame = pd.read_csv(cfg["pred_path"])
        y_true = frame["label"].values.astype(int)
        y_score = frame[f"pred_fold{fold}"].values.astype(float)
        auc = float(roc_auc_score(y_true, y_score))

        if cfg["resample"] == "patient":
            ci_low, ci_high, n_used = bootstrap_ci_patient(y_true, y_score, frame["patient_id"].values)
        else:
            ci_low, ci_high, n_used = bootstrap_ci_image(y_true, y_score)

        metrics = threshold_metrics(y_true, y_score, cutoff, inclusive)
        rows.append({
            "dataset": dataset, "fold": fold, "cutoff_days": cutoff,
            "roc_auc": auc, "ci95_low": ci_low, "ci95_high": ci_high,
            "ensemble_roc_auc": cfg["ensemble_auc"],
            "auc_minus_ensemble": auc - cfg["ensemble_auc"],
            "n_images": int(len(frame)), "n_positive": int((y_true == 1).sum()),
            "n_negative": int((y_true == 0).sum()), "n_bootstrap_used": n_used,
            **metrics,
        })

        print(f"\n{dataset} -- fold {fold}, cutoff {cutoff:.1f} days")
        print(f"  ROC AUC = {auc:.4f} (95% CI [{ci_low:.4f}, {ci_high:.4f}], "
              f"{cfg['resample']}-level resampling)")
        print(f"  five-fold ensemble AUC = {cfg['ensemble_auc']:.3f} "
              f"(difference {auc - cfg['ensemble_auc']:+.4f})")
        print(f"  accuracy={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
              f"recall={metrics['recall']:.4f} specificity={metrics['specificity']:.4f} "
              f"f1={metrics['f1']:.4f}")
        print(f"  TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")

    out = pd.DataFrame(rows)
    path = os.path.join(REVISION_DIR, "external_transfer_metrics.csv")
    out.to_csv(path, index=False)
    print(f"\nSaved {path}")
    print(f"Bootstrap: {N_BOOT} resamples, seed {SEED}, percentile interval.")


if __name__ == "__main__":
    main()
