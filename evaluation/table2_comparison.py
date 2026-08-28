"""
Builds the manuscript's Table II: classification metrics for the imaging,
clinical-only, and integrated models on the full feature set, alongside the
same models on the restricted (triage-time-only) feature set. The imaging
model uses the same feature set in both columns, since it does not take
clinical features as input.

Input: out-of-fold prediction files for the restricted-feature-set metrics.
The full-feature-set ("original") metrics are hardcoded published values
from the original submission, not read from any file.
Output: table2_comparison.csv.
"""
import os
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
)
from lifelines.utils import concordance_index

REVISION_DIR = os.path.dirname(os.path.abspath(__file__))
LOS_THRESHOLD = 5

# The manuscript's published Table II (5-day threshold) and integrated ROC
# AUC. Imaging-only and clinical-only ROC AUC are not published and are
# left as None.
ORIGINAL_TABLE = {
    "integrated":    {"accuracy": 0.876, "precision": 0.836, "recall": 0.958, "f1": 0.893,
                       "specificity": 0.781, "c_index": 0.844, "roc_auc": 0.930},
    "imaging_only":  {"accuracy": 0.739, "precision": 0.718, "recall": 0.847, "f1": 0.779,
                       "specificity": 0.611, "c_index": 0.757, "roc_auc": None},
    "clinical_only": {"accuracy": 0.782, "precision": 0.722, "recall": 0.961, "f1": 0.825,
                       "specificity": 0.577, "c_index": 0.834, "roc_auc": None},
}

FILES = {
    "imaging_only": "oof_imaging_only.csv",
    "clinical_only": "oof_clinical_only_restricted_tuned.csv",
    "integrated": "oof_integrated_restricted.csv",
}
MODEL_DISPLAY = {"integrated": "Integrated", "imaging_only": "Chest X-ray (imaging)", "clinical_only": "Clinical Data"}
METRICS = ["accuracy", "precision", "recall", "f1", "specificity", "c_index", "roc_auc"]


def compute_full_metrics(y_true, y_pred, threshold=LOS_THRESHOLD):
    y_true_bin = (y_true > threshold).astype(int)
    y_pred_bin = (y_pred > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
        "f1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
        "c_index": concordance_index(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true_bin, y_pred),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp, "n": len(y_true),
    }


def main():
    rows = []
    new_metrics = {}
    for model, fname in FILES.items():
        df = pd.read_csv(os.path.join(REVISION_DIR, fname))
        m = compute_full_metrics(df["true_los"].values, df["predicted_los"].values)
        new_metrics[model] = m
        for metric in METRICS:
            orig = ORIGINAL_TABLE[model][metric]
            new = m[metric]
            rows.append({
                "model": MODEL_DISPLAY[model], "metric": metric,
                "original": orig, "new_restricted": round(new, 3),
                "delta": round(new - orig, 3) if orig is not None else None,
                "feature_set_note": "original=full (leaky) feature set; new=restricted (triage-time) feature set"
                                     if model != "imaging_only" else "apples-to-apples (imaging never used clinical features)",
            })

    out = pd.DataFrame(rows)
    path = os.path.join(REVISION_DIR, "table2_comparison.csv")
    out.to_csv(path, index=False)

    print("=" * 100)
    print("TABLE II COMPARISON -- original (full feature set) vs. new (restricted feature set), 5-day threshold")
    print("=" * 100)
    print(out.drop(columns=["feature_set_note"]).to_string(index=False))
    print(f"\nSaved {path}")

    print("\n" + "=" * 100)
    print("HEADLINE SHIFTS (one line per model)")
    print("=" * 100)
    for model in FILES:
        o, n = ORIGINAL_TABLE[model], new_metrics[model]
        auc_o = f"{o['roc_auc']:.3f}" if o["roc_auc"] is not None else "not reported"
        print(f"{MODEL_DISPLAY[model]}: "
              f"AUC {auc_o} -> {n['roc_auc']:.3f}  |  "
              f"C-index {o['c_index']:.3f} -> {n['c_index']:.3f}  |  "
              f"Accuracy {o['accuracy']:.3f} -> {n['accuracy']:.3f}  |  "
              f"Precision {o['precision']:.3f} -> {n['precision']:.3f}  |  "
              f"Recall {o['recall']:.3f} -> {n['recall']:.3f}  |  "
              f"F1 {o['f1']:.3f} -> {n['f1']:.3f}  |  "
              f"Specificity {o['specificity']:.3f} -> {n['specificity']:.3f}")


if __name__ == "__main__":
    main()
