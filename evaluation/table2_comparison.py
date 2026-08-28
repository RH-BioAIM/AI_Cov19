"""
Canonical Table II replacement: original paper's reported metrics (full/leaky
feature set) vs. the new restricted-feature-set models, side by side.

Pure metric computation on existing OOF prediction files -- no retraining.
Uses the TUNED clinical-only-restricted predictions (eta=0.03, depth=3), not
the default underfit run, per the standardized baseline.

NOTE ON COMPARABILITY: original "Integrated"/"Clinical Data" used the full
leaky feature set; "new_restricted" uses triage-time-only features. This is
NOT an apples-to-apples model comparison -- the deltas reflect the cost of
removing leakage, not a regression. "Chest X-ray (imaging)" is the one row
that IS apples-to-apples (the imaging model never used clinical features, so
there was nothing to restrict); its delta reflects only run-to-run variation.
"""
import os
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
)
from lifelines.utils import concordance_index

REVISION_DIR = os.path.dirname(os.path.abspath(__file__))
LOS_THRESHOLD = 5

# Original paper's Table II (5-day threshold) + integrated ROC AUC from text.
# Imaging-only and clinical-only ROC AUC were not reported in the original
# paper -- left as None, not fabricated.
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
