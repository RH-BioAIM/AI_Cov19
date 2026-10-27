"""
Reproduces the manuscript's Table II from the saved out-of-fold prediction
files, for the imaging-only, clinical-only and integrated models.
"""
import os
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
)
from lifelines.utils import concordance_index

REVISION_DIR = os.path.dirname(os.path.abspath(__file__))
LOS_THRESHOLD = 5

FILES = {
    "imaging_only": "oof_imaging_only.csv",
    "clinical_only": "oof_clinical_only_restricted_nested_tuned.csv",
    "integrated": "oof_integrated_restricted_nested_tuned.csv",
}
MODEL_DISPLAY = {"integrated": "Integrated", "imaging_only": "Chest X-ray (imaging)", "clinical_only": "Clinical Data"}
METRICS = ["accuracy", "precision", "recall", "f1", "specificity", "c_index", "roc_auc"]


def round_half_up(value, places=3):
    """Display rounding. Python's round() breaks ties to even, which would
    report a different third decimal for values sitting exactly on a half."""
    quantum = Decimal(1).scaleb(-places)
    return float(Decimal(repr(float(value))).quantize(quantum, rounding=ROUND_HALF_UP))


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
    for model, fname in FILES.items():
        df = pd.read_csv(os.path.join(REVISION_DIR, fname))
        m = compute_full_metrics(df["true_los"].values, df["predicted_los"].values)
        for metric in METRICS:
            rows.append({"model": MODEL_DISPLAY[model], "metric": metric, "value": m[metric]})
        print(f"{MODEL_DISPLAY[model]}: n={m['n']} "
              f"TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")

    out = pd.DataFrame(rows)
    path = os.path.join(REVISION_DIR, "table2_comparison.csv")
    out.to_csv(path, index=False)

    print("\n" + "=" * 70)
    print(f"TABLE II -- all models at the {LOS_THRESHOLD}-day decision threshold")
    print("=" * 70)
    display = out.copy()
    display["value"] = display["value"].map(lambda v: f"{round_half_up(v):.3f}")
    print(display.to_string(index=False))
    print(f"\nSaved {path} (full precision)")


if __name__ == "__main__":
    main()
