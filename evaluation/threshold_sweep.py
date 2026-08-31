"""
Computes classification metrics for the integrated model across
length-of-stay decision thresholds from 3 to 10 days. Generates
Supplementary Table S2.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, accuracy_score,
    precision_score, f1_score,
)
from lifelines.utils import concordance_index

REVISION_DIR = os.path.dirname(os.path.abspath(__file__))
OOF_PATH = os.path.join(REVISION_DIR, "oof_integrated_restricted_nested_tuned.csv")
OUT_PATH = os.path.join(REVISION_DIR, "regenerate_s2_output.csv")


def threshold_sweep(oof):
    y_true, y_score = oof["true_los"].values, oof["predicted_los"].values

    # Threshold-independent sanity check, printed only, not written to the CSV.
    cidx = concordance_index(y_true, y_score)
    print(f"Sanity check -- C-index (threshold-independent): {cidx:.6f} "
          f"(expected ~0.812217, per build_tuned_models.log / delong_comparison.log)")

    rows = []
    for thr in range(3, 11):  # 3..10 inclusive -- same loop as part_a_threshold_sweep()
        y_true_bin = (y_true > thr).astype(int)
        y_pred_bin = (y_score > thr).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        acc = accuracy_score(y_true_bin, y_pred_bin)
        try:
            auc = roc_auc_score(y_true_bin, y_score)
        except ValueError:
            auc = float("nan")

        rows.append({
            "threshold_days": thr,
            "accuracy": acc, "sensitivity": sens, "specificity": spec,
            "precision": prec, "f1": f1, "roc_auc": auc,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        })

    # Column order for the output table.
    out = pd.DataFrame(rows, columns=[
        "threshold_days", "accuracy", "sensitivity", "specificity",
        "precision", "f1", "roc_auc", "tn", "fp", "fn", "tp",
    ])
    return out


def main():
    assert not os.path.exists(OUT_PATH), f"refusing to overwrite existing {OUT_PATH}"

    print(f"Reading {OOF_PATH}")
    oof = pd.read_csv(OOF_PATH)
    assert len(oof) == 1341, f"expected 1341 patients, got {len(oof)}"
    print(f"Loaded {len(oof)} patients")

    out = threshold_sweep(oof)
    print("\nRegenerated sweep:")
    print(out.to_string(index=False))

    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
