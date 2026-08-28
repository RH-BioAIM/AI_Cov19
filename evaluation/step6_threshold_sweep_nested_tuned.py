"""
Committed generator for Supplementary Table S2 (threshold-sweep performance
table). Written to close a provenance gap: supplementary/step6_threshold_
sweep_nested_tuned_CORRECTED.csv (the actual S2 source, AUC=0.871336 at the
5-day threshold) had no generating script anywhere in CR/working/revision/ --
confirmed by exhaustive filename and content grep across the whole tree.

This script reuses the exact per-threshold construction from
step6_threshold_and_mortality.py's part_a_threshold_sweep() (binarize
true_los > thr and predicted_los > thr; compute accuracy, sensitivity,
specificity, precision, F1, ROC AUC, and confusion-matrix counts, for
thr in range(3, 11)) -- READ, not imported or modified, since that function
hardcodes its own output path (step6_threshold_sweep.csv/.png) and would
clobber the existing Gen-1 file if called directly.

Two things are changed relative to that function, per the diagnosed gap:
  (a) input: oof_integrated_restricted_nested_tuned.csv (2026-08-05, the
      finalized nested-CV-tuned restricted integrated model; AUC=0.871336,
      C-index=0.812217 -- independently confirmed elsewhere in this tree,
      e.g. build_nested_tuned_models.log, delong_nested_tuned.log) instead
      of the pre-finalization oof_integrated_restricted.csv.
  (b) output schema: the 11 columns matching the orphan CORRECTED file
      exactly (threshold_days, accuracy, sensitivity, specificity,
      precision, f1, roc_auc, tn, fp, fn, tp) -- no c_index_overall,
      n_positive, or n_negative columns (those were dropped somewhere
      between Gen-1 and the CORRECTED file; C-index is still computed and
      printed to the console below as a sanity cross-check, just not
      written to the CSV).

Output: regenerate_s2_output.csv (new filename -- does NOT overwrite
step6_threshold_sweep.csv, the CORRECTED file, or anything else).

Read-only w.r.t. every existing file: only reads oof_integrated_restricted_
nested_tuned.csv. Writes exactly one new file.
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

    # Threshold-independent sanity check only (not written to the CSV --
    # the CORRECTED file's schema has no c_index_overall column).
    cidx = concordance_index(y_true, y_score)
    print(f"Sanity check -- C-index (threshold-independent): {cidx:.6f} "
          f"(expected ~0.812217, per build_nested_tuned_models.log / delong_nested_tuned.log)")

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

    # exact column order matching the CORRECTED file
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
