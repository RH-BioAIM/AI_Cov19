"""
Full classification metrics at three LOS decision thresholds (5-day fixed,
6.71-day Youden-optimal, 7-day) for all three FINAL nested-tuned restricted
models. Same construction as final_metrics_tuned.py (ground-truth severity
label fixed at true_los > 5 throughout; only the cutoff applied to the
predicted SCORE varies by row; AUC/C-index are threshold-independent,
computed once per model), extended to a third threshold for operating-point
comparison.

6.71-day kept at the same fixed value used throughout the revision (see
final_metrics_by_threshold_v2.py / final_metrics_tuned.py) for comparability
across models, not re-derived per model.

Pure computation on existing OOF prediction files -- no retraining.
"""
import os
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
)
from lifelines.utils import concordance_index

REVISION_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_THRESHOLD = 5
DECISION_THRESHOLDS = [
    ("5-day (fixed)", 5.0),
    ("6.71-day (Youden-optimal, pre-finalization)", 6.712525844573975),
    ("7-day", 7.0),
]

FILES = {
    "imaging_only": "oof_imaging_only.csv",
    "clinical_only_restricted_nested_tuned": "oof_clinical_only_restricted_nested_tuned.csv",
    "integrated_restricted_nested_tuned": "oof_integrated_restricted_nested_tuned.csv",
}


def compute_row(y_true_los, y_score, decision_threshold, auc, cidx):
    y_true_bin = (y_true_los > GROUND_TRUTH_THRESHOLD).astype(int)
    y_pred_bin = (y_score > decision_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
        "f1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
        "c_index": cidx, "roc_auc": auc,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def main():
    rows = []
    for model, fname in FILES.items():
        path = os.path.join(REVISION_DIR, fname)
        df = pd.read_csv(path)
        assert len(df) == 1341, f"{fname}: expected 1341 rows, got {len(df)}"
        y_true, y_score = df["true_los"].values, df["predicted_los"].values

        y_true_bin_fixed = (y_true > GROUND_TRUTH_THRESHOLD).astype(int)
        auc = roc_auc_score(y_true_bin_fixed, y_score)
        cidx = concordance_index(y_true, y_score)

        for label, thr in DECISION_THRESHOLDS:
            m = compute_row(y_true, y_score, thr, auc, cidx)
            rows.append({
                "model": model, "decision_threshold_label": label, "decision_threshold_days": round(thr, 2),
                "accuracy": round(m["accuracy"], 3), "precision": round(m["precision"], 3),
                "recall": round(m["recall"], 3), "f1": round(m["f1"], 3),
                "specificity": round(m["specificity"], 3),
                "c_index": round(m["c_index"], 3), "roc_auc": round(m["roc_auc"], 3),
                "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"], "n": len(df),
            })

    out = pd.DataFrame(rows)
    path = os.path.join(REVISION_DIR, "metrics_by_threshold_5_67_7.csv")
    out.to_csv(path, index=False)
    print(f"Saved {path}\n")
    print(out.to_string(index=False))

    # --- consistency check: C-index/AUC identical across the 3 threshold rows, per model ---
    print("\n" + "=" * 70)
    print("CONSISTENCY CHECK -- C-index and AUC identical across all 3 threshold rows")
    print("=" * 70)
    all_match = True
    for model in FILES:
        sub = out[out["model"] == model]
        cidx_vals, auc_vals = sub["c_index"].unique(), sub["roc_auc"].unique()
        ok = len(cidx_vals) == 1 and len(auc_vals) == 1
        all_match = all_match and ok
        print(f"{model}: C-index={list(cidx_vals)}  AUC={list(auc_vals)}  -> {'OK' if ok else 'MISMATCH'}")
    print(f"OVERALL: {'PASS' if all_match else 'FAIL'}")

    # --- consistency check vs Table II's 5-day integrated row ---
    print("\n" + "=" * 70)
    print("CONSISTENCY CHECK -- 5-day integrated row vs Table II (accuracy 0.783, recall 0.977, specificity 0.557)")
    print("=" * 70)
    integ_5d = out[(out["model"] == "integrated_restricted_nested_tuned") &
                   (out["decision_threshold_days"] == 5.0)].iloc[0]
    expected = {"accuracy": 0.783, "recall": 0.977, "specificity": 0.557}
    for k, v in expected.items():
        actual = integ_5d[k]
        match = abs(actual - v) < 0.0015  # rounding tolerance
        print(f"  {k}: table={v}  this_run={actual}  -> {'MATCH' if match else 'MISMATCH -- INVESTIGATE'}")

    # --- flagged comparison: integrated specificity 5d->7d, recall at 7d ---
    print("\n" + "=" * 70)
    print("FLAGGED -- integrated model: specificity improvement and recall tradeoff, 5-day -> 7-day")
    print("=" * 70)
    integ = out[out["model"] == "integrated_restricted_nested_tuned"].set_index("decision_threshold_days")
    spec_5, spec_7 = integ.loc[5.0, "specificity"], integ.loc[7.0, "specificity"]
    rec_5, rec_7 = integ.loc[5.0, "recall"], integ.loc[7.0, "recall"]
    print(f"  Specificity: {spec_5:.3f} (5-day) -> {spec_7:.3f} (7-day)  (+{spec_7 - spec_5:.3f})")
    print(f"  Recall:      {rec_5:.3f} (5-day) -> {rec_7:.3f} (7-day)  ({rec_7 - rec_5:+.3f})")


if __name__ == "__main__":
    main()
