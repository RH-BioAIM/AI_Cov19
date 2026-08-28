"""
Option C: proper per-model hyperparameter tuning via nested CV, replacing the
asymmetric setup (clinical-only hand-tuned via a one-off sweep, integrated on
defaults) and the forced-symmetric _final setup (both hand-set to the same
clinical-winning config, without checking whether that config actually suits
the integrated model too).

Runs hyperparam_tuning.nested_cv_select_and_fit for clinical_only_restricted
and integrated_restricted (kidney_transplant already dropped -- 116 and 117
features respectively, from build_integrated_model.py's updated
RESTRICTED_DROP_COLS). Full-set models untouched, not regenerated here.

Outputs use a `_nested_tuned` suffix, NOT the literal `_tuned` suggested in
the prompt -- `oof_clinical_only_restricted_tuned.csv` already exists (the
older, non-searched, pre-kidney-transplant-drop fixed-config file, still read
by step4_delong.py / table2_comparison.py) and overwriting or shadowing it
with a same-named-but-different-meaning file would be a silent correctness
hazard. `_nested_tuned` is unambiguous and doesn't collide with anything on
disk. Does not overwrite oof_clinical_only_restricted_final.csv or
oof_integrated_restricted_final.csv (the previous run's forced-symmetric
files) either.
"""
import os
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score, mean_squared_error

from build_integrated_model import load_shared_cohort, build_feature_matrix
from hyperparam_tuning import nested_cv_select_and_fit, modal_combo

REVISION_DIR = os.path.dirname(os.path.abspath(__file__))
LOS_THRESHOLD = 5

EXPECTED_CLINICAL_FEATURES = 116
EXPECTED_INTEGRATED_FEATURES = 117


def save_and_report(df, folds, y_true, y_pred, name):
    out = pd.DataFrame({
        "subjectID": df["to_patient_id"].values,
        "fold": folds.values,
        "true_los": y_true.values,
        "predicted_los": y_pred.values,
    })
    path = os.path.join(REVISION_DIR, f"oof_{name}.csv")
    out.to_csv(path, index=False)

    rmse = float(np.sqrt(mean_squared_error(out["true_los"], out["predicted_los"])))
    cidx = float(concordance_index(out["true_los"], out["predicted_los"]))
    auc = float(roc_auc_score((out["true_los"] > LOS_THRESHOLD).astype(int), out["predicted_los"]))
    print(f"Saved {path} ({len(out)} rows) -- RMSE={rmse:.4f} C-index={cidx:.4f} AUC={auc:.4f}")
    return out, {"rmse": rmse, "c_index": cidx, "roc_auc": auc}


def run_model(df, folds, X, y, label, oof_name):
    print(f"\n{'=' * 70}\nNESTED CV TUNING -- {label}\n{'=' * 70}")
    oof_pred, fold_log, selection_table = nested_cv_select_and_fit(X, y, folds, label)
    oof_df, metrics = save_and_report(df, folds, y, oof_pred, oof_name)

    modal, counts = modal_combo(fold_log)
    print(f"\nPer-outer-fold selections ({label}):")
    print(fold_log[["outer_fold", "max_depth", "eta", "min_child_weight",
                     "selected_by_inner_c_index", "final_num_boost_round"]].to_string(index=False))
    print(f"\nModal (most frequently selected) config for {label}: {modal}")
    print(counts.to_string(index=False))

    return oof_df, metrics, fold_log, selection_table, modal


def main():
    print("=" * 70)
    print("OPTION C: nested-CV hyperparameter search, per model, restricted feature set")
    print("=" * 70)
    df = load_shared_cohort()
    folds = df["fold"]

    X_clin, y_clin = build_feature_matrix(df, restricted=True, include_imaging_pred=False)
    X_int, y_int = build_feature_matrix(df, restricted=True, include_imaging_pred=True)
    print(f"\nFeature counts: clinical-only restricted = {X_clin.shape[1]} "
          f"(expected {EXPECTED_CLINICAL_FEATURES}), integrated restricted = {X_int.shape[1]} "
          f"(expected {EXPECTED_INTEGRATED_FEATURES})")
    assert X_clin.shape[1] == EXPECTED_CLINICAL_FEATURES
    assert X_int.shape[1] == EXPECTED_INTEGRATED_FEATURES

    clin_df, clin_m, clin_log, clin_sel, clin_modal = run_model(
        df, folds, X_clin, y_clin, "clinical_only_restricted", "clinical_only_restricted_nested_tuned")
    int_df, int_m, int_log, int_sel, int_modal = run_model(
        df, folds, X_int, y_int, "integrated_restricted", "integrated_restricted_nested_tuned")

    all_log = pd.concat([clin_log, int_log], ignore_index=True)
    all_log.to_csv(os.path.join(REVISION_DIR, "hyperparam_search_log.csv"), index=False)
    all_sel = pd.concat([clin_sel, int_sel], ignore_index=True)
    all_sel.to_csv(os.path.join(REVISION_DIR, "hyperparam_search_full_grid.csv"), index=False)
    print(f"\nSaved hyperparam_search_log.csv ({len(all_log)} rows: 5 outer folds x 2 models)")
    print(f"Saved hyperparam_search_full_grid.csv ({len(all_sel)} rows: 5 outer folds x 18 combos x 2 models)")

    summary = pd.DataFrame([
        {"model": "clinical_only_restricted_nested_tuned", "n_features": X_clin.shape[1],
         "modal_max_depth": clin_modal["max_depth"], "modal_eta": clin_modal["eta"],
         "modal_min_child_weight": clin_modal["min_child_weight"], **clin_m},
        {"model": "integrated_restricted_nested_tuned", "n_features": X_int.shape[1],
         "modal_max_depth": int_modal["max_depth"], "modal_eta": int_modal["eta"],
         "modal_min_child_weight": int_modal["min_child_weight"], **int_m},
    ])
    summary.to_csv(os.path.join(REVISION_DIR, "metrics_summary_nested_tuned.csv"), index=False)
    print("\n" + "=" * 70)
    print("SUMMARY -- nested-CV tuned restricted models")
    print("=" * 70)
    print(summary.to_string(index=False))
    print(f"\nSaved {os.path.join(REVISION_DIR, 'metrics_summary_nested_tuned.csv')}")


if __name__ == "__main__":
    main()
