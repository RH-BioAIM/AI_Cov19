"""
Trains the restricted clinical-only and integrated models using the
restricted feature set (kidney_transplant dropped, 116 clinical features)
and the tuned XGBoost configuration (eta=0.03, max_depth=3,
min_child_weight=5, num_boost_round=5000, early_stopping_rounds=100).
Full-feature-set models are not processed by this script.

Input: the clinical feature table and the imaging model's out-of-fold
predictions (all_fold_predictions.csv).
Output: oof_clinical_only_restricted_final.csv,
oof_integrated_restricted_final.csv.
"""
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score, mean_squared_error

from build_integrated_model import load_shared_cohort, build_feature_matrix, RESTRICTED_DROP_COLS
# delong_analysis.py lives in ../evaluation/.
HERE = os.path.dirname(os.path.abspath(__file__))
EVALUATION_DIR = os.path.normpath(os.path.join(HERE, "..", "evaluation"))
sys.path.insert(0, EVALUATION_DIR)
from delong_analysis import TUNED_PARAMS, NUM_BOOST_ROUND, EARLY_STOPPING_ROUNDS  # noqa: E402  eta=0.03, depth=3, mcw=5, 5000/100

REVISION_DIR = os.path.dirname(os.path.abspath(__file__))
LOS_THRESHOLD = 5

EXPECTED_CLINICAL_FEATURES = 116
EXPECTED_INTEGRATED_FEATURES = 117


def run_tuned_cv(X, y, folds, label):
    oof_pred = pd.Series(index=X.index, dtype=float)
    for f in sorted(folds.unique()):
        train_idx, test_idx = folds != f, folds == f
        dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx], enable_categorical=True)
        dtest = xgb.DMatrix(X[test_idx], label=y[test_idx], enable_categorical=True)
        model = xgb.train(TUNED_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND,
                           early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                           evals=[(dtest, "eval")], verbose_eval=False)
        oof_pred[test_idx] = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
        print(f"  [{label}] fold {f}: n_train={train_idx.sum()} n_test={test_idx.sum()} "
              f"best_iter={model.best_iteration}")
    assert oof_pred.notna().all(), "every row must get exactly one OOF prediction"
    return oof_pred


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


def main():
    print("=" * 70)
    print("Building restricted clinical and integrated models")
    print("=" * 70)
    print(f"RESTRICTED_DROP_COLS ({len(RESTRICTED_DROP_COLS)} cols): {RESTRICTED_DROP_COLS}")
    assert "kidney_transplant" in RESTRICTED_DROP_COLS, "Change 1 not applied -- kidney_transplant missing from drop list"

    df = load_shared_cohort()
    folds = df["fold"]

    X_clin, y_clin = build_feature_matrix(df, restricted=True, include_imaging_pred=False)
    X_int, y_int = build_feature_matrix(df, restricted=True, include_imaging_pred=True)

    print(f"\nFeature count check:")
    print(f"  restricted clinical-only: {X_clin.shape[1]} features "
          f"(expected {EXPECTED_CLINICAL_FEATURES}, was 117 before dropping kidney_transplant)")
    print(f"  restricted integrated:    {X_int.shape[1]} features "
          f"(expected {EXPECTED_INTEGRATED_FEATURES} = {EXPECTED_CLINICAL_FEATURES} clinical + 1 imaging-derived predicted_los)")
    assert X_clin.shape[1] == EXPECTED_CLINICAL_FEATURES, \
        f"restricted clinical feature count {X_clin.shape[1]} != expected {EXPECTED_CLINICAL_FEATURES}"
    assert X_int.shape[1] == EXPECTED_INTEGRATED_FEATURES, \
        f"restricted integrated feature count {X_int.shape[1]} != expected {EXPECTED_INTEGRATED_FEATURES}"
    assert "kidney_transplant" not in X_clin.columns and "kidney_transplant" not in X_int.columns
    assert "predicted_los" in X_int.columns and "predicted_los" not in X_clin.columns
    print("  PASS -- counts and column membership confirmed.")

    print(f"\nTuned XGBoost config (same for both models): {TUNED_PARAMS}, "
          f"num_boost_round={NUM_BOOST_ROUND}, early_stopping_rounds={EARLY_STOPPING_ROUNDS}")

    print("\n--- clinical_only_restricted_final (tuned, kidney_transplant dropped) ---")
    oof_clin = run_tuned_cv(X_clin, y_clin, folds, "clinical_only_restricted_final")
    clin_df, clin_m = save_and_report(df, folds, y_clin, oof_clin, "clinical_only_restricted_final")

    print("\n--- integrated_restricted_final (NOW TUNED, kidney_transplant dropped) ---")
    oof_int = run_tuned_cv(X_int, y_int, folds, "integrated_restricted_final")
    int_df, int_m = save_and_report(df, folds, y_int, oof_int, "integrated_restricted_final")

    print("\n" + "=" * 70)
    print("SUMMARY -- new (final) restricted models")
    print("=" * 70)
    summary = pd.DataFrame([
        {"model": "clinical_only_restricted_final", "n_features": X_clin.shape[1], **clin_m},
        {"model": "integrated_restricted_final", "n_features": X_int.shape[1], **int_m},
    ])
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(REVISION_DIR, "metrics_summary_final.csv"), index=False)
    print(f"\nSaved {os.path.join(REVISION_DIR, 'metrics_summary_final.csv')}")


if __name__ == "__main__":
    main()
