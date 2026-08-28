"""
Rebuilt integrated (clinical + imaging) LOS model for the major-revision response.

Replaces the old, no-longer-present integrated.py. Fixes vs. that script:
  - 5-fold, using the SAME fold partition as the imaging model (not a fresh
    15-fold KFold split).
  - One clean CV loop producing genuine out-of-fold (OOF) predictions; no
    second stub loop / stale-model reuse for the ROC computation.

Shared-fold caveat (see run log / report): patient_dict_with_folds.csv is a
stale artifact whose fold assignments do NOT match the fold assignments
actually used to produce vit_checkpoints/all_fold_predictions.csv (~20%
overlap, i.e. chance level for a 5-way split). The imaging model cannot be
cheaply retrained to match that stale file, so this script treats the 'fold'
column embedded in all_fold_predictions.csv -- the real, already-trained
imaging model's own OOF partition -- as the shared ground truth, and builds
the clinical-only and integrated CV loops on top of that partition instead.

Runs two feature sets (full vs. restricted) for both clinical-only and
integrated models, plus reformats the imaging-only OOF predictions, and
writes all outputs (predictions + a metrics summary) into this revision/
folder.
"""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score, confusion_matrix, mean_squared_error

REVISION_DIR = os.path.dirname(os.path.abspath(__file__))
ALLDATA_PATH = "/path/to/data/AllData.csv"
IMAGING_PRED_PATH = "/path/to/checkpoints/all_fold_predictions.csv"

LOS_THRESHOLD = 5  # severity cutoff, matches paper / sev_eval.py convention
XGB_PARAMS = {"objective": "reg:squarederror", "tree_method": "hist", "seed": 42}
NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 20

# Approved restricted-feature-set drop list (Step 2 table, section B) plus
# visit_start_datetime (raw admission timestamp, not modeled directly), plus
# kidney_transplant (added in the finalization pass: 98.4% missing -- almost
# entirely imputation, no real signal) = 12 columns total. Names below are as
# they actually appear in AllData.csv.
RESTRICTED_DROP_COLS = [
    "visit_concept_name",
    "is_icu",
    "was_ventilated",
    "invasive_vent_days",
    "AcuteHepaticInjury_duringhospitalization",
    "AcuteKidneyInjury_duringhospitalization",
    "kidney_replacement_therapy",
    "therapeutic exnox Boolean",
    "therapeutic heparin Boolean",
    "Other anticoagulation therapy",
    "visit_start_datetime",
    "kidney_transplant",
]

# Always excluded from any feature set: identifiers, the target, and the
# survival-analysis outcome (last_status is reserved, not a predictor here).
# (No "true_los" here: load_shared_cohort() never merges that column in from
# the imaging predictions CSV, so it was a dead/no-op entry -- removed rather
# than left in, since build_feature_matrix now asserts every listed column
# actually exists.)
ALWAYS_EXCLUDE = ["to_patient_id", "subjectID", "length_of_stay", "last_status", "fold"]


def load_shared_cohort():
    clinical = pd.read_csv(ALLDATA_PATH)
    imaging = pd.read_csv(IMAGING_PRED_PATH)  # subjectID, true_los, predicted_los, fold (1-5)

    if not {"subjectID", "predicted_los", "fold"}.issubset(imaging.columns):
        raise ValueError("Missing subjectID/predicted_los/fold in imaging predictions CSV")

    merged = clinical.merge(
        imaging[["subjectID", "predicted_los", "fold"]],
        left_on="to_patient_id", right_on="subjectID", how="inner",
    )
    print(f"Shared cohort: {len(merged)} / {imaging['subjectID'].nunique()} imaging patients "
          f"matched into AllData.csv ({clinical['to_patient_id'].nunique()} clinical patients total).")
    assert merged["to_patient_id"].is_unique, "duplicate patients after merge"
    return merged


def build_feature_matrix(df, restricted, include_imaging_pred):
    drop_cols = list(ALWAYS_EXCLUDE)
    if restricted:
        drop_cols += RESTRICTED_DROP_COLS
    if not include_imaging_pred:
        drop_cols += ["predicted_los"]

    missing = [c for c in drop_cols if c not in df.columns]
    assert not missing, f"drop columns not found (check spelling): {missing}"
    X = df.drop(columns=drop_cols)
    y = df["length_of_stay"]

    # bool -> 0/1 numeric (xgboost's categorical encoder chokes on bool
    # category values); everything else non-numeric -> string category.
    bool_cols = X.select_dtypes(include="bool").columns.tolist()
    for c in bool_cols:
        X[c] = X[c].astype("int8")

    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype(str).astype("category")
    return X, y


def run_fold_cv(X, y, folds, label):
    """Manual leave-one-fold-out CV using a pre-assigned fold column (not a
    fresh KFold split) so every model shares the same test patients per fold."""
    oof_pred = pd.Series(index=X.index, dtype=float)

    for f in sorted(folds.unique()):
        train_idx = folds != f
        test_idx = folds == f

        dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx], enable_categorical=True)
        dtest = xgb.DMatrix(X[test_idx], label=y[test_idx], enable_categorical=True)

        model = xgb.train(
            XGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            evals=[(dtest, "eval")], verbose_eval=False,
        )
        oof_pred[test_idx] = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
        print(f"  [{label}] fold {f}: n_train={train_idx.sum()} n_test={test_idx.sum()} "
              f"best_iter={model.best_iteration}")

    assert oof_pred.notna().all(), "every row must get exactly one OOF prediction"
    return oof_pred


def compute_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    cidx = float(concordance_index(y_true, y_pred))

    y_true_bin = (y_true > LOS_THRESHOLD).astype(int)
    y_pred_bin = (y_pred > LOS_THRESHOLD).astype(int)
    auc = float(roc_auc_score(y_true_bin, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return {
        "rmse": rmse, "c_index": cidx, "roc_auc": auc,
        "sensitivity": sensitivity, "specificity": specificity,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp, "n": len(y_true),
    }


def save_oof(df, subject_col, fold, y_true, y_pred, name):
    out = pd.DataFrame({
        "subjectID": df[subject_col].values,
        "fold": fold.values,
        "true_los": y_true.values,
        "predicted_los": y_pred.values,
    })
    path = os.path.join(REVISION_DIR, f"oof_{name}.csv")
    out.to_csv(path, index=False)
    print(f"Saved {path} ({len(out)} rows)")
    return out


def main():
    df = load_shared_cohort()
    folds = df["fold"]

    metrics_rows = []

    # --- imaging-only: already genuine OOF from vitfreeze.py's own fold loop ---
    imaging_oof = save_oof(df, "to_patient_id", folds, df["length_of_stay"], df["predicted_los"], "imaging_only")
    m = compute_metrics(imaging_oof["true_los"], imaging_oof["predicted_los"])
    metrics_rows.append({"model": "imaging_only", "feature_set": "n/a", **m})

    # --- clinical-only and integrated, full vs restricted ---
    for restricted in (False, True):
        fs_name = "restricted" if restricted else "full"

        X_clin, y_clin = build_feature_matrix(df, restricted=restricted, include_imaging_pred=False)
        oof_clin = run_fold_cv(X_clin, y_clin, folds, f"clinical_only_{fs_name}")
        clin_oof_df = save_oof(df, "to_patient_id", folds, y_clin, oof_clin, f"clinical_only_{fs_name}")
        m = compute_metrics(clin_oof_df["true_los"], clin_oof_df["predicted_los"])
        metrics_rows.append({"model": "clinical_only", "feature_set": fs_name, **m})

        X_int, y_int = build_feature_matrix(df, restricted=restricted, include_imaging_pred=True)
        oof_int = run_fold_cv(X_int, y_int, folds, f"integrated_{fs_name}")
        int_oof_df = save_oof(df, "to_patient_id", folds, y_int, oof_int, f"integrated_{fs_name}")
        m = compute_metrics(int_oof_df["true_los"], int_oof_df["predicted_los"])
        metrics_rows.append({"model": "integrated", "feature_set": fs_name, **m})

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(REVISION_DIR, "metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved {metrics_path}")
    print("\n=== Metrics summary (pooled OOF, LOS threshold = {} days) ===".format(LOS_THRESHOLD))
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
