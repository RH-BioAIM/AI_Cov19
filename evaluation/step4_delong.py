"""
Step 4: prerequisite (tuned clinical-only-restricted baseline) + paired DeLong
test on ROC AUCs (LOS > 5 days as the binary severity label), on the shared
patient_dict_with_folds-equivalent partition (verified in Check 1).

Prerequisite does NOT overwrite oof_clinical_only_restricted.csv (the
default-hyperparameter run from Step 3) -- writes a new
oof_clinical_only_restricted_tuned.csv instead, and that tuned file is what
DeLong is run against for the restricted comparisons.
"""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from build_integrated_model import load_shared_cohort, build_feature_matrix
from delong import delong_paired_test

REVISION_DIR = os.path.dirname(os.path.abspath(__file__))
LOS_THRESHOLD = 5

# Winning Check 3 config: eta=0.03, max_depth=3, min_child_weight=5, C-index 0.781
TUNED_PARAMS = {"objective": "reg:squarederror", "tree_method": "hist", "seed": 42,
                 "eta": 0.03, "max_depth": 3, "min_child_weight": 5}
NUM_BOOST_ROUND = 5000
EARLY_STOPPING_ROUNDS = 100


def regenerate_tuned_clinical_restricted():
    print("=" * 70)
    print("PREREQUISITE -- regenerating tuned clinical_only_restricted "
          "(eta=0.03, max_depth=3, min_child_weight=5)")
    print("=" * 70)
    df = load_shared_cohort()
    folds = df["fold"]
    X, y = build_feature_matrix(df, restricted=True, include_imaging_pred=False)

    oof_pred = pd.Series(index=X.index, dtype=float)
    for f in sorted(folds.unique()):
        train_idx, test_idx = folds != f, folds == f
        dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx], enable_categorical=True)
        dtest = xgb.DMatrix(X[test_idx], label=y[test_idx], enable_categorical=True)
        model = xgb.train(TUNED_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND,
                          early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                          evals=[(dtest, "eval")], verbose_eval=False)
        oof_pred[test_idx] = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
        print(f"  fold {f}: n_train={train_idx.sum()} n_test={test_idx.sum()} best_iter={model.best_iteration}")

    out = pd.DataFrame({
        "subjectID": df["to_patient_id"].values,
        "fold": folds.values,
        "true_los": y.values,
        "predicted_los": oof_pred.values,
    })
    path = os.path.join(REVISION_DIR, "oof_clinical_only_restricted_tuned.csv")
    out.to_csv(path, index=False)

    from lifelines.utils import concordance_index
    from sklearn.metrics import mean_squared_error
    rmse = float(np.sqrt(mean_squared_error(out["true_los"], out["predicted_los"])))
    cidx = float(concordance_index(out["true_los"], out["predicted_los"]))
    auc = float(roc_auc_score((out["true_los"] > LOS_THRESHOLD).astype(int), out["predicted_los"]))
    print(f"Saved {path} ({len(out)} rows) -- RMSE={rmse:.4f} C-index={cidx:.4f} AUC={auc:.4f}")
    return out


def load_pair(name_a, file_a, name_b, file_b):
    a = pd.read_csv(os.path.join(REVISION_DIR, file_a))[["subjectID", "true_los", "predicted_los"]]
    b = pd.read_csv(os.path.join(REVISION_DIR, file_b))[["subjectID", "true_los", "predicted_los"]]
    merged = a.merge(b, on="subjectID", suffixes=(f"_{name_a}", f"_{name_b}"))
    assert len(merged) == len(a) == len(b), \
        f"pairing lost patients: a={len(a)} b={len(b)} merged={len(merged)}"
    assert (merged[f"true_los_{name_a}"] == merged[f"true_los_{name_b}"]).all(), \
        "true_los disagrees between files for the same patient -- not a valid pair"
    return merged


def run_comparison(label, name_a, file_a, name_b, file_b):
    merged = load_pair(name_a, file_a, name_b, file_b)
    y_true_bin = (merged[f"true_los_{name_a}"] > LOS_THRESHOLD).astype(int).values
    score_a = merged[f"predicted_los_{name_a}"].values
    score_b = merged[f"predicted_los_{name_b}"].values

    # sanity-check DeLong's AUC against sklearn's independent implementation
    sk_auc_a = roc_auc_score(y_true_bin, score_a)
    sk_auc_b = roc_auc_score(y_true_bin, score_b)

    result = delong_paired_test(y_true_bin, score_a, score_b)
    assert abs(result["auc_one"] - sk_auc_a) < 1e-9, "DeLong AUC A != sklearn AUC A"
    assert abs(result["auc_two"] - sk_auc_b) < 1e-9, "DeLong AUC B != sklearn AUC B"

    print(f"\n{label}: {name_a} (AUC={result['auc_one']:.4f}) vs {name_b} (AUC={result['auc_two']:.4f}) "
          f"| n={len(merged)} paired patients | sklearn cross-check: {sk_auc_a:.6f}/{sk_auc_b:.6f} OK")

    return {
        "comparison": label, "model_A": name_a, "model_B": name_b,
        "n_paired": len(merged), "paired": True,
        "auc_A": result["auc_one"], "auc_B": result["auc_two"],
        "delta_auc": result["auc_diff"],
        "ci95_low": result["ci95_low"], "ci95_high": result["ci95_high"],
        "z": result["z"], "p_value": result["p_value"],
    }


def main():
    regenerate_tuned_clinical_restricted()

    print("\n" + "=" * 70)
    print("PAIRED DELONG TEST -- fast DeLong algorithm (Sun & Xu, 2014), "
          f"binary label = LOS > {LOS_THRESHOLD} days")
    print("=" * 70)

    rows = []
    # --- PRIMARY: restricted set ---
    rows.append(run_comparison(
        "restricted: integrated vs clinical_only (tuned)",
        "integrated_restricted", "oof_integrated_restricted.csv",
        "clinical_only_restricted_tuned", "oof_clinical_only_restricted_tuned.csv",
    ))
    rows.append(run_comparison(
        "restricted: integrated vs imaging_only",
        "integrated_restricted", "oof_integrated_restricted.csv",
        "imaging_only", "oof_imaging_only.csv",
    ))
    # --- secondary: full set ---
    rows.append(run_comparison(
        "full: integrated vs clinical_only",
        "integrated_full", "oof_integrated_full.csv",
        "clinical_only_full", "oof_clinical_only_full.csv",
    ))
    rows.append(run_comparison(
        "full: integrated vs imaging_only",
        "integrated_full", "oof_integrated_full.csv",
        "imaging_only", "oof_imaging_only.csv",
    ))

    out = pd.DataFrame(rows)
    path = os.path.join(REVISION_DIR, "delong_results.csv")
    out.to_csv(path, index=False)

    print("\n" + "=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)
    display_cols = ["comparison", "auc_A", "auc_B", "delta_auc", "ci95_low", "ci95_high", "p_value", "paired", "n_paired"]
    print(out[display_cols].to_string(index=False))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
