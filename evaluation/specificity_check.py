"""
Compares two XGBoost configurations on the restricted feature set to test
whether specificity depends on the feature set or the tuning procedure.
"""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
)
from lifelines.utils import concordance_index

import sys
REVISION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # CR/working/revision/
sys.path.insert(0, REVISION_DIR)
from build_integrated_model import load_shared_cohort, build_feature_matrix  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
LOS_THRESHOLD = 5
EXPECTED_INTEGRATED_FEATURES = 117

# Setup (a): original manuscript's hyperparameter config (pure XGBoost
# defaults: max_depth=6, eta=0.3, min_child_weight=1), per the user's
# restated spec -- num_boost_round=1000, early_stopping_rounds=5.
# nthread pinned low deliberately (see hyperparam_tuning.py in the parent
# dir): this shared node's default of nthread=all-visible-cores causes
# ~180x slowdown from thread contention on data this small; nthread=4 avoids
# that without changing results.
DEFAULT_PARAMS = {"objective": "reg:squarederror", "tree_method": "hist", "seed": 42, "nthread": 4}
DEFAULT_NUM_BOOST_ROUND = 1000
DEFAULT_EARLY_STOPPING_ROUNDS = 5


def train_default_setup(X, y, folds):
    """Setup (a): default XGBoost config, same shared 5-fold partition."""
    oof_pred = pd.Series(index=X.index, dtype=float)
    for f in sorted(folds.unique()):
        train_idx, test_idx = folds != f, folds == f
        dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx], enable_categorical=True)
        dtest = xgb.DMatrix(X[test_idx], label=y[test_idx], enable_categorical=True)
        model = xgb.train(DEFAULT_PARAMS, dtrain, num_boost_round=DEFAULT_NUM_BOOST_ROUND,
                           early_stopping_rounds=DEFAULT_EARLY_STOPPING_ROUNDS,
                           evals=[(dtest, "eval")], verbose_eval=False)
        oof_pred[test_idx] = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
        print(f"  [default] fold {f}: n_train={train_idx.sum()} n_test={test_idx.sum()} "
              f"best_iter={model.best_iteration}")
    assert oof_pred.notna().all()
    return oof_pred


def pooled_metrics(y_true, y_pred):
    y_true_bin = (y_true > LOS_THRESHOLD).astype(int)
    y_pred_bin = (y_pred > LOS_THRESHOLD).astype(int)
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


def per_fold_specificity(df):
    rows = []
    for f in sorted(df["fold"].unique()):
        sub = df[df["fold"] == f]
        y_true_bin = (sub["true_los"].values > LOS_THRESHOLD).astype(int)
        y_pred_bin = (sub["predicted_los"].values > LOS_THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        rows.append({"fold": f, "n": len(sub), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                      "specificity": spec, "recall": rec})
    return pd.DataFrame(rows)


def ci95_from_folds(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = values.mean()
    sd = values.std(ddof=1)
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, df=n - 1)
    return mean, sd, (mean - tcrit * se, mean + tcrit * se)


def main():
    print("=" * 70)
    print("Setup (a): XGBoost defaults (original-manuscript hyperparameter config), "
          "restricted feature set (kidney_transplant dropped), shared 5-fold partition")
    print("=" * 70)
    df = load_shared_cohort()
    folds = df["fold"]
    X, y = build_feature_matrix(df, restricted=True, include_imaging_pred=True)
    print(f"Feature count: {X.shape[1]} (expected {EXPECTED_INTEGRATED_FEATURES})")
    assert X.shape[1] == EXPECTED_INTEGRATED_FEATURES
    print(f"Params: {DEFAULT_PARAMS}, num_boost_round={DEFAULT_NUM_BOOST_ROUND}, "
          f"early_stopping_rounds={DEFAULT_EARLY_STOPPING_ROUNDS}")

    oof_default = train_default_setup(X, y, folds)
    default_df = pd.DataFrame({
        "subjectID": df["to_patient_id"].values, "fold": folds.values,
        "true_los": y.values, "predicted_los": oof_default.values,
    })
    default_path = os.path.join(OUT_DIR, "oof_integrated_restricted_defaults.csv")
    default_df.to_csv(default_path, index=False)
    print(f"Saved {default_path} (observation-only, not a committed pipeline output)")

    nested_path = os.path.join(REVISION_DIR, "oof_integrated_restricted_nested_tuned.csv")
    nested_df = pd.read_csv(nested_path)
    assert len(nested_df) == len(default_df) == 1341

    m_default = pooled_metrics(default_df["true_los"].values, default_df["predicted_los"].values)
    m_nested = pooled_metrics(nested_df["true_los"].values, nested_df["predicted_los"].values)

    print("\n" + "=" * 70)
    print("POOLED METRICS -- side by side (5-day threshold)")
    print("=" * 70)
    comp = pd.DataFrame([
        {"setup": "(a) defaults", **m_default},
        {"setup": "(b) nested-CV tuned", **m_nested},
    ])
    comp_display = comp[["setup", "accuracy", "precision", "recall", "f1", "specificity",
                          "c_index", "roc_auc", "tn", "fp", "fn", "tp", "n"]]
    print(comp_display.to_string(index=False))
    comp_display.to_csv(os.path.join(OUT_DIR, "pooled_metrics_comparison.csv"), index=False)

    # --- per-fold specificity, variance, 95% CI ---
    print("\n" + "=" * 70)
    print("PER-FOLD SPECIFICITY")
    print("=" * 70)
    pf_default = per_fold_specificity(default_df)
    pf_nested = per_fold_specificity(nested_df)
    print("\n(a) defaults, per fold:")
    print(pf_default.to_string(index=False))
    print("\n(b) nested-CV tuned, per fold:")
    print(pf_nested.to_string(index=False))

    mean_a, sd_a, ci_a = ci95_from_folds(pf_default["specificity"])
    mean_b, sd_b, ci_b = ci95_from_folds(pf_nested["specificity"])
    print(f"\n(a) defaults:        mean specificity={mean_a:.4f}, SD={sd_a:.4f}, "
          f"95% CI=[{ci_a[0]:.4f}, {ci_a[1]:.4f}] (n=5 folds, t-distribution)")
    print(f"(b) nested-CV tuned: mean specificity={mean_b:.4f}, SD={sd_b:.4f}, "
          f"95% CI=[{ci_b[0]:.4f}, {ci_b[1]:.4f}] (n=5 folds, t-distribution)")

    # paired comparison across the 5 shared folds (same patients per fold in both setups)
    paired = pf_default[["fold", "specificity"]].merge(
        pf_nested[["fold", "specificity"]], on="fold", suffixes=("_a", "_b"))
    paired["diff_b_minus_a"] = paired["specificity_b"] - paired["specificity_a"]
    t_stat, p_val = stats.ttest_rel(paired["specificity_b"], paired["specificity_a"])
    print(f"\nPaired per-fold difference (b - a): mean={paired['diff_b_minus_a'].mean():+.4f}, "
          f"SD={paired['diff_b_minus_a'].std(ddof=1):.4f}")
    print(f"Paired t-test (df=4): t={t_stat:.3f}, p={p_val:.4f} "
          f"(n=5 folds -- low power, interpret cautiously)")
    paired.to_csv(os.path.join(OUT_DIR, "per_fold_specificity_comparison.csv"), index=False)

    ci_overlap = not (ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0])
    print(f"\n95% CIs overlap: {ci_overlap}")
    print(f"Pooled specificity difference (b - a): {m_nested['specificity'] - m_default['specificity']:+.4f} "
          f"vs. within-setup fold-to-fold SD of ~{max(sd_a, sd_b):.4f} "
          f"-- {'within' if abs(m_nested['specificity']-m_default['specificity']) < max(sd_a,sd_b) else 'exceeds'} "
          f"one fold-to-fold SD")

    print("\n" + "=" * 70)
    print("AUC comparison (did tuning move discrimination at all?)")
    print("=" * 70)
    print(f"(a) defaults AUC:        {m_default['roc_auc']:.4f}")
    print(f"(b) nested-CV tuned AUC: {m_nested['roc_auc']:.4f}")
    print(f"Delta: {m_nested['roc_auc'] - m_default['roc_auc']:+.4f}")

    print("\n" + "=" * 70)
    print("BOTTOM LINE")
    print("=" * 70)
    print(f"Pooled specificity: (a) defaults={m_default['specificity']:.4f}  "
          f"(b) nested-tuned={m_nested['specificity']:.4f}  "
          f"delta={m_nested['specificity']-m_default['specificity']:+.4f}")
    print("Nested-CV tuned model remains the committed choice regardless of this result.")


if __name__ == "__main__":
    main()
