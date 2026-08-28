"""
Compares four imaging-fusion representations for the restricted integrated
model, using the same outer 5-fold partition and a fixed XGBoost
configuration (max_depth=4, eta=0.03, min_child_weight=5) for all variants:
V0 the scalar predicted length of stay (117 features), V1 the scalar plus a
16-component PCA of the imaging embeddings (133 features), V2 the scalar
plus the raw 1024-dimensional embeddings (1141 features), and V3 a
fold-honest logistic probability in place of the raw scalar (117 features).
PCA and the probability transform are fit on the other folds only, per
fold.

Input: the clinical feature table and the imaging model's embeddings
(imaging_embeddings.csv).
Output: fusion_variant_comparison.csv.
"""
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from lifelines.utils import concordance_index
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
# build_integrated_model.py lives at ../integrated/.
INTEGRATED_DIR = os.path.join(os.path.dirname(HERE), "integrated")
sys.path.insert(0, INTEGRATED_DIR)
from build_integrated_model import load_shared_cohort, build_feature_matrix  # noqa: E402

LOS_THRESHOLD = 5
EMB_DIM = 1024
PCA_K = 16

# Committed integrated model's MODAL nested-CV-selected config (see build_tuned_models.py log)
FIXED_PARAMS = {"objective": "reg:squarederror", "tree_method": "hist", "seed": 42, "nthread": 4,
                "max_depth": 4, "eta": 0.03, "min_child_weight": 5}
NUM_BOOST_ROUND = 5000
EARLY_STOPPING_ROUNDS = 100

# Reference metrics from the nested-CV-tuned integrated model.
COMMITTED_AUC = 0.871336
COMMITTED_CIDX = 0.812217
COMMITTED_SPEC_AT_5DAY = 0.556634  # recall ~0.976 at literal cutoff=5


def train_fixed_config_cv(X, y, folds, label):
    oof_pred = pd.Series(index=X.index, dtype=float)
    for f in sorted(folds.unique()):
        train_idx, test_idx = folds != f, folds == f
        dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx], enable_categorical=True)
        dtest = xgb.DMatrix(X[test_idx], label=y[test_idx], enable_categorical=True)
        model = xgb.train(FIXED_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND,
                           early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                           evals=[(dtest, "eval")], verbose_eval=False)
        oof_pred[test_idx] = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
        print(f"  [{label}] fold {f}: n_train={train_idx.sum()} n_feat={X.shape[1]} best_iter={model.best_iteration}")
    assert oof_pred.notna().all()
    return oof_pred


def specificity_at_recall(y_true_bin, y_score, target_recall):
    fpr, tpr, thr = roc_curve(y_true_bin, y_score)
    spec = 1 - fpr
    eligible = tpr >= target_recall
    if not eligible.any():
        return float("nan"), float("nan")
    idx = np.where(eligible)[0]
    best = idx[np.argmax(spec[idx])]
    return spec[best], tpr[best]


def per_fold_specificity_ci(y_true_los, y_score, folds, decision_threshold=5.0):
    rows = []
    for f in sorted(folds.unique()):
        m = folds == f
        yt = (y_true_los[m] > LOS_THRESHOLD).astype(int)
        yp = (y_score[m] > decision_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        rows.append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)
    vals = np.array(rows)
    mean, sd = vals.mean(), vals.std(ddof=1)
    se = sd / np.sqrt(len(vals))
    tcrit = stats.t.ppf(0.975, df=len(vals) - 1)
    return mean, sd, (mean - tcrit * se, mean + tcrit * se), vals


def evaluate(name, df, folds, y_true, y_score):
    y_true_bin = (y_true.values > LOS_THRESHOLD).astype(int)
    auc = roc_auc_score(y_true_bin, y_score.values)
    cidx = concordance_index(y_true.values, y_score.values)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, (y_score.values > 5).astype(int), labels=[0, 1]).ravel()
    spec_5day = tn / (tn + fp)
    recall_5day = tp / (tp + fn)
    spec_at_976, achieved_recall = specificity_at_recall(y_true_bin, y_score.values, 0.976)
    mean_spec, sd_spec, ci_spec, per_fold_vals = per_fold_specificity_ci(y_true.values, y_score.values, folds)

    result = {
        "variant": name, "auc": auc, "c_index": cidx,
        "specificity_at_5day_cutoff": spec_5day, "recall_at_5day_cutoff": recall_5day,
        "specificity_at_recall_0.976": spec_at_976, "achieved_recall": achieved_recall,
        "per_fold_spec_mean": mean_spec, "per_fold_spec_sd": sd_spec,
        "per_fold_spec_ci_low": ci_spec[0], "per_fold_spec_ci_high": ci_spec[1],
        "per_fold_specs": list(np.round(per_fold_vals, 4)),
        "delta_auc_vs_committed": auc - COMMITTED_AUC,
        "delta_cidx_vs_committed": cidx - COMMITTED_CIDX,
        "delta_spec_vs_committed": spec_5day - COMMITTED_SPEC_AT_5DAY,
        "exceeds_fold_noise_sd0.05": abs(spec_5day - COMMITTED_SPEC_AT_5DAY) > 0.05,
    }
    print(f"\n[{name}] AUC={auc:.4f} (Δ{result['delta_auc_vs_committed']:+.4f}) "
          f"C-index={cidx:.4f} (Δ{result['delta_cidx_vs_committed']:+.4f}) "
          f"spec@5day={spec_5day:.4f} (Δ{result['delta_spec_vs_committed']:+.4f}) recall@5day={recall_5day:.4f}")
    print(f"    per-fold specificity: mean={mean_spec:.4f} SD={sd_spec:.4f} 95%CI=[{ci_spec[0]:.4f},{ci_spec[1]:.4f}]")
    return result


def fold_honest_pca(embeddings, folds, k=PCA_K):
    """Fit PCA per outer fold on training-fold embeddings only; transform all
    patients (incl. test fold) with that fold's basis. Returns a DataFrame
    of k PCA columns, fold-specific values assembled per patient's own held-out fold."""
    out = pd.DataFrame(index=embeddings.index, columns=[f"pca_{i}" for i in range(k)], dtype=float)
    for f in sorted(folds.unique()):
        train_mask = folds != f
        test_mask = folds == f
        pca = PCA(n_components=k, random_state=42)
        pca.fit(embeddings.loc[train_mask].values)
        out.loc[test_mask, :] = pca.transform(embeddings.loc[test_mask].values)
    return out


def fold_honest_probability(predicted_los, true_los, folds):
    """Fold-honest logistic P(LOS>5) from the raw predicted_los scalar."""
    out = pd.Series(index=predicted_los.index, dtype=float)
    y_bin = (true_los > LOS_THRESHOLD).astype(int)
    for f in sorted(folds.unique()):
        train_mask = folds != f
        test_mask = folds == f
        lr = LogisticRegression()
        lr.fit(predicted_los.loc[train_mask].values.reshape(-1, 1), y_bin.loc[train_mask].values)
        out.loc[test_mask] = lr.predict_proba(predicted_los.loc[test_mask].values.reshape(-1, 1))[:, 1]
    return out


def main():
    print("=" * 70)
    print("Richer imaging-to-clinical fusion -- observation study")
    print("=" * 70)
    print(f"Fixed hyperparameters for all variants (committed model's modal config): {FIXED_PARAMS}")
    print(f"num_boost_round={NUM_BOOST_ROUND}, early_stopping_rounds={EARLY_STOPPING_ROUNDS} "
          f"(on the OUTER test fold -- see module docstring for why)")

    emb_path = os.path.join(HERE, "imaging_embeddings.csv")
    assert os.path.exists(emb_path), f"embeddings not found at {emb_path} -- run extract_embeddings.py first"
    emb_df = pd.read_csv(emb_path)
    print(f"\nLoaded embeddings: {emb_df.shape} ({emb_df.shape[1] - 5} embedding dims expected {EMB_DIM})")
    diff = (emb_df["predicted_los_check"] - emb_df["predicted_los_committed"]).abs()
    print(f"Embedding-extraction sanity check (predicted_los_check vs committed): "
          f"max abs diff={diff.max():.6f}")

    df = load_shared_cohort()
    folds = df["fold"]
    X_clin, y = build_feature_matrix(df, restricted=True, include_imaging_pred=False)
    X_int, _ = build_feature_matrix(df, restricted=True, include_imaging_pred=True)  # has predicted_los col
    print(f"\nClinical-only restricted features: {X_clin.shape[1]}; integrated (scalar fusion): {X_int.shape[1]}")

    # align embeddings to the same patient order as df (merge on subjectID)
    emb_indexed = emb_df.set_index("subjectID")
    subj_order = df["to_patient_id"].values
    assert set(subj_order) == set(emb_indexed.index), "subject mismatch between cohort and embeddings"
    emb_indexed = emb_indexed.loc[subj_order].reset_index(drop=True)
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    assert len(emb_cols) == EMB_DIM
    embeddings = emb_indexed[emb_cols]
    predicted_los = X_int["predicted_los"]

    results = []

    # --- V0: baseline, fixed config, scalar fusion only (calibration reference) ---
    print("\n" + "=" * 70 + "\nV0: baseline_fixed_config (clinical + scalar predicted_los)\n" + "=" * 70)
    oof_v0 = train_fixed_config_cv(X_int, y, folds, "V0")
    results.append(evaluate("V0_baseline_fixed_config", df, folds, y, oof_v0))

    # --- V1: + PCA16(embeddings) ---
    print("\n" + "=" * 70 + "\nV1: embeddings_pca16_added (clinical + scalar + PCA16 embeddings)\n" + "=" * 70)
    pca_feats = fold_honest_pca(embeddings, folds, k=PCA_K)
    X_v1 = pd.concat([X_int.reset_index(drop=True), pca_feats.reset_index(drop=True)], axis=1)
    X_v1.index = X_int.index
    oof_v1 = train_fixed_config_cv(X_v1, y, folds, "V1")
    results.append(evaluate("V1_embeddings_pca16_added", df, folds, y, oof_v1))

    # --- V2: + raw 1024-dim embeddings ---
    print("\n" + "=" * 70 + "\nV2: embeddings_raw1024_added (clinical + scalar + raw 1024 embeddings)\n" + "=" * 70)
    X_v2 = pd.concat([X_int.reset_index(drop=True), embeddings.reset_index(drop=True)], axis=1)
    X_v2.index = X_int.index
    oof_v2 = train_fixed_config_cv(X_v2, y, folds, "V2")
    results.append(evaluate("V2_embeddings_raw1024_added", df, folds, y, oof_v2))

    # --- V3: probability instead of raw LOS ---
    print("\n" + "=" * 70 + "\nV3: probability_instead_of_los (clinical + fold-honest P(LOS>5))\n" + "=" * 70)
    prob = fold_honest_probability(predicted_los, y, folds)
    X_v3 = X_clin.copy()
    X_v3["imaging_prob_severe"] = prob.values
    oof_v3 = train_fixed_config_cv(X_v3, y, folds, "V3")
    results.append(evaluate("V3_probability_instead_of_los", df, folds, y, oof_v3))

    out = pd.DataFrame(results)
    out_path = os.path.join(HERE, "fusion_variant_comparison.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    print("\n" + "=" * 70)
    print("SUMMARY vs COMMITTED (AUC=0.871336, C-index=0.812217, spec@5day=0.556634)")
    print("=" * 70)
    display_cols = ["variant", "auc", "c_index", "specificity_at_5day_cutoff",
                     "delta_auc_vs_committed", "delta_spec_vs_committed", "exceeds_fold_noise_sd0.05"]
    print(out[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
