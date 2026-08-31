"""
Compares replacing the scalar imaging feature with Swin embeddings for the
restricted integrated model.
"""
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score,
)
from lifelines.utils import concordance_index
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
FUSION_DIR = os.path.dirname(HERE)  # fusion_experiments/
# build_integrated_model.py lives in ../../integrated/.
INTEGRATED_DIR = os.path.join(os.path.dirname(FUSION_DIR), "integrated")
sys.path.insert(0, INTEGRATED_DIR)
from build_integrated_model import load_shared_cohort, build_feature_matrix  # noqa: E402

LOS_THRESHOLD = 5
EMB_DIM = 1024

FIXED_PARAMS = {"objective": "reg:squarederror", "tree_method": "hist", "seed": 42, "nthread": 4,
                "max_depth": 4, "eta": 0.03, "min_child_weight": 5}
NUM_BOOST_ROUND = 5000
EARLY_STOPPING_ROUNDS = 100

# Reference rows, for direct comparison in the printed/saved table.
COMMITTED = {
    "variant": "REFERENCE_committed_scalar_nested_cv", "auc": 0.871336, "c_index": 0.812217,
    "specificity": 0.556634, "recall": 0.976, "tn": 344, "fp": 274, "fn": 17, "tp": 706,
    "per_fold_spec_sd": np.nan, "note": "per-outer-fold nested-CV tuned, not fixed-config",
}
# Added-embeddings variants under the same fixed config and partition,
# included as same-config baselines for comparison.
V0_BASELINE_FIXED_CONFIG = {
    "variant": "REFERENCE_V0_scalar_only_fixed_config", "auc": 0.870440, "c_index": 0.812620,
    "specificity": 0.537217, "recall": 0.971, "tn": None, "fp": None, "fn": None, "tp": None,
    "per_fold_spec_sd": 0.0478, "note": "fixed config, scalar predicted_los only, no embeddings",
}
V1_PCA16_ADDED = {
    "variant": "REFERENCE_V1_scalar_plus_pca16_added", "auc": 0.868019, "c_index": 0.811373,
    "specificity": 0.548544, "recall": 0.9696, "tn": None, "fp": None, "fn": None, "tp": None,
    "per_fold_spec_sd": 0.0577, "note": "scalar retained, PCA16 embeddings added on top",
}
V2_RAW1024_ADDED = {
    "variant": "REFERENCE_V2_scalar_plus_raw1024_added", "auc": 0.862334, "c_index": 0.806608,
    "specificity": 0.512945, "recall": 0.9820, "tn": None, "fp": None, "fn": None, "tp": None,
    "per_fold_spec_sd": 0.0867, "note": "scalar retained, raw 1024-dim embeddings added on top",
}


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


def fold_honest_pca(embeddings, folds, k):
    out = pd.DataFrame(index=embeddings.index, columns=[f"pca_{i}" for i in range(k)], dtype=float)
    for f in sorted(folds.unique()):
        train_mask = folds != f
        test_mask = folds == f
        pca = PCA(n_components=k, random_state=42)
        pca.fit(embeddings.loc[train_mask].values)
        out.loc[test_mask, :] = pca.transform(embeddings.loc[test_mask].values)
    return out


def per_fold_specificity_sd(y_true_los, y_score, folds, decision_threshold=5.0):
    rows = []
    for f in sorted(folds.unique()):
        m = folds == f
        yt = (y_true_los[m] > LOS_THRESHOLD).astype(int)
        yp = (y_score[m] > decision_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        rows.append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)
    vals = np.array(rows)
    return vals.mean(), vals.std(ddof=1), vals


def evaluate(name, y_true, y_score, folds, note=""):
    y_true_bin = (y_true.values > LOS_THRESHOLD).astype(int)
    y_pred_bin = (y_score.values > LOS_THRESHOLD).astype(int)
    auc = roc_auc_score(y_true_bin, y_score.values)
    cidx = concordance_index(y_true.values, y_score.values)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    accuracy = accuracy_score(y_true_bin, y_pred_bin)
    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    pf_mean, pf_sd, pf_vals = per_fold_specificity_sd(y_true.values, y_score.values, folds)

    result = {
        "variant": name, "auc": auc, "c_index": cidx,
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        "specificity": specificity,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "per_fold_spec_mean": pf_mean, "per_fold_spec_sd": pf_sd,
        "per_fold_specs": list(np.round(pf_vals, 4)),
        "delta_auc_vs_committed": auc - COMMITTED["auc"],
        "delta_spec_vs_committed": specificity - COMMITTED["specificity"],
        "delta_auc_vs_v0_fixedconfig": auc - V0_BASELINE_FIXED_CONFIG["auc"],
        "delta_spec_vs_v0_fixedconfig": specificity - V0_BASELINE_FIXED_CONFIG["specificity"],
        "exceeds_fold_noise_sd0.05_vs_committed": abs(specificity - COMMITTED["specificity"]) > 0.05,
        "note": note,
    }
    print(f"\n[{name}] AUC={auc:.4f} (Δ{result['delta_auc_vs_committed']:+.4f} vs committed) "
          f"C-index={cidx:.4f} specificity={specificity:.4f} "
          f"(Δ{result['delta_spec_vs_committed']:+.4f} vs committed) recall={recall:.4f}")
    print(f"    accuracy={accuracy:.4f} precision={precision:.4f} f1={f1:.4f} "
          f"TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"    per-fold specificity: mean={pf_mean:.4f} SD={pf_sd:.4f}")
    return result


def main():
    print("=" * 70)
    print("Replace-fusion observation study: embeddings REPLACE scalar predicted_los")
    print("=" * 70)
    print(f"Fixed hyperparameters (same as prior added-embeddings run): {FIXED_PARAMS}")
    print(f"num_boost_round={NUM_BOOST_ROUND}, early_stopping_rounds={EARLY_STOPPING_ROUNDS} (outer test fold)")

    emb_path = os.path.join(FUSION_DIR, "imaging_embeddings.csv")
    assert os.path.exists(emb_path), f"embeddings not found at {emb_path} -- run extract_embeddings.py first"
    emb_df = pd.read_csv(emb_path)
    print(f"\nLoaded embeddings: {emb_df.shape} ({emb_df.shape[1] - 5} embedding dims expected {EMB_DIM})")
    diff = (emb_df["predicted_los_check"] - emb_df["predicted_los_committed"]).abs()
    print(f"Embedding-extraction sanity check (predicted_los_check vs committed): max abs diff={diff.max():.6f}")

    df = load_shared_cohort()
    folds = df["fold"]
    X_clin, y = build_feature_matrix(df, restricted=True, include_imaging_pred=False)
    print(f"\nClinical-only restricted features (NO imaging): {X_clin.shape[1]}")

    emb_indexed = emb_df.set_index("subjectID")
    subj_order = df["to_patient_id"].values
    assert set(subj_order) == set(emb_indexed.index), "subject mismatch between cohort and embeddings"
    emb_indexed = emb_indexed.loc[subj_order].reset_index(drop=True)
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    assert len(emb_cols) == EMB_DIM
    embeddings = emb_indexed[emb_cols]
    embeddings.index = X_clin.index

    results = []

    # --- R1: clinical + PCA16(embeddings), NO scalar ---
    print("\n" + "=" * 70 + "\nR1: pca16_replace (clinical + PCA16 embeddings, no predicted_los)\n" + "=" * 70)
    pca16 = fold_honest_pca(embeddings, folds, k=16)
    X_r1 = pd.concat([X_clin.reset_index(drop=True), pca16.reset_index(drop=True)], axis=1)
    X_r1.index = X_clin.index
    oof_r1 = train_fixed_config_cv(X_r1, y, folds, "R1")
    results.append(evaluate("R1_pca16_replace", y, oof_r1, folds,
                             note="clinical + PCA16(embeddings), scalar predicted_los REMOVED"))

    # --- R2: clinical + raw 1024-dim embeddings, NO scalar ---
    print("\n" + "=" * 70 + "\nR2: raw1024_replace (clinical + raw 1024 embeddings, no predicted_los)\n" + "=" * 70)
    X_r2 = pd.concat([X_clin.reset_index(drop=True), embeddings.reset_index(drop=True)], axis=1)
    X_r2.index = X_clin.index
    oof_r2 = train_fixed_config_cv(X_r2, y, folds, "R2")
    results.append(evaluate("R2_raw1024_replace", y, oof_r2, folds,
                             note="clinical + raw 1024-dim embeddings, scalar predicted_los REMOVED"))

    # --- R3: clinical + PCA-k(embeddings) at k in {8, 32, 64}, NO scalar ---
    for k in (8, 32, 64):
        print("\n" + "=" * 70 + f"\nR3: pca{k}_replace (clinical + PCA{k} embeddings, no predicted_los)\n" + "=" * 70)
        pca_k = fold_honest_pca(embeddings, folds, k=k)
        X_rk = pd.concat([X_clin.reset_index(drop=True), pca_k.reset_index(drop=True)], axis=1)
        X_rk.index = X_clin.index
        oof_rk = train_fixed_config_cv(X_rk, y, folds, f"R3_pca{k}")
        results.append(evaluate(f"R3_pca{k}_replace", y, oof_rk, folds,
                                 note=f"clinical + PCA{k}(embeddings), scalar predicted_los REMOVED"))

    out = pd.DataFrame(results)

    # assemble full comparison table incl. reference rows
    ref_rows = []
    for ref in (COMMITTED, V0_BASELINE_FIXED_CONFIG, V1_PCA16_ADDED, V2_RAW1024_ADDED):
        ref_rows.append({
            "variant": ref["variant"], "auc": ref["auc"], "c_index": ref["c_index"],
            "specificity": ref["specificity"], "recall": ref["recall"],
            "tn": ref["tn"], "fp": ref["fp"], "fn": ref["fn"], "tp": ref["tp"],
            "per_fold_spec_sd": ref["per_fold_spec_sd"], "note": ref["note"],
        })
    ref_df = pd.DataFrame(ref_rows)

    out_path = os.path.join(HERE, "replace_variant_comparison.csv")
    out.to_csv(out_path, index=False)
    ref_path = os.path.join(HERE, "reference_rows.csv")
    ref_df.to_csv(ref_path, index=False)
    print(f"\nSaved {out_path}")
    print(f"Saved {ref_path}")

    print("\n" + "=" * 70)
    print("FULL COMPARISON (reference rows + replace variants)")
    print("=" * 70)
    combined = pd.concat([ref_df, out], axis=0, ignore_index=True, sort=False)
    display_cols = ["variant", "auc", "c_index", "specificity", "recall", "tn", "fp", "fn", "tp",
                     "per_fold_spec_sd"]
    print(combined[display_cols].to_string(index=False))
    combined.to_csv(os.path.join(HERE, "full_comparison_with_references.csv"), index=False)

    print("\n" + "=" * 70)
    print("KEY QUESTION: does any embedding-ONLY (replace) fusion beat the scalar-only")
    print("fixed-config fusion (V0, spec=0.5372, AUC=0.8704) on specificity or AUC?")
    print("=" * 70)
    for r in results:
        beats_v0_spec = r["specificity"] > V0_BASELINE_FIXED_CONFIG["specificity"]
        beats_v0_auc = r["auc"] > V0_BASELINE_FIXED_CONFIG["auc"]
        print(f"  {r['variant']}: spec={r['specificity']:.4f} "
              f"({'beats' if beats_v0_spec else 'does not beat'} V0 scalar-only spec) | "
              f"AUC={r['auc']:.4f} ({'beats' if beats_v0_auc else 'does not beat'} V0 scalar-only AUC) | "
              f"exceeds ±0.05 fold-noise vs committed: {r['exceeds_fold_noise_sd0.05_vs_committed']}")


if __name__ == "__main__":
    main()
