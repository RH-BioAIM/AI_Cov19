"""
Compares replacing the scalar imaging feature with Swin embeddings for
the restricted integrated model, with the number of boosting rounds for
each outer fold chosen by an inner cross-validation on that fold's
training data only.
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
# build_integrated_model.py lives at ../../integrated/.
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
    "per_fold_spec_sd": np.nan, "note": "per-outer-fold nested-CV tuned, NOT fixed-config -- see docstring",
}
# From the prior ADDED-embeddings run (fusion_variant_comparison.csv),
# fixed-config, same partition -- included as the fair same-config baselines/comparators.
V0_BASELINE_FIXED_CONFIG = {
    "variant": "REFERENCE_V0_scalar_only_fixed_config", "auc": 0.870440, "c_index": 0.812620,
    "specificity": 0.537217, "recall": 0.971, "tn": None, "fp": None, "fn": None, "tp": None,
    "per_fold_spec_sd": 0.0478, "note": "prior run -- same fixed config, scalar predicted_los only, no embeddings",
}
V1_PCA16_ADDED = {
    "variant": "REFERENCE_V1_scalar_plus_pca16_added", "auc": 0.868019, "c_index": 0.811373,
    "specificity": 0.548544, "recall": 0.9696, "tn": None, "fp": None, "fn": None, "tp": None,
    "per_fold_spec_sd": 0.0577, "note": "prior run -- scalar KEPT, PCA16 embeddings added on top",
}
V2_RAW1024_ADDED = {
    "variant": "REFERENCE_V2_scalar_plus_raw1024_added", "auc": 0.862334, "c_index": 0.806608,
    "specificity": 0.512945, "recall": 0.9820, "tn": None, "fp": None, "fn": None, "tp": None,
    "per_fold_spec_sd": 0.0867, "note": "prior run -- scalar KEPT, raw 1024-dim embeddings added on top",
}


def fit_transforms(spec, embeddings, fit_mask):
    """Fit every data-dependent transform on fit_mask rows ONLY. Called once
    per inner fold (inner-training rows) and once per outer fold for the
    refit (full outer-training rows)."""
    tf = {}
    emb = spec.get("emb")
    if emb is not None and emb[0] == "pca":
        pca = PCA(n_components=emb[1], random_state=42)
        pca.fit(embeddings.loc[fit_mask].values)
        tf["pca"] = pca
    return tf


def apply_transforms(spec, tf, X_clin, embeddings, mask):
    """Feature matrix for rows in `mask`, using transforms fit on training rows."""
    Xm = X_clin.loc[mask].copy()
    kind, k = spec["emb"]
    if kind == "pca":
        Z = tf["pca"].transform(embeddings.loc[mask].values)
        block = pd.DataFrame(Z, columns=[f"pca_{i}" for i in range(k)], index=Xm.index)
    else:
        block = embeddings.loc[mask].copy()
        block.index = Xm.index
    return pd.concat([Xm, block], axis=1)


def train_leakfree_cv(spec, X_clin, embeddings, y, folds, label):
    """Outer 5-fold CV; the boosting-round count per outer fold comes from a
    4-fold inner CV on that fold's TRAINING rows only. The outer refit uses
    exactly that many rounds, with no early stopping and no test data in
    `evals`. PCA is refit on each level's training rows."""
    oof_pred = pd.Series(index=y.index, dtype=float)
    rounds_log = []

    for f in sorted(folds.unique()):
        train_mask = folds != f
        test_mask = folds == f

        inner_labels = sorted(folds[train_mask].unique())
        inner_iters = []
        for g in inner_labels:
            itr = train_mask & (folds != g)
            iva = train_mask & (folds == g)
            tf_in = fit_transforms(spec, embeddings, itr)
            Xi_tr = apply_transforms(spec, tf_in, X_clin, embeddings, itr)
            Xi_va = apply_transforms(spec, tf_in, X_clin, embeddings, iva)
            d_itr = xgb.DMatrix(Xi_tr, label=y[itr], enable_categorical=True)
            d_iva = xgb.DMatrix(Xi_va, label=y[iva], enable_categorical=True)
            m = xgb.train(FIXED_PARAMS, d_itr, num_boost_round=NUM_BOOST_ROUND,
                          early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                          evals=[(d_iva, "inner_val")], verbose_eval=False)
            inner_iters.append(int(m.best_iteration))

        n_rounds = max(1, int(round(float(np.mean(inner_iters)))))

        tf_out = fit_transforms(spec, embeddings, train_mask)
        X_tr = apply_transforms(spec, tf_out, X_clin, embeddings, train_mask)
        X_te = apply_transforms(spec, tf_out, X_clin, embeddings, test_mask)
        d_tr = xgb.DMatrix(X_tr, label=y[train_mask], enable_categorical=True)
        d_te = xgb.DMatrix(X_te, enable_categorical=True)  # no label
        final = xgb.train(FIXED_PARAMS, d_tr, num_boost_round=n_rounds, verbose_eval=False)
        oof_pred[test_mask] = final.predict(d_te)

        rounds_log.append({"variant": label, "outer_fold": int(f), "n_train": int(train_mask.sum()),
                           "n_test": int(test_mask.sum()), "n_feat": int(X_tr.shape[1]),
                           "inner_best_iters": inner_iters, "n_rounds_refit": n_rounds})
        print(f"  [{label}] fold {f}: n_train={train_mask.sum()} n_feat={X_tr.shape[1]} "
              f"inner_best_iters={inner_iters} -> n_rounds={n_rounds}", flush=True)

    assert oof_pred.notna().all()
    return oof_pred, rounds_log


# Replace variants: clinical(116) only as the base, embeddings as the SOLE imaging input.
VARIANTS = [
    {"name": "R1_pca16_replace", "emb": ("pca", 16),
     "note": "clinical + PCA16(embeddings), scalar predicted_los REMOVED"},
    {"name": "R2_raw1024_replace", "emb": ("raw", EMB_DIM),
     "note": "clinical + raw 1024-dim embeddings, scalar predicted_los REMOVED"},
    {"name": "R3_pca8_replace", "emb": ("pca", 8),
     "note": "clinical + PCA8(embeddings), scalar predicted_los REMOVED"},
    {"name": "R3_pca32_replace", "emb": ("pca", 32),
     "note": "clinical + PCA32(embeddings), scalar predicted_los REMOVED"},
    {"name": "R3_pca64_replace", "emb": ("pca", 64),
     "note": "clinical + PCA64(embeddings), scalar predicted_los REMOVED"},
]


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
    print(f"Fixed hyperparameters (same as the added-embeddings run): {FIXED_PARAMS}")
    print(f"num_boost_round<={NUM_BOOST_ROUND}, early_stopping_rounds={EARLY_STOPPING_ROUNDS} "
          f"monitored on the INNER VALIDATION fold; outer refit uses the mean inner best_iteration "
          f"with NO early stopping and NO test data in evals.")

    # Reference rows come from the added-embeddings run.
    add_variant_path = os.path.join(FUSION_DIR, "fusion_variant_comparison.csv")
    assert os.path.exists(add_variant_path), (
        f"{add_variant_path} not found -- run compare_fusion_variants.py first so that the "
        f"V0/V1/V2 reference rows come from the same procedure.")
    add_df = pd.read_csv(add_variant_path).set_index("variant")
    global V0_BASELINE_FIXED_CONFIG, V1_PCA16_ADDED, V2_RAW1024_ADDED
    for key, src, label, note in [
        ("V0", "V0_baseline_fixed_config", "REFERENCE_V0_scalar_only",
         "scalar predicted_los only, no embeddings"),
        ("V1", "V1_embeddings_pca16_added", "REFERENCE_V1_scalar_plus_pca16_added",
         "scalar retained, PCA16 embeddings added on top"),
        ("V2", "V2_embeddings_raw1024_added", "REFERENCE_V2_scalar_plus_raw1024_added",
         "scalar retained, raw 1024-dim embeddings added on top"),
    ]:
        r = add_df.loc[src]
        d = {"variant": label, "auc": float(r["auc"]), "c_index": float(r["c_index"]),
             "specificity": float(r["specificity_at_5day_cutoff"]),
             "recall": float(r["recall_at_5day_cutoff"]),
             "tn": None, "fp": None, "fn": None, "tp": None,
             "per_fold_spec_sd": float(r["per_fold_spec_sd"]), "note": note}
        if key == "V0":
            V0_BASELINE_FIXED_CONFIG = d
        elif key == "V1":
            V1_PCA16_ADDED = d
        else:
            V2_RAW1024_ADDED = d
    print(f"\nReference rows loaded from {os.path.basename(add_variant_path)}: "
          f"V0 AUC={V0_BASELINE_FIXED_CONFIG['auc']:.6f} spec={V0_BASELINE_FIXED_CONFIG['specificity']:.6f}")

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
    all_rounds = []
    oof_store = {}

    for spec in VARIANTS:
        name = spec["name"]
        print("\n" + "=" * 70 + f"\n{name}: {spec['note']}\n" + "=" * 70)
        oof, rounds_log = train_leakfree_cv(spec, X_clin, embeddings, y, folds, name)
        all_rounds.extend(rounds_log)
        oof_store[name] = oof
        results.append(evaluate(name, y, oof, folds, note=spec["note"]))

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

    rounds_df = pd.DataFrame(all_rounds)
    rounds_path = os.path.join(HERE, "replace_variant_rounds.csv")
    rounds_df.to_csv(rounds_path, index=False)
    print(f"Saved {rounds_path}")

    oof_df = pd.DataFrame({"subjectID": df["to_patient_id"].values, "fold": folds.values,
                           "true_los": y.values})
    for name, oof in oof_store.items():
        oof_df[f"oof_{name}"] = oof.values
    oof_path = os.path.join(HERE, "replace_variant_oof_predictions.csv")
    oof_df.to_csv(oof_path, index=False, float_format="%.8f")
    print(f"Saved {oof_path}")

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
    print(f"scalar fusion (V0, spec={V0_BASELINE_FIXED_CONFIG['specificity']:.4f}, "
          f"AUC={V0_BASELINE_FIXED_CONFIG['auc']:.4f}) on specificity or AUC?")
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
