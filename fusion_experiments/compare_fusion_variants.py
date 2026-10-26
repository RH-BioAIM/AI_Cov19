"""
Compares four imaging-fusion representations for the restricted
integrated model, with the number of boosting rounds for each outer fold
chosen by an inner cross-validation on that fold's training data only.
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

# Committed integrated model's MODAL nested-CV-selected config (see build_nested_tuned_models.py log)
FIXED_PARAMS = {"objective": "reg:squarederror", "tree_method": "hist", "seed": 42, "nthread": 4,
                "max_depth": 4, "eta": 0.03, "min_child_weight": 5}
NUM_BOOST_ROUND = 5000
EARLY_STOPPING_ROUNDS = 100

# Committed (per-outer-fold nested-CV) numbers, for reference in the report
COMMITTED_AUC = 0.871336
COMMITTED_CIDX = 0.812217
COMMITTED_SPEC_AT_5DAY = 0.556634  # recall ~0.976 at literal cutoff=5


def fit_transforms(spec, embeddings, predicted_los, y, fit_mask):
    """Fit every data- or label-dependent transform on fit_mask rows ONLY.
    Called once per inner fold (fit_mask = inner-training rows) and once per
    outer fold for the refit (fit_mask = full outer-training rows)."""
    tf = {}
    emb = spec.get("emb")
    if emb is not None and emb[0] == "pca":
        pca = PCA(n_components=emb[1], random_state=42)
        pca.fit(embeddings.loc[fit_mask].values)
        tf["pca"] = pca
    if spec.get("prob"):
        lr = LogisticRegression()
        y_bin = (y > LOS_THRESHOLD).astype(int)
        lr.fit(predicted_los.loc[fit_mask].values.reshape(-1, 1), y_bin.loc[fit_mask].values)
        tf["lr"] = lr
    return tf


def apply_transforms(spec, tf, X_clin, X_int, embeddings, predicted_los, mask):
    """Build the feature matrix for the rows in `mask` using transforms that
    were fit elsewhere (on training rows only)."""
    base = X_int if spec["base"] == "int" else X_clin
    Xm = base.loc[mask].copy()
    emb = spec.get("emb")
    if emb is not None:
        kind, k = emb
        if kind == "pca":
            Z = tf["pca"].transform(embeddings.loc[mask].values)
            block = pd.DataFrame(Z, columns=[f"pca_{i}" for i in range(k)], index=Xm.index)
        else:
            block = embeddings.loc[mask].copy()
            block.index = Xm.index
        Xm = pd.concat([Xm, block], axis=1)
    if spec.get("prob"):
        Xm["imaging_prob_severe"] = tf["lr"].predict_proba(
            predicted_los.loc[mask].values.reshape(-1, 1))[:, 1]
    return Xm


def train_leakfree_cv(spec, X_clin, X_int, embeddings, predicted_los, y, folds, label):
    """Outer 5-fold CV where the boosting-round count for each outer fold is
    chosen by a 4-fold inner CV on that fold's TRAINING rows only, then the
    model is refit on the full outer training fold for exactly that many
    rounds with no early stopping and no test data in `evals`.

    PCA / logistic are refit at each level on that level's training rows."""
    oof_pred = pd.Series(index=y.index, dtype=float)
    rounds_log = []

    for f in sorted(folds.unique()):
        train_mask = folds != f
        test_mask = folds == f

        # ---- inner CV: the other four outer-fold labels act as inner folds ----
        inner_labels = sorted(folds[train_mask].unique())
        inner_iters = []
        for g in inner_labels:
            itr = train_mask & (folds != g)
            iva = train_mask & (folds == g)
            tf_in = fit_transforms(spec, embeddings, predicted_los, y, itr)
            Xi_tr = apply_transforms(spec, tf_in, X_clin, X_int, embeddings, predicted_los, itr)
            Xi_va = apply_transforms(spec, tf_in, X_clin, X_int, embeddings, predicted_los, iva)
            d_itr = xgb.DMatrix(Xi_tr, label=y[itr], enable_categorical=True)
            d_iva = xgb.DMatrix(Xi_va, label=y[iva], enable_categorical=True)
            m = xgb.train(FIXED_PARAMS, d_itr, num_boost_round=NUM_BOOST_ROUND,
                          early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                          evals=[(d_iva, "inner_val")], verbose_eval=False)
            inner_iters.append(int(m.best_iteration))

        n_rounds = max(1, int(round(float(np.mean(inner_iters)))))

        # ---- outer refit: transforms and model fit on outer-training rows only ----
        tf_out = fit_transforms(spec, embeddings, predicted_los, y, train_mask)
        X_tr = apply_transforms(spec, tf_out, X_clin, X_int, embeddings, predicted_los, train_mask)
        X_te = apply_transforms(spec, tf_out, X_clin, X_int, embeddings, predicted_los, test_mask)
        d_tr = xgb.DMatrix(X_tr, label=y[train_mask], enable_categorical=True)
        d_te = xgb.DMatrix(X_te, enable_categorical=True)  # no label: test is never scored during training
        final = xgb.train(FIXED_PARAMS, d_tr, num_boost_round=n_rounds, verbose_eval=False)
        oof_pred[test_mask] = final.predict(d_te)

        rounds_log.append({"variant": label, "outer_fold": int(f), "n_train": int(train_mask.sum()),
                           "n_test": int(test_mask.sum()), "n_feat": int(X_tr.shape[1]),
                           "inner_best_iters": inner_iters, "n_rounds_refit": n_rounds})
        print(f"  [{label}] fold {f}: n_train={train_mask.sum()} n_feat={X_tr.shape[1]} "
              f"inner_best_iters={inner_iters} -> n_rounds={n_rounds}", flush=True)

    assert oof_pred.notna().all()
    return oof_pred, rounds_log


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


# Variant specs. "base": which feature matrix to start from -- "int" = 116
# clinical + scalar predicted_los (117), "clin" = 116 clinical only.
# "emb": ("pca", k) or ("raw", 1024) embedding block to append, or None.
# "prob": append the fold-honest logistic P(LOS>5) built from predicted_los.
VARIANTS = [
    {"name": "V0_baseline_fixed_config", "base": "int", "emb": None,
     "desc": "clinical(116) + scalar predicted_los"},
    {"name": "V1_embeddings_pca16_added", "base": "int", "emb": ("pca", 16),
     "desc": "clinical(116) + scalar + PCA16(embeddings)"},
    {"name": "V2_embeddings_raw1024_added", "base": "int", "emb": ("raw", EMB_DIM),
     "desc": "clinical(116) + scalar + raw 1024-dim embeddings"},
    {"name": "V3_probability_instead_of_los", "base": "clin", "emb": None, "prob": True,
     "desc": "clinical(116) + fold-honest P(LOS>5) instead of the raw scalar"},
]


def main():
    print("=" * 70)
    print("Richer imaging-to-clinical fusion -- observation study")
    print("=" * 70)
    print(f"Fixed hyperparameters for all variants (committed model's modal config): {FIXED_PARAMS}")
    print(f"num_boost_round<={NUM_BOOST_ROUND}, early_stopping_rounds={EARLY_STOPPING_ROUNDS} "
          f"monitored on the INNER VALIDATION fold (4-fold inner CV on outer-training rows only); "
          f"outer refit uses the mean inner best_iteration with NO early stopping and NO test data in evals.")

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
    embeddings.index = X_int.index  # align for the boolean-mask .loc[] slicing below
    predicted_los = X_int["predicted_los"]

    results = []
    all_rounds = []
    oof_store = {}

    for spec in VARIANTS:
        name = spec["name"]
        print("\n" + "=" * 70 + f"\n{name}: {spec['desc']}\n" + "=" * 70)
        oof, rounds_log = train_leakfree_cv(spec, X_clin, X_int, embeddings,
                                            predicted_los, y, folds, name)
        all_rounds.extend(rounds_log)
        oof_store[name] = oof
        results.append(evaluate(name, df, folds, y, oof))

    out = pd.DataFrame(results)
    # delta vs the v2 scalar control, which is the procedure-matched reference
    v0_auc = float(out.loc[out["variant"] == "V0_baseline_fixed_config", "auc"].iloc[0])
    v0_spec = float(out.loc[out["variant"] == "V0_baseline_fixed_config",
                            "specificity_at_5day_cutoff"].iloc[0])
    out["delta_auc_vs_v0"] = out["auc"] - v0_auc
    out["delta_spec_vs_v0"] = out["specificity_at_5day_cutoff"] - v0_spec
    out["exceeds_fold_noise_sd0.05_vs_v0"] = out["delta_spec_vs_v0"].abs() > 0.05

    out_path = os.path.join(HERE, "fusion_variant_comparison.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    rounds_df = pd.DataFrame(all_rounds)
    rounds_path = os.path.join(HERE, "fusion_variant_rounds.csv")
    rounds_df.to_csv(rounds_path, index=False)
    print(f"Saved {rounds_path}")

    oof_df = pd.DataFrame({"subjectID": df["to_patient_id"].values, "fold": folds.values,
                           "true_los": y.values})
    for name, oof in oof_store.items():
        oof_df[f"oof_{name}"] = oof.values
    oof_path = os.path.join(HERE, "fusion_variant_oof_predictions.csv")
    oof_df.to_csv(oof_path, index=False, float_format="%.8f")
    print(f"Saved {oof_path}")

    print("\n" + "=" * 70)
    print("SUMMARY -- inner-CV boosting rounds")
    print(f"  procedure-matched reference = V0 (this run): AUC={v0_auc:.6f} spec@5day={v0_spec:.6f}")
    print("  (committed nested-CV model, different procedure: AUC=0.871336, "
          "C-index=0.812217, spec@5day=0.556634)")
    print("=" * 70)
    display_cols = ["variant", "auc", "c_index", "specificity_at_5day_cutoff",
                     "delta_auc_vs_v0", "delta_spec_vs_v0",
                     "exceeds_fold_noise_sd0.05_vs_v0",
                     "delta_auc_vs_committed", "delta_spec_vs_committed"]
    print(out[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
