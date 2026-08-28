"""
Mortality-endpoint re-check on the nested-CV per-model-tuned restricted
integrated model (Option C). Same as step6_mortality_final.py, re-pointed at
oof_integrated_restricted_nested_tuned.csv. Re-verifies last_status is absent
from the restricted feature matrix (unchanged from the _final run --
kidney_transplant still dropped, 117 columns -- but re-checked explicitly
per the task's requirement rather than assumed carried over).
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

from build_integrated_model import ALLDATA_PATH, load_shared_cohort, build_feature_matrix
from delong import delong_auc_ci

REVISION_DIR = os.path.dirname(os.path.abspath(__file__))
OOF_PATH = os.path.join(REVISION_DIR, "oof_integrated_restricted_nested_tuned.csv")
FIXED_YOUDEN_THRESHOLD = 6.712525844573975


def main():
    oof = pd.read_csv(OOF_PATH)
    assert len(oof) == 1341

    print("=" * 70)
    print("MORTALITY-ENDPOINT CHECK -- nested-CV per-model-tuned integrated model (Option C)")
    print("=" * 70)

    df = load_shared_cohort()
    X_restricted, _ = build_feature_matrix(df, restricted=True, include_imaging_pred=True)
    assert "last_status" not in X_restricted.columns, \
        "last_status found in restricted feature matrix -- outcome leaking into predictors!"
    assert "kidney_transplant" not in X_restricted.columns
    print(f"CHECK: 'last_status' confirmed ABSENT from the restricted feature matrix "
          f"({X_restricted.shape[1]} columns, expected 117).")
    assert X_restricted.shape[1] == 117

    clinical = pd.read_csv(ALLDATA_PATH)[["to_patient_id", "last_status"]]
    merged = oof.merge(clinical, left_on="subjectID", right_on="to_patient_id", how="left")
    assert merged["last_status"].notna().all()
    assert len(merged) == len(oof)

    counts = merged["last_status"].value_counts()
    n_deceased = int(counts.get("deceased", 0))
    n_discharged = int(counts.get("discharged", 0))
    print(f"\nMatched set: n={len(merged)}  deceased={n_deceased}  discharged={n_discharged}")
    assert n_deceased + n_discharged == len(merged)

    y_mortality = (merged["last_status"] == "deceased").astype(int).values
    y_score = merged["predicted_los"].values

    mort_auc = delong_auc_ci(y_mortality, y_score)
    print(f"\nAUC of predicted severity score for discriminating mortality directly: "
          f"{mort_auc['auc']:.4f} (95% CI [{mort_auc['ci95_low']:.4f}, {mort_auc['ci95_high']:.4f}])")

    thresholds = {"youden_optimal_6.71d": FIXED_YOUDEN_THRESHOLD, "fixed_5day": 5.0}
    rows = []
    for label, thr in thresholds.items():
        high_risk = y_score > thr
        low_risk = ~high_risk
        n_high, n_low = high_risk.sum(), low_risk.sum()
        deaths_high = int(y_mortality[high_risk].sum())
        deaths_low = int(y_mortality[low_risk].sum())
        mort_rate_high = deaths_high / n_high if n_high > 0 else float("nan")
        mort_rate_low = deaths_low / n_low if n_low > 0 else float("nan")

        table = np.array([[deaths_high, n_high - deaths_high],
                           [deaths_low, n_low - deaths_low]])
        chi2, chi2_p, _, _ = chi2_contingency(table)
        _, fisher_p = fisher_exact(table)

        rows.append({
            "threshold_type": label, "threshold_value": thr,
            "n_high_risk": int(n_high), "deaths_high_risk": deaths_high, "mortality_rate_high_risk": mort_rate_high,
            "n_low_risk": int(n_low), "deaths_low_risk": deaths_low, "mortality_rate_low_risk": mort_rate_low,
            "chi2_p_value": chi2_p, "fisher_exact_p_value": fisher_p,
        })
        print(f"\n[{label}, threshold={thr:.3f}d]")
        print(f"  high-risk (n={n_high}): {deaths_high} deaths, mortality rate = {mort_rate_high:.4f}")
        print(f"  low-risk  (n={n_low}): {deaths_low} deaths, mortality rate = {mort_rate_low:.4f}")
        print(f"  chi-square p = {chi2_p:.6g}, Fisher's exact p = {fisher_p:.6g}")

    out = pd.DataFrame(rows)
    out["mortality_auc"] = mort_auc["auc"]
    out["mortality_auc_ci95_low"] = mort_auc["ci95_low"]
    out["mortality_auc_ci95_high"] = mort_auc["ci95_high"]
    out["n_deceased_matched_set"] = n_deceased
    out["n_discharged_matched_set"] = n_discharged
    path = os.path.join(REVISION_DIR, "step6_mortality_endpoint_nested_tuned.csv")
    out.to_csv(path, index=False)
    print(f"\nSaved {path}")

    print("\n" + "=" * 70)
    print("THREE-WAY -- mortality AUC")
    print("=" * 70)
    print(f"(a) ORIGINAL (untuned default, kt included):        AUC = 0.8085 (95% CI [0.7794, 0.8375])")
    print(f"(b) FORCED-SYMMETRIC (kt dropped, both = clinical config): AUC = 0.8266 (95% CI [0.8004, 0.8528])")
    print(f"(c) NESTED-TUNED (kt dropped, per-model optimum):    AUC = {mort_auc['auc']:.4f} "
          f"(95% CI [{mort_auc['ci95_low']:.4f}, {mort_auc['ci95_high']:.4f}])")


if __name__ == "__main__":
    main()
