"""
Runs paired DeLong tests comparing the integrated, clinical-only, and
imaging-only models on the nested-CV-tuned restricted out-of-fold
predictions.

Input: the nested-CV-tuned out-of-fold predictions for each model.
Output: delong_results_nested_tuned.csv.
"""
import os
import pandas as pd
from sklearn.metrics import roc_auc_score

from delong import delong_paired_test

REVISION_DIR = os.path.dirname(os.path.abspath(__file__))
LOS_THRESHOLD = 5


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

    sk_auc_a = roc_auc_score(y_true_bin, score_a)
    sk_auc_b = roc_auc_score(y_true_bin, score_b)
    result = delong_paired_test(y_true_bin, score_a, score_b)
    assert abs(result["auc_one"] - sk_auc_a) < 1e-9
    assert abs(result["auc_two"] - sk_auc_b) < 1e-9

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
    print("=" * 70)
    print("PAIRED DELONG TEST -- nested-CV per-model-tuned restricted models")
    print("=" * 70)

    rows = [
        run_comparison(
            "restricted NESTED-TUNED: integrated vs clinical_only",
            "integrated_restricted_nested_tuned", "oof_integrated_restricted_nested_tuned.csv",
            "clinical_only_restricted_nested_tuned", "oof_clinical_only_restricted_nested_tuned.csv",
        ),
        run_comparison(
            "restricted NESTED-TUNED: integrated vs imaging_only",
            "integrated_restricted_nested_tuned", "oof_integrated_restricted_nested_tuned.csv",
            "imaging_only", "oof_imaging_only.csv",
        ),
    ]

    out = pd.DataFrame(rows)
    path = os.path.join(REVISION_DIR, "delong_results_nested_tuned.csv")
    out.to_csv(path, index=False)

    print("\n" + "=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)
    display_cols = ["comparison", "auc_A", "auc_B", "delta_auc", "ci95_low", "ci95_high", "p_value", "paired", "n_paired"]
    print(out[display_cols].to_string(index=False))
    print(f"\nSaved {path}")

    print("\n" + "=" * 70)
    print("THREE-WAY HEADLINE COMPARISON")
    print("=" * 70)
    orig = {"delta_auc": 0.021167, "p_value": 8.017391e-03, "ci95_low": 0.005519, "ci95_high": 0.036814}
    forced = {"delta_auc": 0.016308, "p_value": 2.772250e-02, "ci95_low": 0.001787, "ci95_high": 0.030828}
    nested = rows[0]
    print(f"(a) ORIGINAL (untuned default, kidney_transplant included):        "
          f"delta_auc={orig['delta_auc']:.4f}, p={orig['p_value']:.4g}, 95% CI [{orig['ci95_low']:.4f}, {orig['ci95_high']:.4f}]")
    print(f"(b) FORCED-SYMMETRIC (both = clinical's tuned config, kt dropped): "
          f"delta_auc={forced['delta_auc']:.4f}, p={forced['p_value']:.4g}, 95% CI [{forced['ci95_low']:.4f}, {forced['ci95_high']:.4f}]")
    print(f"(c) NESTED-TUNED (each model's own optimum, kt dropped):           "
          f"delta_auc={nested['delta_auc']:.4f}, p={nested['p_value']:.4g}, 95% CI [{nested['ci95_low']:.4f}, {nested['ci95_high']:.4f}]")
    print(f"\nStill significant at alpha=0.05: {nested['p_value'] < 0.05}")


if __name__ == "__main__":
    main()
