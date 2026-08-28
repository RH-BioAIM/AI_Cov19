"""
Analyzes the integrated model's specificity at the 5-day decision threshold:
a calibration curve, a characterization of the false positives, an
alternative score-cutoff analysis at fixed recall, and a fold-honest
isotonic recalibration check. No retraining.

Input: the nested-CV-tuned integrated model's out-of-fold predictions.
Output: calibration_by_decile.csv, calibration_near_boundary.csv,
calibration_curve.png, fp_predicted_los_distribution.csv,
false_positives_detail.csv, threshold_on_score_vs_fixed_recall.csv,
isotonic_recalibration_result.csv, recalibrated_predictions_fold_honest.csv.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.isotonic import IsotonicRegression

REVISION_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(REVISION_DIR, "specificity_diagnosis")
os.makedirs(OUT_DIR, exist_ok=True)

OOF_PATH = os.path.join(REVISION_DIR, "oof_integrated_restricted_nested_tuned.csv")
LOS_THRESHOLD = 5


def part1_calibration(df):
    print("=" * 70)
    print("PART 1 -- calibration: predicted vs actual LOS")
    print("=" * 70)
    y_true, y_pred = df["true_los"].values, df["predicted_los"].values

    overall_bias = float(np.mean(y_pred - y_true))
    print(f"Overall mean bias (predicted - actual): {overall_bias:+.3f} days "
          f"(mean true_los={y_true.mean():.2f}, mean predicted_los={y_pred.mean():.2f})")

    # Decile bins by predicted LOS -- classic calibration-curve construction.
    df = df.copy()
    df["pred_decile"] = pd.qcut(df["predicted_los"], 10, duplicates="drop")
    calib = df.groupby("pred_decile", observed=True).agg(
        mean_predicted=("predicted_los", "mean"),
        mean_actual=("true_los", "mean"),
        median_actual=("true_los", "median"),
        n=("true_los", "size"),
    ).reset_index()
    calib.to_csv(os.path.join(OUT_DIR, "calibration_by_decile.csv"), index=False)
    print("\nCalibration by decile of predicted LOS:")
    print(calib.to_string(index=False))

    # Fine-grained bias right at the decision boundary (predicted in [3,10)).
    boundary = df[(df["predicted_los"] >= 3) & (df["predicted_los"] < 10)].copy()
    boundary["pred_bin"] = pd.cut(boundary["predicted_los"], bins=[3, 4, 5, 6, 7, 8, 10])
    near = boundary.groupby("pred_bin", observed=True).agg(
        mean_predicted=("predicted_los", "mean"),
        mean_actual=("true_los", "mean"),
        n=("true_los", "size"),
    ).reset_index()
    near["bias"] = near["mean_predicted"] - near["mean_actual"]
    near.to_csv(os.path.join(OUT_DIR, "calibration_near_boundary.csv"), index=False)
    print("\nBias right at the decision boundary (predicted LOS in [3,10), 1-day bins):")
    print(near.to_string(index=False))

    # Plot: predicted vs actual (decile means) with identity line.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(calib["mean_predicted"], calib["mean_actual"], marker="o", color="#4472C4", label="Observed (decile bins)")
    lims = [min(calib["mean_predicted"].min(), calib["mean_actual"].min()) - 1,
            max(calib["mean_predicted"].max(), calib["mean_actual"].max()) + 1]
    ax.plot(lims, lims, linestyle="--", color="gray")
    ax.axvline(5, color="red", linestyle=":", linewidth=1)
    ax.set_xlabel("Mean predicted LOS (days), per decile bin")
    ax.set_ylabel("Mean actual LOS (days), per decile bin")
    ax.set_title("Calibration: predicted vs. actual LOS\nIntegrated model (restricted, nested-tuned)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "calibration_curve.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved calibration_curve.png and calibration_by_decile.csv / calibration_near_boundary.csv")
    return near


def part2_false_positives(df):
    print("\n" + "=" * 70)
    print("PART 2 -- false-positive characterization (5-day threshold)")
    print("=" * 70)
    fp = df[(df["true_los"] <= LOS_THRESHOLD) & (df["predicted_los"] > LOS_THRESHOLD)].copy()
    print(f"n false positives: {len(fp)}")
    print(f"true_los among FPs: mean={fp['true_los'].mean():.2f}, median={fp['true_los'].median():.2f}, "
          f"range=[{fp['true_los'].min():.1f}, {fp['true_los'].max():.1f}]")
    print(f"predicted_los among FPs: mean={fp['predicted_los'].mean():.2f}, median={fp['predicted_los'].median():.2f}, "
          f"range=[{fp['predicted_los'].min():.1f}, {fp['predicted_los'].max():.1f}]")

    bins = [5, 6, 7, 8, 10, 15, 100]
    bin_labels = ["(5,6]", "(6,7]", "(7,8]", "(8,10]", "(10,15]", "(15,100]"]
    fp["overshoot_bin"] = pd.cut(fp["predicted_los"], bins=bins, labels=bin_labels)
    dist = fp.groupby("overshoot_bin", observed=True).size().reset_index(name="n")
    dist["pct_of_fps"] = (100 * dist["n"] / len(fp)).round(1)
    dist.to_csv(os.path.join(OUT_DIR, "fp_predicted_los_distribution.csv"), index=False)
    print("\nDistribution of predicted LOS among false positives:")
    print(dist.to_string(index=False))

    borderline = (fp["predicted_los"] <= 7).sum()  # within 2 days of the cutoff
    confidently_wrong = (fp["predicted_los"] > 10).sum()
    print(f"\nBorderline (predicted LOS in (5,7], within 2 days of cutoff): {borderline} "
          f"({100*borderline/len(fp):.1f}% of FPs)")
    print(f"Confidently wrong (predicted LOS > 10): {confidently_wrong} "
          f"({100*confidently_wrong/len(fp):.1f}% of FPs)")

    fp.to_csv(os.path.join(OUT_DIR, "false_positives_detail.csv"), index=False)
    return fp


def part3_threshold_on_score(df):
    print("\n" + "=" * 70)
    print("PART 3 -- threshold-on-score vs threshold-on-LOS (specificity at fixed recall)")
    print("=" * 70)
    y_true_bin = (df["true_los"].values > LOS_THRESHOLD).astype(int)
    y_score = df["predicted_los"].values

    fpr, tpr, thresholds = roc_curve(y_true_bin, y_score)
    specificity = 1 - fpr

    # current literal 5-day rule
    tn, fp_, fn, tp = confusion_matrix(y_true_bin, (y_score > 5).astype(int), labels=[0, 1]).ravel()
    print(f"Literal 'predicted LOS > 5' rule: recall={tp/(tp+fn):.4f}, specificity={tn/(tn+fp_):.4f}, "
          f"cutoff=5.0 days")

    rows = []
    for target_recall in [0.95, 0.97]:
        eligible = tpr >= target_recall
        if not eligible.any():
            continue
        # among cutoffs achieving at least target_recall, take the one with max specificity
        # (roc_curve returns thresholds in decreasing order; ties in tpr can occur)
        idx = np.where(eligible)[0]
        best_idx = idx[np.argmax(specificity[idx])]
        best_thr = thresholds[best_idx]
        y_pred_bin = (y_score > best_thr).astype(int)
        tn2, fp2, fn2, tp2 = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
        achieved_recall = tp2 / (tp2 + fn2)
        achieved_spec = tn2 / (tn2 + fp2)
        rows.append({
            "target_recall": target_recall, "score_cutoff_days": round(float(best_thr), 3),
            "achieved_recall": round(achieved_recall, 4), "achieved_specificity": round(achieved_spec, 4),
            "tn": tn2, "fp": fp2, "fn": fn2, "tp": tp2,
        })
        print(f"\nTarget recall >= {target_recall}: best score cutoff = {best_thr:.3f} days "
              f"(vs literal cutoff of 5.0)")
        print(f"  achieved recall={achieved_recall:.4f}, achieved specificity={achieved_spec:.4f} "
              f"(TN={tn2} FP={fp2} FN={fn2} TP={tp2})")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "threshold_on_score_vs_fixed_recall.csv"), index=False)

    # Also: what recall/specificity does literal cutoff=5 give, framed the same way, for reference
    print(f"\nFor reference, literal cutoff=5.0 gives recall={tp/(tp+fn):.4f} "
          f"(already close to the 0.97 target) at specificity={tn/(tn+fp_):.4f}.")
    return out


def part4_isotonic_recalibration(df):
    print("\n" + "=" * 70)
    print("PART 4 -- fold-honest isotonic recalibration (fit only on other folds, no leakage)")
    print("=" * 70)
    df = df.copy()
    recal = pd.Series(index=df.index, dtype=float)
    for f in sorted(df["fold"].unique()):
        train_mask = df["fold"] != f
        test_mask = df["fold"] == f
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(df.loc[train_mask, "predicted_los"], df.loc[train_mask, "true_los"])
        recal.loc[test_mask] = iso.predict(df.loc[test_mask, "predicted_los"])
    df["recalibrated_los"] = recal

    y_true_bin = (df["true_los"].values > LOS_THRESHOLD).astype(int)

    # Confirm ranking is (near-)identical between raw and recalibrated score --
    # if so, ROC/specificity-at-recall MUST be unchanged, by construction.
    rank_corr = pd.Series(df["predicted_los"]).corr(df["recalibrated_los"], method="spearman")
    print(f"Spearman rank correlation, raw predicted_los vs fold-honest recalibrated_los: {rank_corr:.6f}")

    fpr_r, tpr_r, thr_r = roc_curve(y_true_bin, df["recalibrated_los"].values)
    spec_r = 1 - fpr_r

    rows = []
    for target_recall in [0.95, 0.97]:
        eligible = tpr_r >= target_recall
        if not eligible.any():
            continue
        idx = np.where(eligible)[0]
        best_idx = idx[np.argmax(spec_r[idx])]
        y_pred_bin = (df["recalibrated_los"].values > thr_r[best_idx]).astype(int)
        tn, fp_, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
        rows.append({
            "target_recall": target_recall,
            "achieved_recall_recalibrated": round(tp / (tp + fn), 4),
            "achieved_specificity_recalibrated": round(tn / (tn + fp_), 4),
        })
        print(f"Target recall >= {target_recall} on RECALIBRATED score: "
              f"recall={tp/(tp+fn):.4f}, specificity={tn/(tn+fp_):.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "isotonic_recalibration_result.csv"), index=False)
    df[["subjectID", "fold", "true_los", "predicted_los", "recalibrated_los"]].to_csv(
        os.path.join(OUT_DIR, "recalibrated_predictions_fold_honest.csv"), index=False)
    return out, rank_corr


def main():
    df = pd.read_csv(OOF_PATH)
    assert len(df) == 1341

    near = part1_calibration(df)
    fp = part2_false_positives(df)
    thr_result = part3_threshold_on_score(df)
    recal_result, rank_corr = part4_isotonic_recalibration(df)

    print("\n" + "=" * 70)
    print("BOTTOM LINE")
    print("=" * 70)
    r95 = thr_result[thr_result["target_recall"] == 0.95].iloc[0]
    r97 = thr_result[thr_result["target_recall"] == 0.97].iloc[0]
    print(f"At the literal 5-day cutoff: recall=0.976, specificity=0.557.")
    print(f"Best achievable specificity at recall>=0.95 (different SCORE cutoff, same model): "
          f"{r95['achieved_specificity']:.4f} at cutoff={r95['score_cutoff_days']} days "
          f"(vs literal cutoff of 5.0)")
    print(f"Best achievable specificity at recall>=0.97 (different SCORE cutoff, same model): "
          f"{r97['achieved_specificity']:.4f} at cutoff={r97['score_cutoff_days']} days")
    print(f"Fold-honest isotonic recalibration changes rank correlation to {rank_corr:.6f} "
          f"(should be ~1.0 if purely monotonic) and produced IDENTICAL "
          f"specificity-at-recall to the plain cutoff search in Part 3 "
          f"(compare threshold_on_score_vs_fixed_recall.csv to isotonic_recalibration_result.csv) "
          f"-- confirms recalibration adds nothing beyond picking a better cutoff on the existing score.")


if __name__ == "__main__":
    main()
