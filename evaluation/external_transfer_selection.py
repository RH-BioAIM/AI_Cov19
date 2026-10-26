"""
Selects the imaging fold model and the predicted-length-of-stay cutoff for
each external dataset using that dataset's own labels, from the saved
per-image predictions. Writes the selected fold and cutoff used by
external_transfer_metrics.py.
"""
import os

import numpy as np
import pandas as pd

REVISION_DIR = os.path.dirname(os.path.abspath(__file__))
MIDRC_PRED_PATH = os.path.join(REVISION_DIR, "midrc_predictions_all_folds.csv")
COVID19_PRED_PATH = os.path.join(REVISION_DIR, "covid19_radiography_predictions_all_folds.csv")

FOLD_COLS = [f"pred_fold{i}" for i in range(1, 6)]
MIDRC_PROVISIONAL_CUTOFF = 7      # used only to rank the folds in stage one
MIDRC_SWEEP = range(1, 15)        # integer cutoffs, days
COVID19_GRID_STEP = 0.1           # days


def threshold_metrics(y_true, y_score, cutoff, inclusive):
    y_pred = ((y_score >= cutoff) if inclusive else (y_score > cutoff)).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    return {
        "accuracy": (tp + tn) / len(y_true), "precision": precision, "recall": recall,
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"), "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def macro_f1(y_true, y_score, cutoff, inclusive=True):
    """F1 averaged over both severity classes, the criterion used for
    MIDRC-RICORD-1c."""
    y_pred = ((y_score >= cutoff) if inclusive else (y_score > cutoff)).astype(int)
    scores = []
    for cls in (0, 1):
        tp = int(((y_pred == cls) & (y_true == cls)).sum())
        fp = int(((y_pred == cls) & (y_true != cls)).sum())
        fn = int(((y_pred != cls) & (y_true == cls)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return float(np.mean(scores))


def select_midrc(frame):
    """Rank the five folds by macro-F1 at a provisional cutoff, then sweep
    integer cutoffs on the leading fold, again maximising macro-F1."""
    y_true = frame["label"].values.astype(int)
    fold_scores = {}
    for i, col in enumerate(FOLD_COLS, start=1):
        fold_scores[i] = macro_f1(y_true, frame[col].values.astype(float),
                                  MIDRC_PROVISIONAL_CUTOFF, inclusive=True)
    best_fold = max(fold_scores, key=fold_scores.get)

    y_score = frame[f"pred_fold{best_fold}"].values.astype(float)
    sweep = {t: macro_f1(y_true, y_score, t, inclusive=True) for t in MIDRC_SWEEP}
    best_cutoff = max(sweep, key=sweep.get)
    return {
        "fold_scores": fold_scores, "selected_fold": int(best_fold),
        "cutoff_sweep": sweep, "selected_cutoff": float(best_cutoff),
        "inclusive": True,
        "metrics": threshold_metrics(y_true, y_score, best_cutoff, inclusive=True),
    }


def select_covid19(frame):
    """Sweep a fine cutoff grid per fold to maximise accuracy, then take the
    fold whose best accuracy is highest."""
    y_true = frame["label"].values.astype(int)
    per_fold = {}
    for i, col in enumerate(FOLD_COLS, start=1):
        y_score = frame[col].values.astype(float)
        best_cutoff, best_accuracy = None, -1.0
        for cutoff in np.arange(0, float(np.max(y_score)), COVID19_GRID_STEP):
            accuracy = threshold_metrics(y_true, y_score, cutoff, inclusive=False)["accuracy"]
            if accuracy > best_accuracy:
                best_accuracy, best_cutoff = accuracy, float(cutoff)
        per_fold[i] = {"cutoff": round(best_cutoff, 2), "accuracy": best_accuracy}
    best_fold = max(per_fold, key=lambda k: per_fold[k]["accuracy"])

    y_score = frame[f"pred_fold{best_fold}"].values.astype(float)
    cutoff = per_fold[best_fold]["cutoff"]
    return {
        "fold_scores": {k: v["accuracy"] for k, v in per_fold.items()},
        "per_fold_cutoff": {k: v["cutoff"] for k, v in per_fold.items()},
        "selected_fold": int(best_fold), "selected_cutoff": float(cutoff),
        "inclusive": False,
        "metrics": threshold_metrics(y_true, y_score, cutoff, inclusive=False),
    }


def main():
    selections = {
        "MIDRC-RICORD-1c": select_midrc(pd.read_csv(MIDRC_PRED_PATH)),
        "COVID-19 Radiography Database": select_covid19(pd.read_csv(COVID19_PRED_PATH)),
    }

    rows = []
    for dataset, result in selections.items():
        rows.append({
            "dataset": dataset, "selected_fold": result["selected_fold"],
            "selected_cutoff_days": result["selected_cutoff"],
            "cutoff_inclusive": result["inclusive"], **result["metrics"],
        })
        print(f"\n{dataset}")
        print(f"  per-fold selection score: "
              + ", ".join(f"fold {k} {v:.4f}" for k, v in result["fold_scores"].items()))
        if "per_fold_cutoff" in result:
            print(f"  per-fold cutoff:          "
                  + ", ".join(f"fold {k} {v}" for k, v in result["per_fold_cutoff"].items()))
        print(f"  selected fold {result['selected_fold']}, "
              f"cutoff {result['selected_cutoff']:.1f} days")
        m = result["metrics"]
        print(f"  accuracy={m['accuracy']:.4f} precision={m['precision']:.4f} "
              f"recall={m['recall']:.4f} specificity={m['specificity']:.4f} f1={m['f1']:.4f}")

    out = pd.DataFrame(rows)
    path = os.path.join(REVISION_DIR, "external_transfer_selection.csv")
    out.to_csv(path, index=False)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
