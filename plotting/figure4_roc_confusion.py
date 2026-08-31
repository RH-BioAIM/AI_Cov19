"""
Generates Figure 4: an ROC curve and confusion matrix for the restricted
integrated model.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

REVISION_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(REVISION_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

OOF_PATH = os.path.join(REVISION_DIR, "oof_integrated_restricted_nested_tuned.csv")
LOS_THRESHOLD = 5


def main():
    df = pd.read_csv(OOF_PATH)
    assert len(df) == 1341
    y_true_los = df["true_los"].values
    y_score = df["predicted_los"].values
    y_true_bin = (y_true_los > LOS_THRESHOLD).astype(int)
    y_pred_bin = (y_score > LOS_THRESHOLD).astype(int)

    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    auc = roc_auc_score(y_true_bin, y_score)

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()

    print(f"Integrated (restricted, nested-tuned) AUC: {auc:.6f}  (rounds to {auc:.3f})")
    print(f"Confusion matrix @ 5-day threshold: TN={tn} FP={fp} FN={fn} TP={tp}  (n={len(df)})")
    print(f"  accuracy    = {(tn+tp)/len(df):.4f}")
    print(f"  sensitivity = {tp/(tp+fn):.4f}")
    print(f"  specificity = {tn/(tn+fp):.4f}")

    fig, (ax_roc, ax_cm) = plt.subplots(1, 2, figsize=(11, 5))

    # --- panel (a): ROC curve ---
    ax_roc.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"Integrated (restricted) ROC (AUC = {auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")  # reference diagonal, no legend entry
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("(a) ROC curve\nIntegrated model, restricted feature set")
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.grid(True, alpha=0.3)
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])

    # --- panel (b): confusion matrix ---
    cm = np.array([[tn, fp], [fn, tp]])
    im = ax_cm.imshow(cm, cmap="Blues")
    labels = ["Not severe\n(LOS ≤ 5d)", "Severe\n(LOS > 5d)"]
    ax_cm.set_xticks([0, 1]); ax_cm.set_xticklabels(labels)
    ax_cm.set_yticks([0, 1]); ax_cm.set_yticklabels(labels)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("(b) Confusion matrix\n5-day decision threshold")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black", fontsize=14)
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    fig.suptitle("Figure 4. Restricted integrated model performance (triage-time features only)", y=1.02)
    fig.tight_layout()

    pdf_path = os.path.join(FIG_DIR, "Figure4.pdf")
    png_path = os.path.join(FIG_DIR, "Figure4.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"\nSaved {pdf_path}")
    print(f"Saved {png_path} (preview)")


if __name__ == "__main__":
    main()
