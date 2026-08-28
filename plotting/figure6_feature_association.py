"""
Figure 6 (feature-association bar chart), regenerated on the RESTRICTED
(triage-time-only) feature set so none of the 12 dropped leakage/consequence
columns (the 11 in build_integrated_model.RESTRICTED_DROP_COLS, including
kidney_transplant) can appear -- the old version headlined invasive_vent_days
and Visit Concept Name, both of which are on that drop list.

--- Methodology note: could not verify the original chart's construction ---
Searched the accessible filesystem (CR/, gitsevdata/covid-severity/,
including its Regression/ and model_explainability/ subfolders) for the
original Figure6.pdf and any script that computes a per-feature association-
with-LOS bar chart. Found neither -- no such script exists anywhere in this
checkout, and no manuscript source file is present to check the caption
against. This is implemented fresh rather than guessed, using this
project's one consistently-used discriminative-association statistic
(lifelines concordance_index, the same metric used for every model-level
result throughout this revision -- imaging, clinical, integrated, DeLong,
mortality) applied per INDIVIDUAL FEATURE against length_of_stay instead of
per model:

  - Numeric features (labs, vitals, binarized threshold flags already 0/1 in
    AllData.csv): concordance_index(length_of_stay, feature_value), computed
    on the subset of patients with a non-missing value for that feature.
  - Categorical features (multi-level flags like htn_v, or nominal fields
    like gender/encounter type): one-hot encoded (excluding the explicit
    "nan"/missing level -- a feature's *missingness pattern* being the
    reported "association" would be uninterpretable on a chart like this),
    concordance_index computed per level, the feature's score is the best
    (most discriminative) level, and that level is reported alongside it.
  - Because concordance_index is direction-agnostic in neither direction by
    default (a feature that decreases with LOS is just as informative as one
    that increases), the reported association is max(C, 1-C) -- i.e.
    discriminative strength regardless of direction, consistent with reading
    a bar chart of "how associated is this feature with LOS" rather than a
    signed correlation. Direction is reported in the underlying CSV for
    anyone who wants it.
  - Features with a single non-missing level (no variance -- e.g.
    covid19_statuses, constant "positive" in this cohort) are dropped from
    ranking (undefined C-index).

Flagging this prominently: if the manuscript's original Figure 6 used a
different statistic (e.g. mutual information, a univariate regression
coefficient, or Spearman correlation), this reproduction will rank features
differently. The instruction was to confirm the original methodology before
recomputing; that confirmation was not possible because no original artifact
exists in this checkout. Recommend cross-checking against whatever produced
the original chart (if it lives outside this repository) before treating the
ranking as final.

Similarly, "keep the numeric-threshold encoding notes from the original
caption" could not be done literally -- the original caption text is not
accessible either. The caption note below is generated fresh from the
column-name encoding convention itself (e.g. "Sodium_above145" -> "Sodium >
145"), using step7_feature_table.py's own humanize_binned_flag() parser, for
whichever bars in the final top-12 are pre-binarized threshold flags.

Pure computation on existing data (AllData.csv via build_integrated_model's
loader) -- no model retraining.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index

from build_integrated_model import load_shared_cohort, build_feature_matrix, RESTRICTED_DROP_COLS
from step7_feature_table import get_dictionary, dict_name_to_alldata_name, readable_name, humanize_binned_flag

REVISION_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(REVISION_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
TOP_N = 12


def build_readable_map():
    """alldata_column_name -> human-readable label, via the same TCIA
    dictionary + name-cleaning logic used for step7_feature_table.csv."""
    tcia_dict = get_dictionary()
    from build_integrated_model import ALLDATA_PATH
    alldata_cols = pd.read_csv(ALLDATA_PATH, nrows=0).columns.tolist()
    mapping = {}
    for _, r in tcia_dict.iterrows():
        dict_name = r["column_name"]
        alldata_name = dict_name_to_alldata_name(dict_name, alldata_cols)
        if alldata_name is not None:
            mapping[alldata_name] = readable_name(dict_name)
    return mapping


def feature_association(X, y):
    rows = []
    for col in X.columns:
        series = X[col]
        if str(series.dtype) == "category":
            levels = [lv for lv in series.cat.categories if lv != "nan"]
            n_present = int((series != "nan").sum())
            # A 2-level (binary) categorical has only one degree of freedom --
            # max(c, 1-c) on either level gives the identical score, so
            # reporting "which level" is not meaningful (it's an artifact of
            # which label happened to be tested first). Only report a level
            # for genuinely 3+-level nominal features, where the best-level
            # choice conveys real information (e.g. which encounter type).
            report_level = len(levels) > 2
            best_c, best_level, best_level_n, best_level_is_true = np.nan, None, None, None
            for lv in levels:
                indicator = (series == lv).astype(int)
                if indicator.nunique() < 2:
                    continue
                try:
                    c = concordance_index(y, indicator)
                except ZeroDivisionError:
                    continue
                score = max(c, 1 - c)
                if np.isnan(best_c) or score > best_c:
                    best_c = score
                    best_level = lv if report_level else None
                    best_level_n = int(indicator.sum())
                    # for a binary True/False-style flag, record whether the
                    # more-discriminative level was the "abnormal"/flagged
                    # state (True) or the majority "normal" state (False) --
                    # several of these turn out to be driven by the majority
                    # class on a skewed flag, not the rare abnormal one, and
                    # that's worth knowing before writing a caption.
                    best_level_is_true = (lv == "True") if not report_level else None
            rows.append({"feature": col, "association": best_c, "level": best_level,
                          "feature_type": "categorical", "n": n_present,
                          "n_at_selected_level": best_level_n,
                          "selected_level_is_flagged_state": best_level_is_true})
        else:
            valid = series.notna()
            if valid.sum() < 10 or series[valid].nunique() < 2:
                rows.append({"feature": col, "association": np.nan, "level": None,
                             "feature_type": "numeric", "n": int(valid.sum()),
                             "n_at_selected_level": None, "selected_level_is_flagged_state": None})
                continue
            c = concordance_index(y[valid], series[valid].astype(float))
            rows.append({"feature": col, "association": max(c, 1 - c), "level": None,
                         "feature_type": "numeric", "n": int(valid.sum()),
                         "n_at_selected_level": None, "selected_level_is_flagged_state": None})
    return pd.DataFrame(rows)


def main():
    df = load_shared_cohort()
    X, y = build_feature_matrix(df, restricted=True, include_imaging_pred=False)
    print(f"Restricted clinical feature matrix: {X.shape[1]} features, n={len(X)} patients")

    dropped_present = [c for c in RESTRICTED_DROP_COLS if c in X.columns]
    assert not dropped_present, f"leakage columns leaked into restricted X: {dropped_present}"
    print(f"Confirmed: none of the {len(RESTRICTED_DROP_COLS)} dropped columns are in X "
          f"({RESTRICTED_DROP_COLS})")

    assoc = feature_association(X, y)
    assoc = assoc.dropna(subset=["association"]).sort_values("association", ascending=False)

    readable = build_readable_map()
    assoc["readable_name"] = assoc["feature"].map(lambda f: readable.get(f, f))

    full_path = os.path.join(REVISION_DIR, "figure6_feature_association_full.csv")
    assoc.to_csv(full_path, index=False)
    print(f"Saved full ranked association table ({len(assoc)} features): {full_path}")

    top = assoc.head(TOP_N).iloc[::-1]  # reverse for horizontal bar chart (largest at top)

    print(f"\nTop {TOP_N} retained features by association with length of stay:")
    for _, r in top.iloc[::-1].iterrows():
        level_note = f" [level: {r['level']}]" if r["level"] else ""
        if pd.notna(r["n_at_selected_level"]):
            direction = ("flagged/abnormal state" if r["selected_level_is_flagged_state"] is True
                          else "majority/normal state" if r["selected_level_is_flagged_state"] is False
                          else "one nominal level")
            prevalence_note = f", driven by n={int(r['n_at_selected_level'])} patients in the {direction}"
        else:
            prevalence_note = ""
        print(f"  {r['readable_name']:<45s} C={r['association']:.4f}{level_note}  (n={r['n']}{prevalence_note})")

    # double-check none of the top 12 are leakage variables (redundant with the
    # X.columns assert above, but explicit at the final plotted set too)
    assert not set(top["feature"]) & set(RESTRICTED_DROP_COLS)

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = top["readable_name"].tolist()
    values = top["association"].values
    bar_positions = range(len(labels))
    ax.barh(bar_positions, values, color="#4472C4", edgecolor="black", linewidth=1)
    ax.set_yticks(bar_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Association with length of stay (concordance index, direction-agnostic)")
    x_max = values.max() * 1.12  # headroom so the C= label on the longest bar isn't clipped
    ax.set_xlim(0.5, x_max)
    for pos, v in zip(bar_positions, values):
        ax.text(v + 0.003, pos, f"C={v:.3f}", va="center", fontsize=10, fontweight="bold")
    ax.set_title(f"Figure 6. Top {TOP_N} triage-time clinical features\nby association with length of stay (restricted feature set)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()

    pdf_path = os.path.join(FIG_DIR, "Figure6.pdf")
    png_path = os.path.join(FIG_DIR, "Figure6.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"\nSaved {pdf_path}")
    print(f"Saved {png_path} (preview)")

    # --- caption note for pre-binarized threshold flags among the top 12 ---
    print("\nEncoding notes for threshold-binned features in the final top-12 (for caption):")
    any_note = False
    for _, r in top.iloc[::-1].iterrows():
        raw = r["feature"]
        if any(tok in raw for tok in ["above", "below", "between", "over", "under"]) or "_" in raw and any(ch.isdigit() for ch in raw):
            note = humanize_binned_flag(raw)
            if note != raw:
                print(f"  {r['readable_name']}: encoded as '{raw}' -> {note}")
                any_note = True
    if not any_note:
        print("  (none of the top 12 are pre-binarized numeric-threshold flags)")


if __name__ == "__main__":
    main()
