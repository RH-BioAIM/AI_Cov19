"""
Ranks the restricted clinical features by association with length of stay
and plots the top 12 as a bar chart (Figure 6).
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index

# build_integrated_model.py and feature_table.py live in ../integrated/.
HERE = os.path.dirname(os.path.abspath(__file__))
INTEGRATED_DIR = os.path.normpath(os.path.join(HERE, "..", "integrated"))
sys.path.insert(0, INTEGRATED_DIR)
from build_integrated_model import load_shared_cohort, build_feature_matrix, RESTRICTED_DROP_COLS  # noqa: E402
from feature_table import get_dictionary, dict_name_to_alldata_name, readable_name, humanize_binned_flag  # noqa: E402

REVISION_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(REVISION_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
TOP_N = 12


def build_readable_map():
    """alldata_column_name -> human-readable label, via the same TCIA
    dictionary + name-cleaning logic used for feature_table.csv."""
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
                    # For a binary True/False flag, record whether the
                    # more-discriminative level was the flagged (True) state
                    # or the majority (False) state.
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
