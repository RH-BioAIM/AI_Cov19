"""
Tests whether feathering the imaging preprocessing boundary changes the
model's predictions and Grad-CAM lung localization.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
# swin_explain.py, lung_segmentation.py, and preprocess.py are direct
# siblings of this file in explainability/.
sys.path.insert(0, HERE)

from swin_explain import (  # noqa: E402
    VitRegressor, apply_val_normalize, StageGradCAM, resize_cam, make_overlay, label,
    rectangular_lung_roi, corner_regions, mass_fraction,
)
from lung_segmentation import lung_mask_224  # noqa: E402
from preprocess import (
    preprocess_image_hard, preprocess_image_soft, boundary_band,
    boundary_gradient_stats, FEATHER_SIGMA_224,
)

OUT_DIR = os.path.join(HERE, "overlays")
DIAG_DIR = os.path.join(HERE, "diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DIAG_DIR, exist_ok=True)

FOLD = 1
CHECKPOINT_PATH = f"/path/to/checkpoints/model_fold_{FOLD}_epoch_25.pth"
PREDICTIONS_PATH = f"/path/to/checkpoints/predictions_fold_{FOLD}.csv"
PATIENT_DICT_PATH = "/path/to/data/patient_dict.csv"
N_CASES = 10
STAGE_IDX = 1  # 28x28 tokens -- the stage established as artifact-free-enough

# Baseline numbers for the head-to-head report.
BASELINE_SEG_MASS_MEAN = 0.326
BASELINE_SEG_AREA_MEAN = 0.279
BASELINE_CORNER_MASS_MEAN = 0.066
PORTABLE_FILM_CASE = "A751083"  # flagged as tracking tubing/grid markers

# Guardrail thresholds for the prediction-consistency gate (Phase A).
MIN_PEARSON_R = 0.8
MAX_MEAN_ABS_PCT_DIFF = 0.30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_representative_cases():
    preds = pd.read_csv(PREDICTIONS_PATH)
    patient_dict = pd.read_csv(PATIENT_DICT_PATH)[["subjectID", "CR_image_paths"]]
    df = preds.merge(patient_dict, on="subjectID", how="inner")
    df = df.sort_values("true_los").reset_index(drop=True)
    idx = np.linspace(0, len(df) - 1, N_CASES).round().astype(int)
    idx = sorted(set(idx.tolist()))
    cases = df.iloc[idx].reset_index(drop=True)
    cases["fold"] = FOLD
    return cases


def load_model():
    print(f"Loading VitRegressor checkpoint (read-only): {CHECKPOINT_PATH}")
    model = VitRegressor(pretrained=False).to(device)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def phase_a_prediction_consistency(model, cases):
    """The guardrail. Returns (df, go: bool, pearson_r, mean_abs_pct_diff)."""
    rows = []
    case_data = {}
    for _, row in cases.iterrows():
        subj, path = row["subjectID"], row["CR_image_paths"]
        tensor_hard, gray_hard, mask_hard = preprocess_image_hard(path)
        tensor_soft, gray_soft, mask_soft = preprocess_image_soft(path)
        if tensor_hard is None or tensor_soft is None:
            print(f"  SKIP {subj}: preprocessing failed")
            continue

        with torch.no_grad():
            pred_hard = model(apply_val_normalize(tensor_hard).unsqueeze(0).to(device)).item()
            pred_soft = model(apply_val_normalize(tensor_soft).unsqueeze(0).to(device)).item()

        abs_diff = abs(pred_soft - pred_hard)
        pct_diff = abs_diff / max(abs(pred_hard), 1e-6)
        rows.append({
            "subjectID": subj,
            "true_los": row["true_los"],
            "predicted_los_fold1_csv": row["predicted_los"],
            "pred_hard_preproc": pred_hard,
            "pred_soft_preproc": pred_soft,
            "abs_diff": abs_diff,
            "pct_diff": pct_diff,
        })
        case_data[subj] = {
            "gray_hard": gray_hard, "mask_hard": mask_hard,
            "gray_soft": gray_soft, "mask_soft": mask_soft,
            "true_los": row["true_los"], "predicted_los": row["predicted_los"],
        }
        print(f"  {subj}: true_los={row['true_los']:.0f} "
              f"pred_hard={pred_hard:.2f} pred_soft={pred_soft:.2f} "
              f"abs_diff={abs_diff:.2f} pct_diff={pct_diff:.1%}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DIAG_DIR, "phase_a_prediction_consistency.csv"), index=False)

    # Sanity: pred_hard here should match the imaging model's predictions_fold_1.csv
    # almost exactly (same preprocessing, same model) -- confirms nothing else broke.
    csv_vs_recomputed_diff = (df["predicted_los_fold1_csv"] - df["pred_hard_preproc"]).abs()

    r, _ = pearsonr(df["pred_hard_preproc"], df["pred_soft_preproc"])
    mean_abs_pct = df["pct_diff"].mean()
    go = (r >= MIN_PEARSON_R) and (mean_abs_pct <= MAX_MEAN_ABS_PCT_DIFF)

    summary = "\n".join([
        f"Phase A: train/inference mismatch guardrail (n={len(df)} cases)",
        f"pred_hard_preproc vs predictions_fold_1.csv (should match, same pipeline): "
        f"max abs diff = {csv_vs_recomputed_diff.max():.4f}",
        f"Pearson r(pred_hard, pred_soft) across cases: {r:.3f}  (threshold >= {MIN_PEARSON_R})",
        f"Mean |pct diff| between pred_hard and pred_soft: {mean_abs_pct:.1%}  "
        f"(threshold <= {MAX_MEAN_ABS_PCT_DIFF:.0%})",
        f"Per-case pct diff range: [{df['pct_diff'].min():.1%}, {df['pct_diff'].max():.1%}]",
        "",
        f"DECISION: {'GO -- predictions stable under corrected preprocessing, proceeding to Phase B.' if go else 'STOP -- predictions diverge too much under corrected preprocessing; corrected-preprocessing CAMs would be confounded with out-of-distribution model behavior, not interpretable. Not proceeding to Grad-CAM generation.'}",
    ])
    print("\n" + summary)
    with open(os.path.join(DIAG_DIR, "phase_a_prediction_consistency.txt"), "w") as f:
        f.write(summary + "\n")

    return df, go, case_data


def boundary_softening_check(case_data):
    rows = []
    for subj, d in case_data.items():
        band = boundary_band(d["mask_hard"])
        stats_hard = boundary_gradient_stats(d["gray_hard"], band)
        stats_soft = boundary_gradient_stats(d["gray_soft"], band)
        rows.append({
            "subjectID": subj,
            "band_pixels": int(band.sum()),
            **{f"hard_{k}": v for k, v in stats_hard.items()},
            **{f"soft_{k}": v for k, v in stats_soft.items()},
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DIAG_DIR, "boundary_gradient_check.csv"), index=False)

    mean_hard_mean = df["hard_grad_band_mean"].mean()
    mean_soft_mean = df["soft_grad_band_mean"].mean()
    mean_hard_p90 = df["hard_grad_band_p90"].mean()
    mean_soft_p90 = df["soft_grad_band_p90"].mean()
    summary = "\n".join([
        f"Boundary-softening check (feather sigma={FEATHER_SIGMA_224}px @ 224x224, n={len(df)} cases)",
        f"In-band Sobel-gradient mean:  hard={mean_hard_mean:.1f} -> soft={mean_soft_mean:.1f} "
        f"({(1 - mean_soft_mean / mean_hard_mean):.0%} reduction)",
        f"In-band Sobel-gradient p90:   hard={mean_hard_p90:.1f} -> soft={mean_soft_p90:.1f} "
        f"({(1 - mean_soft_p90 / mean_hard_p90):.0%} reduction)",
    ])
    print("\n" + summary)
    with open(os.path.join(DIAG_DIR, "boundary_gradient_check.txt"), "w") as f:
        f.write(summary + "\n")
    return df


def main():
    model = load_model()
    cases = select_representative_cases()
    print(f"Selected {len(cases)} cases spanning LOS range "
          f"[{cases['true_los'].min()}, {cases['true_los'].max()}]:")
    print(cases[["subjectID", "true_los", "predicted_los"]].to_string(index=False))

    print("\n--- Phase A: prediction-consistency guardrail ---")
    pred_df, go, case_data = phase_a_prediction_consistency(model, cases)

    print("\n--- Boundary-softening check (does the feather actually reduce the edge?) ---")
    boundary_softening_check(case_data)

    if not go:
        print("\n" + "=" * 70)
        print("STOPPING: Phase A guardrail failed. Corrected-preprocessing predictions")
        print("diverge too much from the original -- the model is out-of-distribution")
        print("on the corrected inputs. Any CAM difference downstream would be confounded")
        print("with this, not attributable to the boundary artifact. Not generating")
        print("corrected-preprocessing Grad-CAMs. This requires retraining to fix cleanly,")
        print("which is out of scope for this attempt.")
        print("=" * 70)
        return

    print("\n--- Phase B: stage-1 Grad-CAM on corrected preprocessing ---")
    roi = rectangular_lung_roi()
    corners = corner_regions()

    rows_sanity = []
    rows_corner = []
    rows_seg = []
    cam_cache = {}

    for _, case_row in cases.iterrows():
        subj = case_row["subjectID"]
        if subj not in case_data:
            continue
        d = case_data[subj]
        path = case_row["CR_image_paths"]
        tensor_soft, gray_soft, mask_soft = preprocess_image_soft(path)

        norm = apply_val_normalize(tensor_soft).unsqueeze(0).to(device)
        norm.requires_grad_(True)
        cam_mod = StageGradCAM(model, STAGE_IDX)
        out = cam_mod.compute(norm)
        cam_mod.remove()
        cam_224 = resize_cam(out["cam"])
        cam_cache[subj] = cam_224

        rows_sanity.append({
            "subjectID": subj,
            "grad_abs_mean": out["grad_abs_mean"],
            "grad_abs_max": out["grad_abs_max"],
            "act_abs_mean": out["act_abs_mean"],
            "cam_argmax_row": int(np.unravel_index(np.argmax(out["cam"]), out["cam"].shape)[0]),
            "cam_argmax_col": int(np.unravel_index(np.argmax(out["cam"]), out["cam"].shape)[1]),
        })

        corner_mass = mass_fraction(cam_224, corners)
        roi_mass = mass_fraction(cam_224, roi)
        rows_corner.append({"subjectID": subj, "corner_mass_fraction": corner_mass,
                             "rectangular_roi_mass_fraction": roi_mass})

        seg_mask, _, _ = lung_mask_224(gray_soft)
        seg_mass = mass_fraction(cam_224, seg_mask)
        rows_seg.append({
            "subjectID": subj, "true_los": case_row["true_los"], "predicted_los": case_row["predicted_los"],
            "seg_lung_area_fraction": float(seg_mask.mean()),
            "in_lung_mass_fraction_segmentation_corrected": seg_mass,
        })

        print(f"  {subj}: grad_abs_mean={out['grad_abs_mean']:.3e} "
              f"corner_mass={corner_mass:.3f} seg_mass={seg_mass:.3f}")

        gray_bgr_hard = cv2.cvtColor(d["gray_hard"], cv2.COLOR_GRAY2BGR)
        gray_bgr_soft = cv2.cvtColor(gray_soft, cv2.COLOR_GRAY2BGR)
        heat_overlay = make_overlay(gray_soft, cam_224)
        boundary_seg = cv2.morphologyEx(seg_mask.astype(np.uint8) * 255, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        heat_overlay_seg = heat_overlay.copy()
        heat_overlay_seg[boundary_seg] = (0, 255, 0)
        panel = np.hstack([
            label(gray_bgr_hard, "original (hard mask)"),
            label(gray_bgr_soft, "corrected (feathered)"),
            label(heat_overlay, f"stage{STAGE_IDX} Grad-CAM (corrected)"),
            label(heat_overlay_seg, "+ lung seg boundary"),
        ])
        cv2.imwrite(os.path.join(OUT_DIR, f"{subj}_before_after.png"), panel)

    sanity_df = pd.DataFrame(rows_sanity)
    sanity_df.to_csv(os.path.join(DIAG_DIR, "sanity_check_corrected.csv"), index=False)
    maps = np.stack([cam_cache[s] for s in sanity_df["subjectID"]])
    maps_c = maps.reshape(len(maps), -1)
    maps_c = maps_c - maps_c.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(maps_c, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1e-8
    maps_n = maps_c / norms
    corr = maps_n @ maps_n.T
    n = len(maps)
    off_diag = corr[~np.eye(n, dtype=bool)]
    unique_argmax = len(set(zip(sanity_df["cam_argmax_row"], sanity_df["cam_argmax_col"])))
    sanity_summary = "\n".join([
        f"Sanity check, corrected preprocessing, stage {STAGE_IDX} (n={n} cases)",
        f"grad_abs_mean range: [{sanity_df['grad_abs_mean'].min():.3e}, {sanity_df['grad_abs_mean'].max():.3e}]",
        f"pairwise map correlation (off-diag): mean={off_diag.mean():.3f}, max={off_diag.max():.3f}",
        f"distinct argmax locations: {unique_argmax}/{n} "
        f"({'PASS' if unique_argmax > 1 else 'FAIL -- maps collapsed'})",
    ])
    print("\n" + sanity_summary)
    with open(os.path.join(DIAG_DIR, "sanity_check_corrected.txt"), "w") as f:
        f.write(sanity_summary + "\n")

    corner_df = pd.DataFrame(rows_corner)
    corner_df.to_csv(os.path.join(DIAG_DIR, "corner_artifact_corrected.csv"), index=False)
    mean_corner_corrected = corner_df["corner_mass_fraction"].mean()
    mean_roi_corrected = corner_df["rectangular_roi_mass_fraction"].mean()

    seg_df = pd.DataFrame(rows_seg)
    seg_df.to_csv(os.path.join(DIAG_DIR, "segmentation_mass_fraction_corrected.csv"), index=False)
    mean_seg_corrected = seg_df["in_lung_mass_fraction_segmentation_corrected"].mean()
    mean_seg_area = seg_df["seg_lung_area_fraction"].mean()

    portable_row = seg_df[seg_df["subjectID"] == PORTABLE_FILM_CASE]
    portable_corner = corner_df[corner_df["subjectID"] == PORTABLE_FILM_CASE]
    portable_summary = (
        f"Portable-film case {PORTABLE_FILM_CASE}: seg_mass={portable_row['in_lung_mass_fraction_segmentation_corrected'].values[0]:.3f}, "
        f"corner_mass={portable_corner['corner_mass_fraction'].values[0]:.3f} "
        f"(cohort mean seg_mass={mean_seg_corrected:.3f})"
        if len(portable_row) else f"Portable-film case {PORTABLE_FILM_CASE}: not in this case set."
    )

    final_summary = "\n".join([
        "=" * 70,
        "HEAD-TO-HEAD: original (hard mask) vs corrected (feathered boundary)",
        "=" * 70,
        f"Corner-mass fraction (stage {STAGE_IDX}):  original=0.066  ->  corrected={mean_corner_corrected:.3f}",
        f"Rectangular-ROI mass fraction (stage {STAGE_IDX}): corrected={mean_roi_corrected:.3f} (original was 0.535)",
        "",
        f"Segmentation-based in-lung mass fraction (the deliverable metric):",
        f"  chance baseline (mean lung area fraction): original={BASELINE_SEG_AREA_MEAN:.3f}  corrected={mean_seg_area:.3f}",
        f"  ORIGINAL (hard mask):    0.326",
        f"  CORRECTED (feathered):  {mean_seg_corrected:.3f}",
        f"  delta: {mean_seg_corrected - BASELINE_SEG_MASS_MEAN:+.3f}",
        "",
        portable_summary,
        "",
        "Decision criterion (reported, not applied): >=~0.55-0.6 -> publishable localization result; "
        "0.3-0.45 -> artifact wasn't the main problem, model isn't strongly lung-localized.",
        f"Observed corrected value: {mean_seg_corrected:.3f}",
    ])
    print("\n" + final_summary)
    with open(os.path.join(DIAG_DIR, "final_head_to_head.txt"), "w") as f:
        f.write(final_summary + "\n")


if __name__ == "__main__":
    main()
