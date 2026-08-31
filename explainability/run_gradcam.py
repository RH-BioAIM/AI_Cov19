"""
Runs Grad-CAM on a set of representative cases and quantifies the
resulting attention maps.
"""
import os
import numpy as np
import pandas as pd
import torch
import cv2

from swin_explain import (
    VitRegressor, preprocess_image, apply_val_normalize,
    StageGradCAM, resize_cam, make_overlay, label,
    rectangular_lung_roi, corner_regions, mass_fraction,
    expected_stage_hw, PATCH_GRID,
)
from lung_segmentation import lung_mask_224

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "overlays")
DIAG_DIR = os.path.join(HERE, "diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DIAG_DIR, exist_ok=True)

FOLD = 1
CHECKPOINT_PATH = f"/path/to/checkpoints/model_fold_{FOLD}_epoch_25.pth"
PREDICTIONS_PATH = f"/path/to/checkpoints/predictions_fold_{FOLD}.csv"
PATIENT_DICT_PATH = "/path/to/data/patient_dict.csv"
N_CASES = 10
STAGE_IDX = 2          # 0-indexed model.backbone.layers[2] -> 14x14 tokens, C=512
                        # ("stage 3" in 1-indexed Swin-paper terminology)
STAGE_IDX_CROSSCHECK = 1  # 28x28 tokens, C=256 ("stage 2") -- computed too, for comparison

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


def reshape_geometry_check(model):
    """Run one dummy forward through each candidate stage and confirm the
    hooked activation's H,W matches the architecture's expected token grid
    exactly. Writes a plain-text report -- this is the correctness detail
    the task flagged as most important."""
    lines = []
    lines.append("Reshape / token-grid verification for swin_base_patch4_window7_224")
    lines.append(f"patch_size=4, img_size=224 -> patch_grid = {PATCH_GRID}x{PATCH_GRID} tokens after patch_embed")
    lines.append("")
    lines.append("timm's SwinTransformerStage.forward keeps the token grid in NHWC spatial")
    lines.append("layout throughout (verified against installed timm source, see")
    lines.append("swin_explain.py module docstring): each hooked stage output is already")
    lines.append("(B, H, W, C) with H, W in absolute image-grid coordinates -- no manual")
    lines.append("flat-sequence reshape is performed or needed; shift/window bookkeeping is")
    lines.append("resolved internally by the block before it returns.")
    lines.append("")
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    shapes = {}

    def make_hook(idx):
        def hook(module, inp, out):
            shapes[idx] = tuple(out.shape)
        return hook

    handles = [model.backbone.layers[i].register_forward_hook(make_hook(i)) for i in range(4)]
    with torch.no_grad():
        _ = model(dummy)
    for h in handles:
        h.remove()

    for stage_idx in range(4):
        _, H, W, C = shapes[stage_idx]
        expected = expected_stage_hw(stage_idx)
        token_count = H * W
        ok = (H == expected and W == expected)
        lines.append(
            f"stage {stage_idx}: hooked activation shape (1,{H},{W},{C}) | "
            f"token_count={token_count} | expected grid={expected}x{expected} "
            f"(={expected*expected} tokens) | match={ok}"
        )
        assert ok, f"stage {stage_idx} geometry mismatch"
    lines.append("")
    lines.append(f"CHOSEN STAGE for Grad-CAM: stage {STAGE_IDX} "
                  f"({expected_stage_hw(STAGE_IDX)}x{expected_stage_hw(STAGE_IDX)} tokens) -- "
                  "1-indexed 'stage 3' in Swin-paper terminology. Chosen because it is the "
                  "earliest stage past the initial windowed-attention-only regime that still "
                  "carries most of the network's depth (18 of the model's 24 total blocks live "
                  "in this stage), balancing spatial specificity against semantic relevance. "
                  f"Stage {STAGE_IDX_CROSSCHECK} ({expected_stage_hw(STAGE_IDX_CROSSCHECK)}x"
                  f"{expected_stage_hw(STAGE_IDX_CROSSCHECK)} tokens) is computed alongside as a "
                  "cross-check.")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(DIAG_DIR, "reshape_check.txt"), "w") as f:
        f.write(report + "\n")


def compute_case_cams(model, cases):
    """For every case, run stage-2 and stage-1 Grad-CAM. Returns a list of
    per-case dicts holding both raw HxW cams and 224x224 resized cams, plus
    the preprocessing intermediates needed for the corner-artifact check."""
    results = []
    for _, row in cases.iterrows():
        subj, path = row["subjectID"], row["CR_image_paths"]
        img_tensor, gray_224, mask_224 = preprocess_image(path)
        if img_tensor is None:
            print(f"  SKIP {subj}: preprocessing failed ({path})")
            continue

        norm = apply_val_normalize(img_tensor).unsqueeze(0).to(device)
        norm.requires_grad_(True)

        cam_primary_mod = StageGradCAM(model, STAGE_IDX)
        primary = cam_primary_mod.compute(norm)
        cam_primary_mod.remove()

        norm2 = apply_val_normalize(img_tensor).unsqueeze(0).to(device)
        norm2.requires_grad_(True)
        cam_cross_mod = StageGradCAM(model, STAGE_IDX_CROSSCHECK)
        crosscheck = cam_cross_mod.compute(norm2)
        cam_cross_mod.remove()

        results.append({
            "subjectID": subj,
            "true_los": row["true_los"],
            "predicted_los": row["predicted_los"],
            "gray_224": gray_224,
            "mask_224": mask_224,
            "primary": primary,
            "crosscheck": crosscheck,
            "primary_cam_224": resize_cam(primary["cam"]),
            "crosscheck_cam_224": resize_cam(crosscheck["cam"]),
        })
        print(f"  {subj}: true_los={row['true_los']:.0f} pred_los={row['predicted_los']:.2f} "
              f"model_out_check(stage{STAGE_IDX})={primary['pred']:.2f} "
              f"grad_abs_mean={primary['grad_abs_mean']:.3e} grad_abs_max={primary['grad_abs_max']:.3e}")
    return results


def sanity_checks(results, stage_idx, cam_key, raw_key):
    """(1) gradient/activation magnitudes are non-trivial; (2) maps differ
    meaningfully across different input images. Run once per candidate
    stage (cam_key/raw_key select which stage's precomputed cams to use)."""
    rows = []
    for r in results:
        p = r[raw_key]
        rows.append({
            "subjectID": r["subjectID"],
            "stage": stage_idx,
            "grad_abs_mean": p["grad_abs_mean"],
            "grad_abs_max": p["grad_abs_max"],
            "act_abs_mean": p["act_abs_mean"],
            "cam_mean": float(p["cam"].mean()),
            "cam_max": float(p["cam"].max()),
            "cam_argmax_row": int(np.unravel_index(np.argmax(p["cam"]), p["cam"].shape)[0]),
            "cam_argmax_col": int(np.unravel_index(np.argmax(p["cam"]), p["cam"].shape)[1]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DIAG_DIR, f"sanity_check_stage{stage_idx}.csv"), index=False)

    # Pairwise correlation between cases' normalized 224x224 maps: if the
    # hook were wrong (e.g. detached graph, wrong tensor), all cams would
    # collapse to ~identical (near-1.0 correlation, near-0 variance in argmax
    # location) regardless of input.
    maps = np.stack([r[cam_key].flatten() for r in results])
    maps = maps - maps.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(maps, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1e-8
    maps_n = maps / norms
    corr = maps_n @ maps_n.T
    n = len(results)
    off_diag = corr[~np.eye(n, dtype=bool)]

    unique_argmax = len(set((row["cam_argmax_row"], row["cam_argmax_col"]) for row in rows))

    summary_lines = [
        f"Grad-CAM sanity check (stage {stage_idx}, n={n} cases)",
        f"grad_abs_mean range across cases: [{df['grad_abs_mean'].min():.3e}, {df['grad_abs_mean'].max():.3e}]",
        f"grad_abs_max range across cases:  [{df['grad_abs_max'].min():.3e}, {df['grad_abs_max'].max():.3e}]",
        f"act_abs_mean range across cases:  [{df['act_abs_mean'].min():.3e}, {df['act_abs_mean'].max():.3e}]",
        f"pairwise map correlation (off-diagonal): mean={off_diag.mean():.3f}, "
        f"min={off_diag.min():.3f}, max={off_diag.max():.3f}",
        f"distinct argmax locations across {n} cases: {unique_argmax} "
        f"({'PASS -- maps differ by case' if unique_argmax > 1 else 'FAIL -- identical argmax every case, hook is suspect'})",
    ]
    summary = "\n".join(summary_lines)
    print(summary)
    with open(os.path.join(DIAG_DIR, f"sanity_check_stage{stage_idx}.txt"), "w") as f:
        f.write(summary + "\n")
    return df


def corner_artifact_check(results, stage_idx, cam_key, save_visuals=False):
    """Quantify whether heatmap mass concentrates in the four image corners,
    and whether those corners are inside the hard preprocess_image mask
    boundary (img <= 500 -> 0) rather than anatomy."""
    corners = corner_regions()
    rows = []
    for r in results:
        cam224 = r[cam_key]
        mask = r["mask_224"]  # uint8 {0,255}, 255 = kept (nonzero) pixel

        corner_mass = mass_fraction(cam224, corners)
        corner_zero_frac = float((mask[corners] == 0).mean())  # fraction of corner pixels masked OUT (background)

        # Boundary of the kept region: pixels where the binary mask transitions.
        mask_bin = (mask > 0).astype(np.uint8)
        boundary = cv2.morphologyEx(mask_bin * 255, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        boundary_dilated = cv2.dilate(boundary.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        boundary_mass = mass_fraction(cam224, boundary_dilated)

        rows.append({
            "subjectID": r["subjectID"],
            "corner_mass_fraction": corner_mass,
            "corner_pixels_masked_out_fraction": corner_zero_frac,
            "mask_boundary_mass_fraction": boundary_mass,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DIAG_DIR, f"corner_artifact_stage{stage_idx}.csv"), index=False)

    mean_corner = df["corner_mass_fraction"].mean()
    mean_boundary = df["mask_boundary_mass_fraction"].mean()
    mean_corner_masked_out = df["corner_pixels_masked_out_fraction"].mean()
    corner_expected_by_area = corners.mean()  # if mass were spread uniform, corners get this fraction by pure area

    verdict_lines = [
        f"Corner-artifact quantification (stage {stage_idx}, n={len(df)} cases)",
        f"Corner boxes cover {corner_expected_by_area:.3f} of frame area (uniform-mass baseline).",
        f"Mean fraction of heatmap mass falling in the 4 corner boxes: {mean_corner:.3f}",
        f"Mean fraction of *those same corner pixels* that are masked-out background (img<=500->0): {mean_corner_masked_out:.3f}",
        f"Mean fraction of heatmap mass within 3px-dilated preprocess-mask boundary: {mean_boundary:.3f}",
        "",
    ]
    if mean_corner <= corner_expected_by_area * 1.5:
        verdict_lines.append(
            f"VERDICT: corner mass ({mean_corner:.3f}) is close to the uniform-mass baseline "
            f"({corner_expected_by_area:.3f}) -- no evidence of a warm-corner artifact at stage {stage_idx}."
        )
    else:
        verdict_lines.append(
            f"VERDICT: corner mass ({mean_corner:.3f}) exceeds the uniform-mass baseline "
            f"({corner_expected_by_area:.3f}) by >1.5x -- corner artifact may still be present at stage {stage_idx}."
        )
    verdict = "\n".join(verdict_lines)
    print(verdict)
    with open(os.path.join(DIAG_DIR, f"corner_artifact_stage{stage_idx}.txt"), "w") as f:
        f.write(verdict + "\n")

    if save_visuals:
        # Visual check for the 5 cases spanning the widest true_los range.
        sample_idx = np.linspace(0, len(results) - 1, min(5, len(results))).round().astype(int)
        for i in sample_idx:
            r = results[i]
            gray_bgr = cv2.cvtColor(r["gray_224"], cv2.COLOR_GRAY2BGR)
            mask_bin = (r["mask_224"] > 0).astype(np.uint8)
            boundary = cv2.morphologyEx(mask_bin * 255, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
            boundary_img = gray_bgr.copy()
            boundary_img[boundary] = (0, 255, 255)  # yellow = mask boundary
            heat_overlay = make_overlay(r["gray_224"], r[cam_key])
            panel = np.hstack([
                label(gray_bgr.copy(), "CLAHE image"),
                label(boundary_img, "mask boundary (yellow)"),
                label(heat_overlay, f"Grad-CAM stage{stage_idx}"),
            ])
            cv2.imwrite(os.path.join(DIAG_DIR, f"corner_artifact_stage{stage_idx}_{r['subjectID']}.png"), panel)

    return df, mean_corner, corner_expected_by_area


def roi_mass_fraction_check(results):
    roi = rectangular_lung_roi()
    rows = []
    for r in results:
        cam224 = r["primary_cam_224"]
        cam224_cross = r["crosscheck_cam_224"]
        rows.append({
            "subjectID": r["subjectID"],
            "true_los": r["true_los"],
            "predicted_los": r["predicted_los"],
            f"in_lung_mass_fraction_stage{STAGE_IDX}": mass_fraction(cam224, roi),
            f"in_lung_mass_fraction_stage{STAGE_IDX_CROSSCHECK}": mass_fraction(cam224_cross, roi),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DIAG_DIR, "roi_mass_fraction.csv"), index=False)

    mean_primary = df[f"in_lung_mass_fraction_stage{STAGE_IDX}"].mean()
    mean_cross = df[f"in_lung_mass_fraction_stage{STAGE_IDX_CROSSCHECK}"].mean()
    roi_area_frac = roi.mean()
    summary = "\n".join([
        f"Rectangular lung-ROI mass fraction (n={len(df)} cases)",
        f"ROI covers {roi_area_frac:.3f} of frame area (uniform-mass baseline).",
        f"Mean in-ROI mass fraction, stage {STAGE_IDX} (14x14): {mean_primary:.3f} "
        f"-> {'GO' if mean_primary >= 0.5 else 'NO-GO'}",
        f"Mean in-ROI mass fraction, stage {STAGE_IDX_CROSSCHECK} (28x28): {mean_cross:.3f} "
        f"-> {'GO' if mean_cross >= 0.5 else 'NO-GO'}",
        "",
        "GO/NO-GO threshold: proceed to segmentation only for a stage whose mean >= ~0.5.",
    ])
    print(summary)
    with open(os.path.join(DIAG_DIR, "roi_mass_fraction.txt"), "w") as f:
        f.write(summary + "\n")
    return df, {STAGE_IDX: mean_primary, STAGE_IDX_CROSSCHECK: mean_cross}, roi_area_frac


def segmentation_mass_fraction_check(results, stage_idx, cam_key):
    """Recompute in-lung mass fraction against a real pretrained lung
    segmentation mask (torchxrayvision ChestX-Det PSPNet) instead of the
    crude rectangle. Only called for a stage that already cleared the
    rectangle GO/NO-GO threshold."""
    rows = []
    for r in results:
        mask, left_p, right_p = lung_mask_224(r["gray_224"])
        cam224 = r[cam_key]
        rows.append({
            "subjectID": r["subjectID"],
            "true_los": r["true_los"],
            "predicted_los": r["predicted_los"],
            "seg_lung_area_fraction": float(mask.mean()),
            "in_lung_mass_fraction_segmentation": mass_fraction(cam224, mask),
        })
        r["_seg_mask"] = mask  # stash for overlay generation

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DIAG_DIR, f"segmentation_mass_fraction_stage{stage_idx}.csv"), index=False)

    mean_seg = df["in_lung_mass_fraction_segmentation"].mean()
    mean_seg_area = df["seg_lung_area_fraction"].mean()
    summary = "\n".join([
        f"Segmentation-based in-lung mass fraction (stage {stage_idx}, n={len(df)} cases)",
        f"Pretrained lung mask (torchxrayvision ChestX-Det PSPNet) covers "
        f"{mean_seg_area:.3f} of frame area on average (uniform-mass baseline).",
        f"Mean in-lung mass fraction (real segmentation): {mean_seg:.3f}",
    ])
    print(summary)
    with open(os.path.join(DIAG_DIR, f"segmentation_mass_fraction_stage{stage_idx}.txt"), "w") as f:
        f.write(summary + "\n")

    # Overlay: CLAHE image with segmentation boundary + Grad-CAM heat.
    for r in results:
        mask = r["_seg_mask"]
        gray_bgr = cv2.cvtColor(r["gray_224"], cv2.COLOR_GRAY2BGR)
        boundary = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        seg_img = gray_bgr.copy()
        seg_img[boundary] = (0, 255, 0)  # green = segmented lung boundary
        heat_overlay = make_overlay(r["gray_224"], r[cam_key])
        heat_overlay[boundary] = (0, 255, 0)
        panel = np.hstack([
            label(gray_bgr.copy(), "CLAHE image"),
            label(seg_img, "segmented lung (green)"),
            label(heat_overlay, f"Grad-CAM stage{stage_idx} + lung boundary"),
        ])
        cv2.imwrite(os.path.join(DIAG_DIR, f"segmentation_stage{stage_idx}_{r['subjectID']}.png"), panel)

    return df, mean_seg


def save_overlays_and_catalog(results):
    roi = rectangular_lung_roi()
    catalog_rows = []
    for r in results:
        subj = r["subjectID"]
        gray_224 = r["gray_224"]
        primary_overlay = label(make_overlay(gray_224, r["primary_cam_224"]),
                                 f"Grad-CAM stage{STAGE_IDX} (14x14)")
        cross_overlay = label(make_overlay(gray_224, r["crosscheck_cam_224"]),
                               f"Grad-CAM stage{STAGE_IDX_CROSSCHECK} (28x28)")
        original_bgr = label(cv2.cvtColor(gray_224, cv2.COLOR_GRAY2BGR),
                              f"{subj} true={r['true_los']:.0f}d pred={r['predicted_los']:.1f}d")

        roi_vis = cv2.cvtColor(gray_224, cv2.COLOR_GRAY2BGR)
        roi_overlay_layer = roi_vis.copy()
        roi_overlay_layer[roi] = (0, 200, 0)
        roi_vis = cv2.addWeighted(roi_vis, 0.7, roi_overlay_layer, 0.3, 0)
        roi_vis = label(roi_vis, "lung ROI (rectangle)")

        side_by_side = np.hstack([original_bgr, primary_overlay, cross_overlay, roi_vis])

        primary_path = os.path.join(OUT_DIR, f"{subj}_gradcam_stage{STAGE_IDX}.png")
        cross_path = os.path.join(OUT_DIR, f"{subj}_gradcam_stage{STAGE_IDX_CROSSCHECK}.png")
        combined_path = os.path.join(OUT_DIR, f"{subj}_sidebyside.png")
        cv2.imwrite(primary_path, primary_overlay)
        cv2.imwrite(cross_path, cross_overlay)
        cv2.imwrite(combined_path, side_by_side)

        catalog_rows.append({
            "subjectID": subj, "fold": FOLD,
            "true_los": r["true_los"], "predicted_los": r["predicted_los"],
            "gradcam_stage2_file": os.path.relpath(primary_path, HERE),
            "gradcam_stage1_file": os.path.relpath(cross_path, HERE),
            "sidebyside_file": os.path.relpath(combined_path, HERE),
        })
    catalog = pd.DataFrame(catalog_rows)
    catalog.to_csv(os.path.join(HERE, "catalog.csv"), index=False)
    print(f"Saved catalog: {os.path.join(HERE, 'catalog.csv')} ({len(catalog)} cases)")


def main():
    model = load_model()
    reshape_geometry_check(model)

    cases = select_representative_cases()
    print(f"Selected {len(cases)} cases spanning LOS range "
          f"[{cases['true_los'].min()}, {cases['true_los'].max()}]:")
    print(cases[["subjectID", "true_los", "predicted_los"]].to_string(index=False))

    results = compute_case_cams(model, cases)

    sanity_checks(results, STAGE_IDX, "primary_cam_224", "primary")
    sanity_checks(results, STAGE_IDX_CROSSCHECK, "crosscheck_cam_224", "crosscheck")
    corner_artifact_check(results, STAGE_IDX, "primary_cam_224", save_visuals=True)
    corner_artifact_check(results, STAGE_IDX_CROSSCHECK, "crosscheck_cam_224", save_visuals=True)
    _, means_by_stage, _ = roi_mass_fraction_check(results)
    save_overlays_and_catalog(results)

    cam_key_by_stage = {STAGE_IDX: "primary_cam_224", STAGE_IDX_CROSSCHECK: "crosscheck_cam_224"}
    for stage_idx, mean_val in means_by_stage.items():
        if mean_val >= 0.5:
            print(f"\nStage {stage_idx} cleared the rectangle GO/NO-GO threshold "
                  f"(mean={mean_val:.3f}) -- running segmentation-based recompute.")
            segmentation_mass_fraction_check(results, stage_idx, cam_key_by_stage[stage_idx])
        else:
            print(f"\nStage {stage_idx} did NOT clear the rectangle threshold "
                  f"(mean={mean_val:.3f}) -- skipping segmentation for this stage.")


if __name__ == "__main__":
    main()
