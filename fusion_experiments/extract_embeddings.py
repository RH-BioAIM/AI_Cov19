"""
Extracts the Swin backbone's embeddings for every patient, out-of-fold.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# swin_explain.py lives at ../explainability/swin_explain.py.
EXPLAINABILITY_DIR = os.path.normpath(os.path.join(HERE, "..", "explainability"))
sys.path.insert(0, EXPLAINABILITY_DIR)
from swin_explain import VitRegressor, preprocess_image, apply_val_normalize  # noqa: E402

CHECKPOINT_DIR = "/path/to/checkpoints"
ALL_FOLD_PRED_PATH = os.path.join(CHECKPOINT_DIR, "all_fold_predictions.csv")
PATIENT_DICT_PATH = "/path/to/data/patient_dict.csv"
OUT_PATH = os.path.join(HERE, "imaging_embeddings.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cpu":
    torch.set_num_threads(4)


def main():
    fold_map = pd.read_csv(ALL_FOLD_PRED_PATH)[["subjectID", "true_los", "predicted_los", "fold"]]
    patient_dict = pd.read_csv(PATIENT_DICT_PATH)[["subjectID", "CR_image_paths"]]
    df = fold_map.merge(patient_dict, on="subjectID", how="inner")
    assert len(df) == len(fold_map) == 1341, f"expected 1341 patients, got merge={len(df)} fold_map={len(fold_map)}"
    print(f"Loaded {len(df)} patients across {df['fold'].nunique()} folds")

    all_rows = []
    t_start = time.time()
    n_done = 0

    for fold in sorted(df["fold"].unique()):
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_fold_{fold}_epoch_25.pth")
        print(f"\n--- Fold {fold}: loading {ckpt_path} ---")
        model = VitRegressor(pretrained=False).to(device)
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        fold_df = df[df["fold"] == fold]
        print(f"  {len(fold_df)} patients in this held-out fold")

        for _, row in fold_df.iterrows():
            subj, path = row["subjectID"], row["CR_image_paths"]
            result = preprocess_image(path)
            img_tensor = result[0]
            if img_tensor is None:
                print(f"    SKIP {subj}: preprocessing failed")
                continue

            with torch.no_grad():
                x = apply_val_normalize(img_tensor).unsqueeze(0).to(device)
                emb = model.backbone(x)  # (1, 1024), already globally pooled
                pred = model.regressor(model.dropout(emb)).item()

            emb_np = emb[0].cpu().numpy()
            all_rows.append({
                "subjectID": subj, "fold": fold,
                "true_los": row["true_los"],
                "predicted_los_committed": row["predicted_los"],
                "predicted_los_check": pred,
                **{f"emb_{i}": emb_np[i] for i in range(1024)},
            })
            n_done += 1
            if n_done % 100 == 0:
                elapsed = time.time() - t_start
                rate = elapsed / n_done
                eta = rate * (1341 - n_done)
                print(f"    [{n_done}/1341] elapsed={elapsed:.0f}s rate={rate:.2f}s/img eta={eta:.0f}s")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = pd.DataFrame(all_rows)
    assert len(out) == 1341, f"expected 1341 embeddings, got {len(out)}"

    # sanity check: re-derived prediction should match committed all_fold_predictions.csv
    diff = (out["predicted_los_check"] - out["predicted_los_committed"]).abs()
    print(f"\nSanity check -- predicted_los_check vs committed all_fold_predictions.csv: "
          f"max abs diff = {diff.max():.6f}, mean abs diff = {diff.mean():.6f}")

    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH} ({len(out)} rows, {out.shape[1]} cols)")
    print(f"Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
