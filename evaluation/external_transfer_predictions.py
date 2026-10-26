"""
Runs the five imaging fold models over both external datasets and saves the
per-image predicted length of stay. Preprocessing matches the validation
transform used in imaging/train_swin.py.
"""
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from skimage.transform import resize
from torchvision import transforms

REVISION_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = "/path/to/checkpoints"
MIDRC_CSV = "/path/to/data/a_ricord_image_severity.csv"
COVID19_COVID_DIR = "/path/to/data/COVID-19_Radiography_Dataset/COVID/images"
COVID19_NORMAL_DIR = "/path/to/data/COVID-19_Radiography_Dataset/Normal/images"

SEED = 42
BATCH_SIZE = 32

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Validation transform from imaging/train_swin.py, applied after the
# intensity preprocessing below so external inputs are on the same scale the
# model was trained on.
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def preprocess_xray_png(path):
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = resize(img, (224, 224), preserve_range=True).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img)
        img_tensor = torch.tensor(img_clahe, dtype=torch.float32).unsqueeze(0) / 255.0
        img_tensor = img_tensor.repeat(3, 1, 1)
        img_tensor = val_transforms(img_tensor)
        return img_tensor
    except Exception as e:
        print(f"Error processing {path}: {e}", file=sys.stderr)
        return None


class VitRegressor(nn.Module):
    def __init__(self, pretrained=False, dropout_rate=0.3):
        super(VitRegressor, self).__init__()
        self.backbone = timm.create_model("swin_base_patch4_window7_224", pretrained=pretrained, num_classes=0)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.regressor = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        out = self.regressor(features)
        return out


def load_models():
    models = {}
    for fold in range(1, 6):
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_fold_{fold}_epoch_25.pth")
        m = VitRegressor(pretrained=False).to(device)
        m.load_state_dict(torch.load(ckpt_path, map_location=device))
        m.eval()
        models[fold] = m
    return models


def run_inference(image_paths, models, batch_size=BATCH_SIZE):
    """Each image is preprocessed once and scored by all five fold models."""
    n = len(image_paths)
    preds = {fold: np.full(n, np.nan, dtype=np.float64) for fold in models}
    n_failed = 0
    t0 = time.time()

    buf_tensors, buf_idx = [], []

    def flush():
        if not buf_tensors:
            return
        batch = torch.stack(buf_tensors).to(device)
        with torch.no_grad():
            for fold, m in models.items():
                out = m(batch).squeeze(-1).detach().cpu().numpy()
                out = np.atleast_1d(out)
                for j, idx in enumerate(buf_idx):
                    preds[fold][idx] = out[j]
        buf_tensors.clear()
        buf_idx.clear()

    for i, path in enumerate(image_paths):
        t = preprocess_xray_png(path)
        if t is None:
            n_failed += 1
            continue
        buf_tensors.append(t)
        buf_idx.append(i)
        if len(buf_tensors) >= batch_size:
            flush()
        if (i + 1) % 1000 == 0:
            print(f"  processed {i + 1}/{n} images, {n_failed} preprocessing failures so far, "
                  f"elapsed {time.time() - t0:.1f}s", flush=True)
    flush()

    print(f"Done: {n} images, {n_failed} preprocessing failures, elapsed {time.time() - t0:.1f}s")
    return preds, n_failed


def save_predictions(frame, preds, name):
    fold_cols = [f"pred_fold{f}" for f in range(1, 6)]
    for fold in range(1, 6):
        frame[f"pred_fold{fold}"] = preds[fold]
    frame["pred_ensemble_mean"] = frame[fold_cols].mean(axis=1)
    path = os.path.join(REVISION_DIR, f"{name}_predictions_all_folds.csv")
    frame.to_csv(path, index=False, float_format="%.8f")
    print(f"Saved {path} ({len(frame)} rows)")
    return frame


def main():
    print(f"Device: {device} (CUDA available: {torch.cuda.is_available()})")
    models = load_models()
    print(f"Loaded {len(models)} fold checkpoints from {CHECKPOINT_DIR}")

    # --- MIDRC-RICORD-1c: positive class is severity level 2 ---
    print("\nMIDRC-RICORD-1c")
    midrc = pd.read_csv(MIDRC_CSV)
    n_raw = len(midrc)
    midrc["severity_num"] = midrc["severity"].map({"level1": 1, "level2": 2})
    midrc = midrc.dropna(subset=["severity_num"]).reset_index(drop=True)
    print(f"Raw rows: {n_raw}; dropped {n_raw - len(midrc)} rows with missing or invalid "
          f"severity; {len(midrc)} remain.")
    midrc["label"] = (midrc["severity_num"] == 2).astype(int)

    preds, _ = run_inference(midrc["image_path"].tolist(), models)
    save_predictions(midrc[["image_path", "patient_id", "label"]].copy(), preds, "midrc")

    # --- COVID-19 Radiography Database: positive class is COVID-19 ---
    print("\nCOVID-19 Radiography Database")
    covid_paths = [os.path.join(COVID19_COVID_DIR, f) for f in sorted(os.listdir(COVID19_COVID_DIR))]
    normal_paths = [os.path.join(COVID19_NORMAL_DIR, f) for f in sorted(os.listdir(COVID19_NORMAL_DIR))]
    print(f"COVID-19 images: {len(covid_paths)}; normal images: {len(normal_paths)}")

    all_paths = covid_paths + normal_paths
    labels = [1] * len(covid_paths) + [0] * len(normal_paths)

    preds, _ = run_inference(all_paths, models)
    save_predictions(pd.DataFrame({"image_path": all_paths, "label": labels}),
                     preds, "covid19_radiography")


if __name__ == "__main__":
    main()
