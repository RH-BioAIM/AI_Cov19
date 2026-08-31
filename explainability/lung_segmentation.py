"""
Segments the lungs in a chest radiograph using a pretrained
lung-segmentation model.
"""
import numpy as np
import torch
import cv2
import torchxrayvision as xrv

_model = None


def get_model():
    global _model
    if _model is None:
        _model = xrv.baseline_models.chestx_det.PSPNet()
        _model.eval()
    return _model


def lung_mask_224(gray_224_uint8, device="cpu", prob_thresh=0.5):
    """gray_224_uint8: (224,224) uint8 CLAHE image (same array used for the
    Grad-CAM overlay). Returns (mask_bool_224, left_prob_224, right_prob_224)."""
    model = get_model().to(device)
    x01 = torch.tensor(gray_224_uint8, dtype=torch.float32) / 255.0
    x = (x01 * 2048 - 1024).unsqueeze(0).unsqueeze(0).to(device)  # model un-does this to recover [0,1]
    with torch.no_grad():
        logits = model(x)  # (1,14,512,512)
    probs = torch.sigmoid(logits)[0]
    left = probs[model.targets.index("Left Lung")].cpu().numpy()
    right = probs[model.targets.index("Right Lung")].cpu().numpy()

    left_224 = cv2.resize(left, (224, 224), interpolation=cv2.INTER_LINEAR)
    right_224 = cv2.resize(right, (224, 224), interpolation=cv2.INTER_LINEAR)
    mask = (left_224 >= prob_thresh) | (right_224 >= prob_thresh)
    return mask, left_224, right_224
