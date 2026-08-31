"""
Provides two preprocessing variants for the imaging model's input
pipeline: preprocess_image_hard and preprocess_image_soft.
"""
import numpy as np
import torch
import cv2
import nibabel as nib
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

THRESHOLD = 500
FEATHER_SIGMA_224 = 8.0  # px, at the 224x224 scale


def preprocess_image_hard(path):
    """Copied verbatim from vitfreeze.py (as in attempt 1/2's swin_explain*.py).
    Also returns the pre-CLAHE binary mask, needed to build the feathered
    version and for the boundary-gradient check."""
    try:
        nii = nib.load(path)
        img = nii.get_fdata()
        img[img <= THRESHOLD] = 0

        nonzero = img[img > 0]
        if nonzero.size == 0:
            return None, None, None
        min_val, max_val = nonzero.min(), nonzero.max()
        img = (img - min_val) / (max_val - min_val + 1e-8)
        img[img == (0 - min_val) / (max_val - min_val + 1e-8)] = 0

        img = resize(img, (224, 224), preserve_range=True)
        img_uint8 = np.uint8(img * 255)
        if np.max(img_uint8) == 0:
            return None, None, None

        mask_224 = np.uint8(np.where(img_uint8 > 0, 255, 0))

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_uint8)

        img_tensor = torch.tensor(img_clahe, dtype=torch.float32).unsqueeze(0) / 255.0
        img_tensor = img_tensor.repeat(3, 1, 1)
        return img_tensor, img_clahe, mask_224
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None, None, None


def preprocess_image_soft(path, feather_sigma=FEATHER_SIGMA_224):
    """Identical to preprocess_image_hard through CLAHE; then feathers the
    hard boundary in CLAHE's output. See module docstring."""
    img_tensor_hard, img_clahe_hard, mask_hard = preprocess_image_hard(path)
    if img_tensor_hard is None:
        return None, None, None

    soft_mask = gaussian_filter((mask_hard > 0).astype(np.float32), sigma=feather_sigma)
    img_clahe_soft = np.clip(img_clahe_hard.astype(np.float32) * soft_mask, 0, 255).astype(np.uint8)

    img_tensor = torch.tensor(img_clahe_soft, dtype=torch.float32).unsqueeze(0) / 255.0
    img_tensor = img_tensor.repeat(3, 1, 1)
    mask_224 = np.uint8(soft_mask * 255)  # continuous-valued, for diagnostics
    return img_tensor, img_clahe_soft, mask_224


def boundary_band(mask_hard, dilate_px=5):
    """Fixed spatial band straddling the unfeathered hard-mask boundary, used to
    measure gradient magnitude in the same location for both variants (a
    whole-image gradient max/mean is dominated by unrelated high-contrast
    markers/ID tags, not the masking boundary -- this isolates the boundary
    specifically)."""
    mask_bin = (mask_hard > 0).astype(np.uint8)
    edge = cv2.morphologyEx(mask_bin * 255, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    k = 2 * dilate_px + 1
    return cv2.dilate(edge.astype(np.uint8), np.ones((k, k), np.uint8)) > 0


def sobel_mag(gray):
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


def boundary_gradient_stats(gray_224, band):
    mag = sobel_mag(gray_224)
    in_band = mag[band]
    return {
        "grad_band_mean": float(in_band.mean()),
        "grad_band_max": float(in_band.max()),
        "grad_band_p90": float(np.percentile(in_band, 90)),
    }
