"""
Provides Grad-CAM for the Swin Transformer imaging model, along with the
model definition and preprocessing used to run it.
"""
import numpy as np
import torch
import torch.nn as nn
import cv2
import nibabel as nib
from skimage.transform import resize
import timm

# ---------------------------------------------------------------------------
# Preprocessing and model definition, matching imaging/train_swin.py.
# ---------------------------------------------------------------------------

def preprocess_image(path):
    """Returns (img_tensor[3,224,224] in [0,1], gray_224 uint8 CLAHE image,
    mask_224 uint8 {0,255} pre-CLAHE nonzero mask) or (None, None, None)."""
    try:
        nii = nib.load(path)
        img = nii.get_fdata()
        img[img <= 500] = 0

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


class VitRegressor(nn.Module):
    def __init__(self, pretrained=True, dropout_rate=0.3):
        super(VitRegressor, self).__init__()
        self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained, num_classes=0)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.regressor = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        out = self.regressor(features)
        return out


def apply_val_normalize(img_tensor):
    """Matches imaging/train_swin.py's validation transform: Resize((224,224))
    (a no-op here, already 224x224 from preprocess_image) then
    Normalize(mean=0.5, std=0.5)."""
    return (img_tensor - 0.5) / 0.5


# ---------------------------------------------------------------------------
# Stage-level Grad-CAM
# ---------------------------------------------------------------------------

PATCH_GRID = 56  # 224 / patch_size(4), swin_base_patch4_window7_224


def expected_stage_hw(stage_idx):
    """Stage 0 has no PatchMerging (output = patch_grid); stages 1..3 each
    halve H,W once via PatchMerging before their blocks run. Both cases
    collapse to patch_grid / 2**stage_idx."""
    return PATCH_GRID // (2 ** stage_idx)


class StageGradCAM:
    """Hooks the output of model.backbone.layers[stage_idx] (an
    SwinTransformerStage). That module's forward is `x = self.downsample(x);
    x = self.blocks(x)`, and both downsample (PatchMerging) and every block
    operate on / return (B, H, W, C) -- so the hooked tensor is already a
    proper 2D spatial grid, no manual reshape from a flat sequence needed
    (see module docstring)."""

    def __init__(self, model, stage_idx):
        self.model = model
        self.stage_idx = stage_idx
        self.activation = None
        self.handle = model.backbone.layers[stage_idx].register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        out.retain_grad()
        self.activation = out

    def remove(self):
        self.handle.remove()

    def compute(self, img_tensor):
        """img_tensor: (1,3,224,224), normalized. Returns dict with the raw
        (H,W) CAM, geometry, and gradient-magnitude diagnostics."""
        self.model.zero_grad(set_to_none=True)
        out = self.model(img_tensor)  # (1,1)
        out.backward()

        act = self.activation.detach()[0]   # (H, W, C)
        grad = self.activation.grad[0]      # (H, W, C)
        H, W, C = act.shape

        expected = expected_stage_hw(self.stage_idx)
        token_count = H * W
        assert H == expected and W == expected, (
            f"stage {self.stage_idx}: got grid {H}x{W}, expected {expected}x{expected} "
            f"from architecture (patch_grid={PATCH_GRID}, halved at each stage>0 start)"
        )

        weights = grad.mean(dim=(0, 1))            # (C,) GAP'd gradient per channel
        cam = torch.einsum('hwc,c->hw', act, weights)
        cam = torch.relu(cam)

        return {
            "cam": cam.cpu().numpy(),
            "H": H, "W": W, "token_count": token_count,
            "pred": out.item(),
            "grad_abs_mean": grad.abs().mean().item(),
            "grad_abs_max": grad.abs().max().item(),
            "act_abs_mean": act.abs().mean().item(),
        }


def resize_cam(cam_hw, size=224):
    """Linear (not cubic) upsample so non-negative CAM values can't overshoot
    negative -- keeps the resized map usable directly for mass-fraction
    calculations without a second clip step. Used identically for both the
    quantitative metric and the visual overlay so the two are consistent."""
    return cv2.resize(cam_hw.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)


def normalize_for_display(m):
    m = m - m.min()
    denom = m.max()
    if denom > 1e-8:
        m = m / denom
    return m


def make_overlay(gray_224, cam_224, alpha=0.45):
    heat = normalize_for_display(cam_224)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    gray_bgr = cv2.cvtColor(gray_224, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(gray_bgr, 1 - alpha, heatmap_color, alpha, 0)


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Crude rectangular lung ROI on the 224x224 frame
# ---------------------------------------------------------------------------

def rectangular_lung_roi(size=224):
    """Two rectangles approximating left/right lung fields: excludes the top
    margin (neck/shoulders), bottom margin (upper abdomen), outer margins
    (soft tissue/rib edge), and a central mediastinal strip."""
    y0, y1 = int(0.15 * size), int(0.90 * size)
    xL0, xL1 = int(0.10 * size), int(0.42 * size)
    xR0, xR1 = int(0.58 * size), int(0.90 * size)
    mask = np.zeros((size, size), dtype=bool)
    mask[y0:y1, xL0:xL1] = True
    mask[y0:y1, xR0:xR1] = True
    return mask


def corner_regions(size=224, frac=0.16):
    """Four corner boxes, side length = frac * size, matching the region
    attempt-1's warm-corner artifact visually occupied."""
    s = int(frac * size)
    mask = np.zeros((size, size), dtype=bool)
    mask[:s, :s] = True
    mask[:s, -s:] = True
    mask[-s:, :s] = True
    mask[-s:, -s:] = True
    return mask


def mass_fraction(cam_224, roi_mask):
    total = cam_224.sum()
    if total <= 1e-8:
        return float("nan")
    return float(cam_224[roi_mask].sum() / total)
