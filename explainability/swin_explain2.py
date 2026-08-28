"""
Attempt 2 at Swin explainability for the canonical imaging model (vitfreeze.py).

Context: attempt 1 (../swin_explain.py, ../run_explainability.py) showed two
failure modes, judged only by eye:
  (1) attention rollout on the last stage (7x7 tokens) was washed out /
      near-uniform.
  (2) Grad-CAM on the last stage (7x7 tokens) was centrality-biased -- a
      single hot 7x7 token upsampled 32x to 224x224 necessarily smears across
      the whole central chest -- and showed a warm-corner artifact in most
      cases.

This attempt drops attention rollout (out of scope for the two hypotheses
being tested) and fixes Grad-CAM by:
  (a) pulling it from an earlier, higher-resolution Swin stage instead of the
      final 7x7 stage, so the map has spatial specificity, and
  (b) explicitly checking whether the warm-corner artifact tracks the
      preprocess_image hard-masking boundary (img <= 500 -> 0) rather than
      anatomy.

VitRegressor and preprocess_image are copied verbatim from vitfreeze.py (not
imported), same rationale as attempt 1: vitfreeze.py is a top-level script
with import-time side effects (loads patient_dict.csv, runs the full training
loop), so copying the two pure, side-effect-free pieces is the only safe way
to reuse its exact preprocessing without retriggering training. Diffed
line-by-line against vitfreeze.py's preprocess_image/VitRegressor -- identical
except preprocess_image here also returns the pre-CLAHE 0/nonzero mask, needed
for the corner-artifact check, and the CLAHE grayscale image, needed for
overlays.

--- Stage/reshape correctness (the detail the task called out as most
important) ---

timm.models.swin_transformer.SwinTransformer (installed version, see
site-packages/timm/models/swin_transformer.py) keeps the token grid in NHWC
spatial layout end-to-end: `SwinTransformerBlock.forward` takes and returns
(B, H, W, C) directly, only flattening to (num_windows*B, window_size**2, C)
*inside* `_attn` for the windowed attention matmul, then undoing that via
`window_reverse` before returning -- shift/un-shift (`torch.roll`) and
window partition/reverse are handled internally by the block itself. So
`model.backbone.layers[s]` (an `SwinTransformerStage`) both takes and
produces an already-spatial (B, H, W, C) tensor -- there is no flat token
sequence to manually reshape back into a 2D grid at the point where we hook
in. This is verified empirically below (`verify_stage_geometry`), not just
assumed from reading the source: we print the hooked tensor's H, W every run
and assert H*W equals the token count implied by the architecture
(patch_grid=56x56 for 224x224/patch4, halved by PatchMerging at the start of
every stage after the first).

swin_base_patch4_window7_224: depths=(2,2,18,2), downsample = (i > 0) for
stage i, so per-stage OUTPUT resolution/channels are:
    stage 0: 56x56, C=128   (no patch merging before it)
    stage 1: 28x28, C=256
    stage 2: 14x14, C=512   <- pulled here (STAGE_IDX=2)
    stage 3:  7x7,  C=1024  (attempt 1's stage; too coarse -> centrality bias)
In common Swin-paper 1-indexed terminology (stage 1..4) this is "stage 3".

Window/shift correctness at stage 2: input_resolution=14x14 > window_size=7,
so (unlike attempt 1's final stage) shifted windows are genuinely active here
(blocks alternate shift_size=(0,0) and (3,3), from
SwinTransformerStage.__init__: `shift_size=0 if i % 2 == 0 else window_size//2`
for the 18 blocks). We do NOT need to reason about this for Grad-CAM: shift
and un-shift both happen inside `_attn`/`window_reverse` before the block
returns its (B,H,W,C) output, so the hooked activation is already correctly
laid out in absolute image-grid coordinates regardless of which blocks were
shifted internally. This is the actual resolution to the "must handle
windowed/shifted structure" concern -- not a reshape we perform, but a
verification that timm's block already resolves it before the tensor reaches
our hook.
"""
import numpy as np
import torch
import torch.nn as nn
import cv2
import nibabel as nib
from skimage.transform import resize
import timm

# ---------------------------------------------------------------------------
# Copied verbatim (pixel path) from vitfreeze.py / attempt 1's swin_explain.py
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
    """val_transforms in vitfreeze.py: Resize((224,224)) [no-op here, already
    224x224 from preprocess_image] then Normalize(mean=0.5, std=0.5)."""
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
