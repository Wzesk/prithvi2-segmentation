"""
Prithvi-EO 2.0 segmentation: inference and ``mask_from_folder`` interface.

Adds a lightweight decoder (UPerNet-style) on top of the Prithvi ViT encoder
to produce binary land / water masks.  The interface matches the existing
YOLO / SAM2 segmentation models so that the pipeline orchestrator can call
``mask_from_folder()`` identically.
"""

import os
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

# Workaround: cuDNN conv3d fails on some driver/hardware combos (V100 + PyTorch 2.7)
torch.backends.cudnn.enabled = False

from config import (
    DEFAULT_DEVICE,
    DEFAULT_MODEL_ID,
    DEFAULT_PATCH_SIZE,
    DEFAULT_PATCH_OVERLAP,
    MASK_DIR,
    MODEL_CONFIG,
)
from model import load_mae, resolve_device
from data_loader import (
    read_tiff,
    select_bands,
    normalize,
    tile_image,
    reassemble_patches,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Segmentation decoder head
# ---------------------------------------------------------------------------

class SegmentationHead(nn.Module):
    """Lightweight decoder that converts ViT token features into a binary mask.

    Architecture:
        1. Reshape tokens back to a spatial feature map.
        2. Two 3×3 conv blocks with BN + ReLU.
        3. Bilinear upsample to the original patch resolution.
        4. 1×1 conv → 1-channel logit map.
    """

    def __init__(self, embed_dim: int = 1280, patch_token_h: int = 16, patch_token_w: int = 16):
        super().__init__()
        self.patch_token_h = patch_token_h
        self.patch_token_w = patch_token_w

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, tokens: torch.Tensor, target_h: int = 224, target_w: int = 224) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : (B, N, D)
            Encoder output tokens (excluding any cls token).
        target_h, target_w : int
            Desired spatial output size.

        Returns
        -------
        torch.Tensor
            ``(B, 1, target_h, target_w)`` logits.
        """
        B, N, D = tokens.shape
        h = target_h // 14
        w = target_w // 14
        x = tokens.transpose(1, 2).reshape(B, D, h, w)   # (B, D, h, w)
        x = self.proj(x)                                   # (B, 64, h, w)
        x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return self.head(x)                                 # (B, 1, H, W)


# ---------------------------------------------------------------------------
# Full segmentation model (encoder + head)
# ---------------------------------------------------------------------------

class Prithvi2Segmenter(nn.Module):
    """Wraps the Prithvi MAE encoder + a binary segmentation head."""

    def __init__(self, mae: nn.Module, head: SegmentationHead):
        super().__init__()
        self.encoder = mae
        self.head = head

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference and return probability mask.

        Parameters
        ----------
        x : (B, C, H, W)  float32 normalized patches.

        Returns
        -------
        torch.Tensor
            ``(B, 1, H, W)`` probabilities in [0, 1].
        """
        # forward_features returns list of features from each block;
        # last element is the final norm'd output. Includes cls token at pos 0.
        features = self.encoder.forward_features(x)
        latent = features[-1]

        # Strip cls token (position 0) to keep only spatial tokens
        expected_tokens = (x.shape[2] // 14) * (x.shape[3] // 14)
        if latent.shape[1] > expected_tokens:
            latent = latent[:, -expected_tokens:]

        logits = self.head(latent, target_h=x.shape[2], target_w=x.shape[3])
        return torch.sigmoid(logits)


# ---------------------------------------------------------------------------
# Loading & saving
# ---------------------------------------------------------------------------

def load_segmenter(
    checkpoint: Optional[str] = None,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = DEFAULT_DEVICE,
    freeze_encoder: bool = True,
) -> Prithvi2Segmenter:
    """Build the segmentation model, optionally loading a fine-tuned head.

    Parameters
    ----------
    checkpoint : str or None
        Path to a ``.pt`` file containing the segmentation head weights.
        If ``None``, the head is randomly initialised (for fine-tuning or
        zero-shot evaluation).
    model_id : str
        HuggingFace model id for the backbone.
    device : str
        ``'auto'``, ``'cuda'``, or ``'cpu'``.
    freeze_encoder : bool
        Whether to freeze encoder parameters (only head is trainable).

    Returns
    -------
    Prithvi2Segmenter
    """
    dev = resolve_device(device)

    mae = load_mae(model_id, device=device)
    embed_dim = MODEL_CONFIG["embed_dim"]
    img_size = MODEL_CONFIG["img_size"]
    token_h = img_size // MODEL_CONFIG["patch_size"][1]
    token_w = img_size // MODEL_CONFIG["patch_size"][2]

    head = SegmentationHead(embed_dim=embed_dim, patch_token_h=token_h, patch_token_w=token_w)

    if checkpoint and os.path.isfile(checkpoint):
        state = torch.load(checkpoint, map_location=dev, weights_only=True)
        head.load_state_dict(state)
        logger.info("Loaded segmentation head from %s", checkpoint)

    seg = Prithvi2Segmenter(mae, head).to(dev)

    if freeze_encoder:
        for p in seg.encoder.parameters():
            p.requires_grad = False

    seg.eval()
    return seg


# ---------------------------------------------------------------------------
# Post-processing (matching existing pipeline convention)
# ---------------------------------------------------------------------------

def group_contiguous_pixels(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component (land mass).

    Parameters
    ----------
    mask : np.ndarray
        ``(H, W)`` binary mask (non-zero = land).

    Returns
    -------
    np.ndarray
        ``(H, W)`` uint8 with the largest component as 255, rest 0.
    """
    labeled, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    sizes = ndimage.sum(mask.astype(bool), labeled, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    return ((labeled == largest).astype(np.uint8)) * 255


# ---------------------------------------------------------------------------
# Single-image segmentation
# ---------------------------------------------------------------------------

def segment_tiff(
    tiff_path: str,
    segmenter: Prithvi2Segmenter,
    *,
    device: str = DEFAULT_DEVICE,
    patch_size: int = DEFAULT_PATCH_SIZE,
    overlap: int = DEFAULT_PATCH_OVERLAP,
    threshold: float = 0.5,
) -> np.ndarray:
    """Segment a single multispectral GeoTIFF into a binary land/water mask.

    Parameters
    ----------
    tiff_path : str
        Path to the input GeoTIFF (6 or 7 bands, or 6-band Prithvi cloudless).
    segmenter : Prithvi2Segmenter
        Loaded segmentation model.
    device, patch_size, overlap : see defaults.
    threshold : float
        Probability threshold for land classification.

    Returns
    -------
    np.ndarray
        ``(H, W)`` uint8 binary mask (land = 255, water = 0).
    """
    dev = resolve_device(device)

    data, _ = read_tiff(tiff_path)
    data = select_bands(data)
    normed = normalize(data)

    patches, tile_info = tile_image(normed, patch_size, overlap)

    BATCH = 4
    prob_patches = []

    for start in range(0, len(patches), BATCH):
        end = min(start + BATCH, len(patches))
        x = torch.tensor(patches[start:end], dtype=torch.float32, device=dev)
        prob = segmenter.predict(x)  # (B, 1, H, W)
        prob_patches.append(prob.cpu().numpy())

    prob_patches = np.concatenate(prob_patches, axis=0)  # (N, 1, ps, ps)

    # Reassemble probability map
    prob_full = reassemble_patches(prob_patches, tile_info)  # (1, H, W)
    prob_full = prob_full[0]  # (H, W)

    # Threshold → binary
    binary = (prob_full >= threshold).astype(np.uint8) * 255

    # Post-process: largest connected component
    binary = group_contiguous_pixels(binary)

    return binary


# ---------------------------------------------------------------------------
# mask_from_folder interface (matches YOLO / SAM2 convention)
# ---------------------------------------------------------------------------

class Prithvi2Seg:
    """Segmentation model wrapper with the ``mask_from_folder`` interface.

    This class can be dropped into the pipeline exactly where YOLOV8 or
    SAM2Seg are used.
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = DEFAULT_DEVICE,
    ):
        self.device = device
        self._segmenter = load_segmenter(
            checkpoint, model_id=model_id, device=device,
        )

    def mask_from_folder(self, folder: str, output_dir: Optional[str] = None) -> List[str]:
        """Generate binary masks for all GeoTIFFs in *folder*.

        The output masks are saved in the ``MASK/`` directory at the site
        root (derived by walking up from the input folder past any sub-
        directories under ``TARGETS/``).

        Parameters
        ----------
        folder : str
            Input directory containing ``*.tif`` files (e.g. the Prithvi2
            cloudless output, or ``TARGETS/`` directly).
        output_dir : str or None
            Explicit mask output directory.  When *None* the directory is
            computed automatically.

        Returns
        -------
        list[str]
            Paths to the saved mask PNGs.
        """
        if output_dir is not None:
            mask_dir = output_dir
        else:
            # Walk up from the input folder to find the site root (parent of TARGETS)
            parent = os.path.abspath(folder)
            while True:
                up = os.path.dirname(parent)
                if os.path.basename(parent) == "TARGETS" or up == parent:
                    break
                parent = up
            site_root = os.path.dirname(parent) if os.path.basename(parent) == "TARGETS" else os.path.dirname(folder.rstrip("/"))
            mask_dir = os.path.join(site_root, MASK_DIR)
        os.makedirs(mask_dir, exist_ok=True)

        tiff_files = sorted(
            f for f in os.listdir(folder) if f.lower().endswith((".tif", ".tiff"))
        )
        masks: List[str] = []

        for fname in tiff_files:
            src = os.path.join(folder, fname)
            stem = os.path.splitext(fname)[0]
            # Remove _pred suffix (from cloud infill output) for clean naming
            if stem.endswith("_pred"):
                stem = stem[:-5]
            # Use _nir_mask.png naming to match downstream conventions
            # (extract_boundary, refine_boundary, geo_transform all expect _nir_ prefix)
            mask_path = os.path.join(mask_dir, f"{stem}_nir_mask.png")

            try:
                binary = segment_tiff(src, self._segmenter, device=self.device)
                img = Image.fromarray(binary, mode="L")
                img.save(mask_path)
                masks.append(mask_path)
                logger.info("Mask → %s", mask_path)
            except Exception as exc:
                logger.error("Segmentation failed for %s: %s", fname, exc)

        logger.info("Generated %d masks in %s", len(masks), mask_dir)
        return masks

    def mask_from_img(self, tiff_path: str) -> Image.Image:
        """Segment a single GeoTIFF and return a PIL mask image."""
        binary = segment_tiff(tiff_path, self._segmenter, device=self.device)
        return Image.fromarray(binary, mode="L")
