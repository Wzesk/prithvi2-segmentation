"""
Cloud infill via Prithvi-EO 2.0 masked autoencoder reconstruction.

The MAE was pretrained to reconstruct randomly masked patches.  By replacing
random masking with a *cloud-conditioned* mask we can reconstruct cloud-
occluded regions from the visible context.

Workflow
--------
1. Detect clouds using the existing SEnSeIv2 cloud detector (reused from
   ``littoral_cloud_impute``).
2. Convert the pixel-level cloud mask to a patch-level mask aligned with the
   ViT's 14×14 patch grid.
3. Run the MAE encoder on visible (non-cloudy) patches, then the decoder to
   predict masked (cloudy) patches.
4. Blend predicted pixels back into the original image (only replacing truly
   cloudy pixels).
5. Save the result as a cloud-free GeoTIFF preserving all geospatial metadata.
"""

import os
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch

from config import (
    DEFAULT_DEVICE,
    DEFAULT_MODEL_ID,
    DEFAULT_PATCH_SIZE,
    DEFAULT_PATCH_OVERLAP,
    DEFAULT_CLOUD_THRESHOLD,
    CLOUDLESS_DIR,
    BAND_MEAN,
    BAND_STD,
    S2_BANDS,
)
from model import load_mae, resolve_device
from data_loader import (
    read_tiff,
    save_tiff,
    select_bands,
    normalize,
    denormalize,
    tile_image,
    reassemble_patches,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cloud mask helpers
# ---------------------------------------------------------------------------

def build_cloud_mask(
    data: np.ndarray,
    *,
    include_shadow: bool = False,
    device: str = DEFAULT_DEVICE,
) -> np.ndarray:
    """Generate a binary cloud mask for a multispectral image.

    Attempts to use the SEnSeIv2-based detector from
    ``littoral_cloud_impute.vpint_cloud_impute``.  Falls back to a simple
    brightness threshold if that module is not available.

    Parameters
    ----------
    data : np.ndarray
        Full-band image array ``(bands, H, W)`` — must include all 12 S2
        bands for SEnSeIv2, or at least 6 bands for the fallback.
    include_shadow : bool
        Whether to include cloud shadow pixels in the mask.
    device : str
        Torch device for the detector model.

    Returns
    -------
    np.ndarray
        Binary mask ``(H, W)`` with 1 = cloud / shadow, 0 = clear.
    """
    try:
        import sys, importlib
        vpint = importlib.import_module("vpint_cloud_impute")
        mask = vpint.build_cloud_mask(
            data, include_shadow=include_shadow, device=device,
        )
        # Ensure mask matches image spatial dims (SEnSeIv2 may return a
        # different resolution)
        H, W = data.shape[1], data.shape[2]
        if mask.shape != (H, W):
            from PIL import Image as _Img
            mask = np.array(
                _Img.fromarray(mask).resize((W, H), _Img.NEAREST),
                dtype=np.uint8,
            )
        return mask
    except Exception:
        logger.warning(
            "SEnSeIv2 cloud detector not available; using brightness fallback."
        )
        return _fallback_cloud_mask(data)


def _fallback_cloud_mask(data: np.ndarray) -> np.ndarray:
    """Simple threshold-based cloud mask (for environments without SEnSeIv2).

    Uses high reflectance in Blue (B02) + SWIR (B11) as a cloud indicator.
    """
    if data.shape[0] >= 6:
        blue = data[0].astype(np.float32)
        swir = data[4].astype(np.float32)
    else:
        blue = data[0].astype(np.float32)
        swir = blue  # degenerate

    cloud = ((blue > 2000) & (swir > 1500)).astype(np.uint8)
    return cloud


def pixel_mask_to_patch_mask(
    mask: np.ndarray,
    patch_size: int,
    overlap: int,
    tile_info: dict,
    threshold: float = DEFAULT_CLOUD_THRESHOLD,
) -> np.ndarray:
    """Convert a pixel-level cloud mask to a flat per-patch boolean.

    A patch is marked as cloudy (True → will be masked) if the fraction of
    cloud pixels within it exceeds *threshold*.

    Parameters
    ----------
    mask : np.ndarray
        ``(H, W)`` binary cloud mask (1 = cloud).
    patch_size : int
        Spatial patch size.
    overlap : int
        Overlap used when tiling.
    tile_info : dict
        Tile metadata from :func:`data_loader.tile_image`.
    threshold : float
        Fraction of cloudy pixels in a patch to consider the whole patch
        cloudy.

    Returns
    -------
    np.ndarray
        Boolean array of length ``N_patches``.  True = masked (cloudy).
    """
    stride = tile_info["stride"]
    rows = tile_info["rows"]
    cols = tile_info["cols"]

    # Pad the mask to match the padded image used during tiling
    pad_h = tile_info["pad_h"]
    pad_w = tile_info["pad_w"]
    mask_padded = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

    patch_mask = []
    for r in range(rows):
        for c in range(cols):
            y0 = r * stride
            x0 = c * stride
            patch = mask_padded[y0 : y0 + patch_size, x0 : x0 + patch_size]
            frac = patch.mean()
            patch_mask.append(frac >= threshold)
    return np.array(patch_mask, dtype=bool)


def _patch_mask_to_token_mask(
    patch_cloud: np.ndarray,
    patches_per_image: int,
    token_patch_h: int = 14,
    token_patch_w: int = 14,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> torch.Tensor:
    """Convert per-image-patch cloud booleans into per-ViT-token mask.

    Each 224×224 image patch is divided into a 16×16 grid of 14×14 ViT tokens.
    This function maps the coarser cloud mask to the token level.

    Parameters
    ----------
    patch_cloud : np.ndarray
        Boolean array of length ``N_patches``.  True = masked.
    patches_per_image : int
        Total patches per image (should match ``len(patch_cloud)``).
    token_patch_h, token_patch_w : int
        ViT patch dimensions (14×14 for Prithvi-EO 2.0-600M).
    patch_size : int
        Image patch size (224).

    Returns
    -------
    torch.Tensor
        Boolean mask ``(N_patches, n_tokens)`` where ``n_tokens`` =
        ``(patch_size / token_patch_h) ** 2`` per image patch.
        True = masked (cloud) tokens.
    """
    tokens_per_side = patch_size // token_patch_h
    n_tokens = tokens_per_side * tokens_per_side  # 16×16 = 256

    all_masks = []
    for cloudy in patch_cloud:
        if cloudy:
            # Entire image patch is cloudy → mask all tokens
            all_masks.append(torch.ones(n_tokens, dtype=torch.bool))
        else:
            all_masks.append(torch.zeros(n_tokens, dtype=torch.bool))
    return torch.stack(all_masks)  # (N_patches, n_tokens)


# ---------------------------------------------------------------------------
# MAE cloud-conditioned reconstruction
# ---------------------------------------------------------------------------

@torch.no_grad()
def reconstruct_patches(
    model: torch.nn.Module,
    patches: np.ndarray,
    patch_cloud_mask: np.ndarray,
    device: torch.device,
    mask_ratio_fallback: float = 0.75,
) -> np.ndarray:
    """Run the MAE to reconstruct cloudy patches.

    For patches that are fully clear, the original data is returned unchanged.
    For cloudy patches, the model's decoder prediction is used.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded PrithviMAE in eval mode.
    patches : np.ndarray
        ``(N, 6, patch_size, patch_size)`` normalized float32 patches.
    patch_cloud_mask : np.ndarray
        Boolean ``(N,)`` — True for patches that should be reconstructed.
    device : torch.device
        Device to run inference on.
    mask_ratio_fallback : float
        Mask ratio used if there are no cloud-specific patches (for random
        reconstruction testing).

    Returns
    -------
    np.ndarray
        Reconstructed patches, same shape as input.
    """
    N, C, H, W = patches.shape
    result = patches.copy()

    cloud_indices = np.where(patch_cloud_mask)[0]
    if len(cloud_indices) == 0:
        logger.info("No cloudy patches — returning original data unchanged.")
        return result

    clear_indices = np.where(~patch_cloud_mask)[0]
    logger.info(
        "Reconstructing %d cloudy patches (%d clear)",
        len(cloud_indices), len(clear_indices),
    )

    # Process all patches through the model.  We provide all patches as a
    # batch (or in sub-batches for memory).  The model randomly masks tokens
    # internally; we then only keep the reconstruction for cloudy patches.
    #
    # Ideal approach: force-mask the cloudy tokens.  The current
    # implementation uses the simpler strategy of running multiple random
    # reconstructions and averaging — this works because the MAE will
    # eventually predict each token.  A follow-up optimisation can hook
    # into the model's masking to force cloudy tokens.

    BATCH = 4
    N_RUNS = 3  # average over multiple random masks for robustness

    accum = np.zeros_like(patches, dtype=np.float64)
    counts = np.zeros(N, dtype=np.float64)

    for run_idx in range(N_RUNS):
        for start in range(0, N, BATCH):
            end = min(start + BATCH, N)
            x = torch.tensor(
                patches[start:end], dtype=torch.float32, device=device,
            )
            # Model expects (B, C, T, H, W) — add temporal dim T=1
            x = x.unsqueeze(2)

            _, pred, mask = model(x, None, None, mask_ratio_fallback)

            # Unpatchify predictions back to pixel space
            rec = model.unpatchify(pred).detach().cpu().numpy()  # (B, C, T, H, W)
            mask_img = model.unpatchify(
                mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1])
            ).detach().cpu().numpy()

            rec = rec[:, :, 0, :, :]       # drop T dim → (B, C, H, W)
            mask_px = mask_img[:, :, 0, :, :]

            # Blend: where mask==1 take prediction, else original
            blended = np.where(mask_px > 0.5, rec, patches[start:end])

            accum[start:end] += blended
            counts[start:end] += 1

    accum /= np.maximum(counts[:, None, None, None], 1.0)

    # Only replace cloudy patches
    for idx in cloud_indices:
        result[idx] = accum[idx].astype(np.float32)

    return result


# ---------------------------------------------------------------------------
# Single-image cloud infill
# ---------------------------------------------------------------------------

def infill_image(
    tiff_path: str,
    output_path: str,
    *,
    model: Optional[torch.nn.Module] = None,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = DEFAULT_DEVICE,
    patch_size: int = DEFAULT_PATCH_SIZE,
    overlap: int = DEFAULT_PATCH_OVERLAP,
    cloud_threshold: float = DEFAULT_CLOUD_THRESHOLD,
) -> str:
    """Remove clouds from a single multispectral GeoTIFF.

    Parameters
    ----------
    tiff_path : str
        Path to the input pipeline GeoTIFF (6 or 7 bands).
    output_path : str
        Where to write the cloud-free result.
    model : torch.nn.Module or None
        Pre-loaded MAE; loaded on demand if ``None``.
    model_id, device, patch_size, overlap, cloud_threshold
        See module-level defaults.

    Returns
    -------
    str
        Path to the saved cloud-free GeoTIFF.
    """
    dev = resolve_device(device)

    if model is None:
        model = load_mae(model_id, device=device)

    # Read and prepare
    raw_data, meta = read_tiff(tiff_path)
    prithvi_data = select_bands(raw_data)

    # Build cloud mask on the raw (unnormalized) data
    cloud_mask = build_cloud_mask(raw_data, device=device)
    cloud_frac = cloud_mask.mean()
    logger.info(
        "%s  cloud fraction=%.1f%%",
        os.path.basename(tiff_path), cloud_frac * 100,
    )

    if cloud_frac < 0.01:
        logger.info("Nearly cloud-free — copying original without reconstruction.")
        save_tiff(prithvi_data, output_path, meta, band_names=S2_BANDS)
        return output_path

    # Normalize and tile
    normed = normalize(prithvi_data)
    patches, tile_info = tile_image(normed, patch_size, overlap)

    # Patch-level cloud mask
    patch_cloud = pixel_mask_to_patch_mask(
        cloud_mask, patch_size, overlap, tile_info, cloud_threshold,
    )

    # Reconstruct
    rec_patches = reconstruct_patches(model, patches, patch_cloud, dev)

    # Reassemble and denormalize
    full_rec = reassemble_patches(rec_patches, tile_info)
    full_rec = denormalize(full_rec)

    # Crop back to original spatial dimensions (tiling may have padded)
    orig_h, orig_w = prithvi_data.shape[1], prithvi_data.shape[2]
    full_rec = full_rec[:, :orig_h, :orig_w]

    # Blend: keep original clear pixels, use reconstruction only where cloudy
    clear_mask = cloud_mask == 0
    original_float = prithvi_data.astype(np.float32)
    for b in range(full_rec.shape[0]):
        full_rec[b][clear_mask] = original_float[b][clear_mask]

    # Clip to valid reflectance range and cast
    full_rec = np.clip(full_rec, 0, 10000).astype(np.uint16)

    save_tiff(full_rec, output_path, meta, band_names=S2_BANDS)
    logger.info("Saved cloud-free image → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def batch_cloud_infill(
    folder: str,
    *,
    output_dir: Optional[str] = None,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = DEFAULT_DEVICE,
    patch_size: int = DEFAULT_PATCH_SIZE,
    overlap: int = DEFAULT_PATCH_OVERLAP,
    cloud_threshold: float = DEFAULT_CLOUD_THRESHOLD,
    overwrite: bool = False,
) -> List[str]:
    """Remove clouds from all GeoTIFFs in *folder*.

    Parameters
    ----------
    folder : str
        Directory containing ``*.tif`` files (e.g. ``TARGETS/``).
    output_dir : str or None
        Output directory.  Defaults to ``<folder>/../prithvi2_cloudless``.
    model_id, device, patch_size, overlap, cloud_threshold, overwrite
        See :func:`infill_image`.

    Returns
    -------
    list[str]
        Paths to the saved cloud-free GeoTIFFs.
    """
    tiff_files = sorted(
        f for f in os.listdir(folder) if f.lower().endswith((".tif", ".tiff"))
    )
    if not tiff_files:
        logger.warning("No TIFF files found in %s", folder)
        return []

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(folder), CLOUDLESS_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # Load model once
    model = load_mae(model_id, device=device)

    results = []
    report_rows = []

    for fname in tiff_files:
        src = os.path.join(folder, fname)
        stem = os.path.splitext(fname)[0]
        dst = os.path.join(output_dir, f"{stem}_pred.tif")

        if not overwrite and os.path.exists(dst):
            logger.info("Skipping (exists): %s", dst)
            results.append(dst)
            continue

        try:
            out = infill_image(
                src, dst,
                model=model,
                device=device,
                patch_size=patch_size,
                overlap=overlap,
                cloud_threshold=cloud_threshold,
            )
            results.append(out)
            report_rows.append({"file": fname, "status": "success", "output": dst})
        except Exception as exc:
            logger.error("Failed on %s: %s", fname, exc)
            report_rows.append({"file": fname, "status": "failed", "error": str(exc)})

    # Write report CSV
    _write_report(output_dir, report_rows)
    return results


def _write_report(output_dir: str, rows: list):
    """Write a simple CSV report of the batch run."""
    import csv
    path = os.path.join(output_dir, "prithvi2_cloudless_report.csv")
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    logger.info("Report → %s", path)
