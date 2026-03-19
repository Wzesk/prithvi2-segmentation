"""
Data loading utilities for Prithvi-EO 2.0.

Handles reading multispectral GeoTIFFs, selecting / reordering bands to
match the Prithvi input order, normalizing to HLS reflectance statistics,
and tiling images into 224×224 patches (with overlap) for model inference.
"""

import os
import logging
from typing import List, Optional, Tuple, Dict

import numpy as np
import rasterio

from config import (
    TIFF_7BAND_ORDER,
    TIFF_6BAND_ORDER,
    PRITHVI_INDICES_FROM_7BAND,
    S2_BANDS,
    BAND_MEAN,
    BAND_STD,
    NO_DATA,
    NO_DATA_FLOAT,
    DEFAULT_PATCH_SIZE,
    DEFAULT_PATCH_OVERLAP,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GeoTIFF I/O
# ---------------------------------------------------------------------------

def read_tiff(path: str) -> Tuple[np.ndarray, dict]:
    """Read a multi-band GeoTIFF.

    Returns
    -------
    data : np.ndarray
        Shape ``(bands, H, W)``, dtype matching the file (usually uint16).
    meta : dict
        Rasterio metadata (CRS, transform, dtype, …).
    """
    with rasterio.open(path) as src:
        data = src.read()
        meta = src.meta.copy()
    return data, meta


def save_tiff(
    data: np.ndarray,
    path: str,
    meta: dict,
    band_names: Optional[List[str]] = None,
) -> str:
    """Save a multi-band array as a GeoTIFF, preserving CRS / transform.

    Parameters
    ----------
    data : np.ndarray
        Shape ``(bands, H, W)``.
    path : str
        Output file path.
    meta : dict
        Rasterio metadata to use (modified automatically for count/dtype).
    band_names : list[str] or None
        Optional band description tags.

    Returns
    -------
    str
        The saved file path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    meta = meta.copy()
    meta.update(
        driver="GTiff",
        count=data.shape[0],
        height=data.shape[1],
        width=data.shape[2],
        dtype=data.dtype,
        compress="lzw",
    )
    with rasterio.open(path, "w", **meta) as dst:
        for i in range(data.shape[0]):
            dst.write(data[i], i + 1)
        if band_names:
            dst.update_tags(bands=",".join(band_names))
    return path


# ---------------------------------------------------------------------------
# Band selection & normalization
# ---------------------------------------------------------------------------

def detect_band_count(data: np.ndarray) -> str:
    """Return ``'7band'`` or ``'6band'`` depending on array shape."""
    n = data.shape[0]
    if n >= 7:
        return "7band"
    if n == 6:
        return "6band"
    raise ValueError(f"Expected 6 or 7 bands, got {n}")


def select_bands(data: np.ndarray) -> np.ndarray:
    """Select and reorder bands from a pipeline TIFF to Prithvi order.

    For a 7-band TIFF ``[B02, B03, B04, B08, B8A, B11, B12]`` keeps
    indices ``[0, 1, 2, 4, 5, 6]`` → ``[B02, B03, B04, B8A, B11, B12]``.

    For a legacy 6-band TIFF (no B12), copies B11 into the B12 slot as a
    fallback and logs a warning.

    Parameters
    ----------
    data : np.ndarray
        Shape ``(bands, H, W)`` with 6 or 7 bands.

    Returns
    -------
    np.ndarray
        Shape ``(6, H, W)`` in Prithvi band order.
    """
    kind = detect_band_count(data)
    if kind == "7band":
        return data[PRITHVI_INDICES_FROM_7BAND]

    # 6-band fallback: B12 is missing — duplicate B11 as approximate SWIR-2
    logger.warning(
        "6-band TIFF detected (no B12). Duplicating B11 as SWIR-2 stand-in. "
        "For best results, re-run the download step to capture B12."
    )
    selected = data[[0, 1, 2, 4, 5]]             # B02,B03,B04,B8A,B11
    b12_proxy = data[5:6]                          # copy B11
    return np.concatenate([selected, b12_proxy], axis=0)


def normalize(data: np.ndarray) -> np.ndarray:
    """Normalize 6-band reflectance data to zero-mean / unit-std.

    Expects ``data`` in HLS reflectance units (0–10000 scale, uint16 or float).
    No-data pixels (== :data:`NO_DATA`) are set to :data:`NO_DATA_FLOAT`.

    Parameters
    ----------
    data : np.ndarray
        Shape ``(6, H, W)`` or ``(6, T, H, W)``.

    Returns
    -------
    np.ndarray
        float32 normalized array, same shape.
    """
    mean = np.array(BAND_MEAN, dtype=np.float32)
    std = np.array(BAND_STD, dtype=np.float32)

    data = data.astype(np.float32)

    if data.ndim == 3:
        # (C, H, W) — reshape mean/std to (C, 1, 1)
        mean = mean[:, None, None]
        std = std[:, None, None]
    elif data.ndim == 4:
        # (C, T, H, W) — reshape to (C, 1, 1, 1)
        mean = mean[:, None, None, None]
        std = std[:, None, None, None]

    nodata_mask = data == NO_DATA
    data = (data - mean) / std
    data[nodata_mask] = NO_DATA_FLOAT
    return data


def denormalize(data: np.ndarray) -> np.ndarray:
    """Reverse :func:`normalize` — return values to reflectance scale."""
    mean = np.array(BAND_MEAN, dtype=np.float32)
    std = np.array(BAND_STD, dtype=np.float32)

    if data.ndim == 3:
        mean = mean[:, None, None]
        std = std[:, None, None]
    elif data.ndim == 4:
        mean = mean[:, None, None, None]
        std = std[:, None, None, None]

    return data * std + mean


# ---------------------------------------------------------------------------
# Patch tiling / reassembly
# ---------------------------------------------------------------------------

def tile_image(
    data: np.ndarray,
    patch_size: int = DEFAULT_PATCH_SIZE,
    overlap: int = DEFAULT_PATCH_OVERLAP,
) -> Tuple[np.ndarray, Dict]:
    """Split a ``(C, H, W)`` image into overlapping patches.

    The image is first reflect-padded so that it divides evenly into patches
    at the given stride (``patch_size - overlap``).

    Parameters
    ----------
    data : np.ndarray
        Shape ``(C, H, W)``.
    patch_size : int
        Spatial size of each square patch.
    overlap : int
        Number of overlapping pixels between adjacent patches.

    Returns
    -------
    patches : np.ndarray
        Shape ``(N, C, patch_size, patch_size)`` where N = rows × cols.
    tile_info : dict
        Metadata needed by :func:`reassemble_patches` to stitch patches back.
    """
    C, H, W = data.shape
    stride = patch_size - overlap

    # Compute padding so H and W divide evenly by stride
    pad_h = (stride - ((H - patch_size) % stride)) % stride
    pad_w = (stride - ((W - patch_size) % stride)) % stride
    padded = np.pad(data, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    _, pH, pW = padded.shape

    rows = (pH - patch_size) // stride + 1
    cols = (pW - patch_size) // stride + 1

    patches = []
    for r in range(rows):
        for c in range(cols):
            y0 = r * stride
            x0 = c * stride
            patches.append(padded[:, y0 : y0 + patch_size, x0 : x0 + patch_size])

    patches = np.stack(patches, axis=0)  # (N, C, ps, ps)

    tile_info = {
        "orig_H": H,
        "orig_W": W,
        "pad_h": pad_h,
        "pad_w": pad_w,
        "rows": rows,
        "cols": cols,
        "stride": stride,
        "patch_size": patch_size,
    }
    return patches, tile_info


def reassemble_patches(
    patches: np.ndarray,
    tile_info: Dict,
) -> np.ndarray:
    """Re-stitch patches into a full image, averaging overlapping regions.

    Parameters
    ----------
    patches : np.ndarray
        Shape ``(N, C, patch_size, patch_size)``.
    tile_info : dict
        Returned by :func:`tile_image`.

    Returns
    -------
    np.ndarray
        Shape ``(C, H, W)`` matching the original (pre-padding) dimensions.
    """
    rows = tile_info["rows"]
    cols = tile_info["cols"]
    stride = tile_info["stride"]
    ps = tile_info["patch_size"]
    C = patches.shape[1]

    pH = (rows - 1) * stride + ps
    pW = (cols - 1) * stride + ps

    canvas = np.zeros((C, pH, pW), dtype=np.float64)
    weight = np.zeros((1, pH, pW), dtype=np.float64)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            y0 = r * stride
            x0 = c * stride
            canvas[:, y0 : y0 + ps, x0 : x0 + ps] += patches[idx].astype(np.float64)
            weight[:, y0 : y0 + ps, x0 : x0 + ps] += 1.0
            idx += 1

    canvas /= np.maximum(weight, 1.0)

    # Crop back to original size
    H = tile_info["orig_H"]
    W = tile_info["orig_W"]
    return canvas[:, :H, :W].astype(np.float32)


# ---------------------------------------------------------------------------
# High-level convenience
# ---------------------------------------------------------------------------

def load_and_prepare(
    path: str,
    patch_size: int = DEFAULT_PATCH_SIZE,
    overlap: int = DEFAULT_PATCH_OVERLAP,
) -> Tuple[np.ndarray, Dict, dict]:
    """Read a pipeline TIFF, select bands, normalize, and tile.

    Returns
    -------
    patches : np.ndarray
        ``(N, 6, patch_size, patch_size)`` float32 normalized patches.
    tile_info : dict
        Metadata for reassembly.
    meta : dict
        Rasterio metadata from the original file.
    """
    data, meta = read_tiff(path)
    data = select_bands(data)
    data = normalize(data)
    patches, tile_info = tile_image(data, patch_size, overlap)
    return patches, tile_info, meta
