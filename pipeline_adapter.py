"""
Pipeline adapter: bridges standalone prithvi2/ code with the Littoral pipeline.

Exposes ``Prithvi2Pipeline`` which the pipeline orchestrator can instantiate
just like the YOLO or SAM2 segmentation models.  When ``--seg-model prithvi2``
is selected, the orchestrator skips the RGB/NIR creation, upsampling, and
normalization steps and instead routes through:

    TARGETS/ (or coregistered/) → Prithvi2 cloud infill → Prithvi2 segment → MASK/

It also generates the two downstream assets that the skipped steps would have
produced:
  - NORMALIZED/  — NDWI-blended NIR images (for boundary refinement)
  - cloudless_report.csv — affine transform metadata (for geotransformation)
"""

import csv
import os
import logging
from typing import List, Optional

import numpy as np
import rasterio
from PIL import Image

from config import (
    DEFAULT_MODEL_ID, DEFAULT_DEVICE, CLOUDLESS_DIR,
    TIFF_6BAND_ORDER, TIFF_7BAND_ORDER,
)
from cloud_infill import batch_cloud_infill
from segment import Prithvi2Seg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone helpers for downstream asset generation
# ---------------------------------------------------------------------------

def _compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDWI = (Green - NIR) / (Green + NIR) × -1  (land bright, water dark).

    Mirrors ``planetary_s2.compute_ndwi`` so the images look identical.
    """
    eps = 1e-6
    return (green - nir) / (green + nir + eps) * -1


def generate_ndwi_images(
    tiff_folder: str,
    output_folder: str,
) -> List[str]:
    """Create NDWI-blended NIR PNGs from multispectral TIFFs.

    Replicates the NIR image the normal pipeline puts into
    ``NORMALIZED/{stem}_nir_up.png`` so that boundary refinement can
    sample pixel values along transect normals.

    Parameters
    ----------
    tiff_folder : str
        Folder containing the 6- or 7-band GeoTIFFs (``TARGETS/``
        or the Prithvi2 cloudless sub-folder).
    output_folder : str
        Destination directory (typically ``NORMALIZED/``).

    Returns
    -------
    list[str]
        Paths to the saved PNG files.
    """
    os.makedirs(output_folder, exist_ok=True)
    tiffs = sorted(
        f for f in os.listdir(tiff_folder) if f.lower().endswith((".tif", ".tiff"))
    )
    saved: List[str] = []

    for fname in tiffs:
        src_path = os.path.join(tiff_folder, fname)
        stem = os.path.splitext(fname)[0]
        if stem.endswith("_pred"):
            stem = stem[:-5]

        try:
            with rasterio.open(src_path) as src:
                n_bands = src.count
                bands = src.read()  # (C, H, W)

            # Determine Green and NIR band positions
            if n_bands >= 7:
                band_order = TIFF_7BAND_ORDER
            else:
                band_order = TIFF_6BAND_ORDER

            green_idx = band_order.index("B03")
            # Prefer B08 (10 m NIR) if present, fall back to B8A
            nir_idx = (
                band_order.index("B08")
                if "B08" in band_order
                else band_order.index("B8A")
            )

            green = bands[green_idx].astype(np.float32)
            real_nir = bands[nir_idx].astype(np.float32)

            ndwi = _compute_ndwi(green, real_nir)
            blended = (real_nir + ndwi) / 2.0
            bmax = np.max(blended)
            normed = (
                (blended / bmax * 255).astype(np.uint8)
                if bmax > 0
                else blended.astype(np.uint8)
            )

            img = Image.fromarray(np.dstack([normed, normed, normed]))
            out_path = os.path.join(output_folder, f"{stem}_nir_up.png")
            img.save(out_path)
            saved.append(out_path)
        except Exception as exc:
            logger.error("NDWI generation failed for %s: %s", fname, exc)

    logger.info("Generated %d NDWI images in %s", len(saved), output_folder)
    return saved


def generate_cloudless_report(
    tiff_folder: str,
    output_path: str,
) -> str:
    """Write a ``cloudless_report.csv`` from the GeoTIFFs in *tiff_folder*.

    This file is consumed by ``geo_transform.batch_geotransform`` which reads
    ``image_Transform`` to map pixel coordinates to geographic coordinates.

    Parameters
    ----------
    tiff_folder : str
        Folder containing GeoTIFFs (``TARGETS/`` or cloudless sub-folder).
    output_path : str
        Full path for the output CSV (e.g.
        ``TARGETS/cloudless/cloudless_report.csv``).

    Returns
    -------
    str
        *output_path* — the written file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tiffs = sorted(
        f for f in os.listdir(tiff_folder) if f.lower().endswith((".tif", ".tiff"))
    )

    header = [
        "image_name",
        "image_date",
        "cloud_coverage %",
        "original_image_size",
        "cloudless_image_size",
        "image_CRS",
        "image_Transform",
        "band_names",
    ]

    rows: list = []
    for fname in tiffs:
        src_path = os.path.join(tiff_folder, fname)
        stem = os.path.splitext(fname)[0]
        if stem.endswith("_pred"):
            stem = stem[:-5]
        try:
            with rasterio.open(src_path) as src:
                # Use HxW format to match vpint_cloud_impute convention
                # (geo_transform.get_first_transform compares against
                #  rasterio .shape which returns (H, W))
                rows.append([
                    stem,
                    stem[:8] if len(stem) >= 8 else stem,
                    0.0,
                    f"{src.height}x{src.width}",
                    f"{src.height}x{src.width}",
                    str(src.crs),
                    str(src.transform),
                    str(list(src.descriptions) if src.descriptions[0] else
                        [f"B{i+1}" for i in range(src.count)]),
                ])
        except Exception as exc:
            logger.error("Could not read metadata from %s: %s", fname, exc)

    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)

    logger.info("Wrote cloudless_report.csv with %d entries → %s", len(rows), output_path)
    return output_path


class Prithvi2Pipeline:
    """Unified interface for the Prithvi-EO 2.0 alternative pipeline path.

    This class is designed to be instantiated by the pipeline orchestrator and
    called identically to the existing segmentation models.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.
    seg_checkpoint : str or None
        Path to the fine-tuned segmentation head weights.
    device : str
        ``'auto'``, ``'cuda'``, or ``'cpu'``.
    skip_cloud_infill : bool
        If ``True``, skip the cloud-removal step and feed TIFFs directly to
        segmentation.  Useful when cloudless TIFFs already exist.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        seg_checkpoint: Optional[str] = None,
        device: str = DEFAULT_DEVICE,
        skip_cloud_infill: bool = False,
    ):
        self.model_id = model_id
        self.device = device
        self.skip_cloud_infill = skip_cloud_infill
        self._seg = Prithvi2Seg(
            checkpoint=seg_checkpoint, model_id=model_id, device=device,
        )

    # ------------------------------------------------------------------
    # Cloud infill step
    # ------------------------------------------------------------------

    def cloud_infill_folder(
        self,
        input_folder: str,
        output_folder: Optional[str] = None,
    ) -> List[str]:
        """Run Prithvi2 MAE cloud infill on all TIFFs in *input_folder*.

        Parameters
        ----------
        input_folder : str
            Path to ``TARGETS/`` (or ``coregistered/``).
        output_folder : str or None
            Output directory.  Defaults to ``TARGETS/prithvi2_cloudless``.

        Returns
        -------
        list[str]
            Paths to the cloud-free GeoTIFFs.
        """
        return batch_cloud_infill(
            input_folder,
            output_dir=output_folder,
            model_id=self.model_id,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Segmentation step (matches existing mask_from_folder interface)
    # ------------------------------------------------------------------

    def mask_from_folder(self, folder: str) -> List[str]:
        """Generate binary land/water masks for all TIFFs in *folder*.

        This is the drop-in replacement for ``YOLOV8.mask_from_folder()`` and
        ``SAM2Seg.mask_from_folder()``.

        Parameters
        ----------
        folder : str
            Directory containing GeoTIFFs (typically the Prithvi2 cloudless
            output).

        Returns
        -------
        list[str]
            Paths to the saved mask PNGs.
        """
        return self._seg.mask_from_folder(folder)

    # ------------------------------------------------------------------
    # Combined: cloud infill → segmentation
    # ------------------------------------------------------------------

    def run(self, site_path: str) -> dict:
        """Run the full Prithvi2 alternative path for a site.

        Parameters
        ----------
        site_path : str
            Root site directory (e.g. ``geotools_sites/Fenfushi``).

        Returns
        -------
        dict
            ``{'cloudless_paths': [...], 'mask_paths': [...]}``.
        """
        targets_dir = os.path.join(site_path, "TARGETS")
        cloudless_dir = os.path.join(site_path, CLOUDLESS_DIR)

        # Step 1: cloud infill
        if self.skip_cloud_infill:
            seg_input = targets_dir if not os.path.isdir(cloudless_dir) else cloudless_dir
            cloudless_paths = []
            logger.info("Skipping cloud infill — using %s", seg_input)
        else:
            cloudless_paths = self.cloud_infill_folder(targets_dir, cloudless_dir)
            seg_input = cloudless_dir

        # Step 2: segmentation
        mask_paths = self.mask_from_folder(seg_input)

        return {
            "cloudless_paths": cloudless_paths,
            "mask_paths": mask_paths,
        }
