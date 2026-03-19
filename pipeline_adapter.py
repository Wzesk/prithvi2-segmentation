"""
Pipeline adapter: bridges standalone prithvi2/ code with the Littoral pipeline.

Exposes ``Prithvi2Pipeline`` which the pipeline orchestrator can instantiate
just like the YOLO or SAM2 segmentation models.  When ``--seg-model prithvi2``
is selected, the orchestrator skips the RGB/NIR creation, upsampling, and
normalization steps and instead routes through:

    TARGETS/ (or coregistered/) → Prithvi2 cloud infill → Prithvi2 segment → MASK/
"""

import os
import logging
from typing import List, Optional

from config import DEFAULT_MODEL_ID, DEFAULT_DEVICE, CLOUDLESS_DIR
from cloud_infill import batch_cloud_infill
from segment import Prithvi2Seg

logger = logging.getLogger(__name__)


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
