"""
Prithvi-EO 2.0 model loading utilities.

Supports loading the full MAE model (for cloud infill / reconstruction)
and building the encoder backbone (for segmentation fine-tuning).
"""

import os
import logging
from typing import Optional, Dict, Any

import torch

from config import (
    DEFAULT_MODEL_ID,
    MODEL_CONFIG,
    BAND_MEAN,
    BAND_STD,
    DEFAULT_DEVICE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def resolve_device(device: str = DEFAULT_DEVICE) -> torch.device:
    """Return a ``torch.device``, resolving ``'auto'`` to CUDA if available."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# ---------------------------------------------------------------------------
# PrithviMAE loading (full encoder + decoder — used for reconstruction)
# ---------------------------------------------------------------------------

def _import_prithvi_mae():
    """Import the ``PrithviMAE`` class, trying terratorch first."""
    try:
        from terratorch.models.backbones.prithvi_mae import PrithviMAE
        return PrithviMAE
    except ImportError:
        pass

    # Fallback: download prithvi_mae.py from the HuggingFace repo and import
    try:
        from huggingface_hub import hf_hub_download
        import importlib.util

        local_path = hf_hub_download(
            repo_id=DEFAULT_MODEL_ID,
            filename="prithvi_mae.py",
        )
        spec = importlib.util.spec_from_file_location("prithvi_mae", local_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.PrithviMAE
    except Exception as exc:
        raise ImportError(
            "Could not import PrithviMAE. Install terratorch "
            "(pip install terratorch) or huggingface_hub."
        ) from exc


def load_mae(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    num_frames: int = 1,
    device: str = DEFAULT_DEVICE,
    mask_ratio: Optional[float] = None,
) -> torch.nn.Module:
    """Load the full Prithvi-EO 2.0 MAE (encoder + decoder).

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.
    num_frames : int
        Number of temporal frames the model should expect.
    device : str
        ``'auto'``, ``'cuda'``, or ``'cpu'``.
    mask_ratio : float or None
        Override pretraining mask ratio (default from config).

    Returns
    -------
    torch.nn.Module
        The loaded MAE model in eval mode on *device*.
    """
    from huggingface_hub import hf_hub_download

    dev = resolve_device(device)
    PrithviMAE = _import_prithvi_mae()

    # Build config for this instantiation
    cfg = dict(MODEL_CONFIG)
    cfg["num_frames"] = num_frames
    if mask_ratio is not None:
        cfg["mask_ratio"] = mask_ratio

    model = PrithviMAE(**cfg)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("PrithviMAE created with %s parameters", f"{total_params:,}")

    # Download checkpoint
    ckpt_name = _checkpoint_filename(model_id)
    ckpt_path = hf_hub_download(repo_id=model_id, filename=ckpt_name)
    state_dict = torch.load(ckpt_path, map_location=dev, weights_only=True)

    # Discard fixed positional embeddings — they are regenerated for the
    # actual input size and num_frames.
    for k in list(state_dict.keys()):
        if "pos_embed" in k:
            del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    logger.info("Loaded checkpoint from %s", ckpt_path)

    model.to(dev)
    model.eval()
    return model


def _checkpoint_filename(model_id: str) -> str:
    """Derive the ``.pt`` checkpoint filename from the model id."""
    # Convention: Prithvi_EO_V2_600M.pt, Prithvi_EO_V2_300M.pt, etc.
    tag = model_id.rsplit("/", 1)[-1]  # e.g. "Prithvi-EO-2.0-600M"
    tag = tag.replace("Prithvi-EO-2.0-", "")  # "600M"
    return f"Prithvi_EO_V2_{tag}.pt"


# ---------------------------------------------------------------------------
# Backbone loading (encoder only — used for segmentation fine-tuning)
# ---------------------------------------------------------------------------

def load_backbone(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    device: str = DEFAULT_DEVICE,
    pretrained: bool = True,
) -> torch.nn.Module:
    """Load only the encoder backbone via ``terratorch``.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier (determines architecture variant).
    device : str
        ``'auto'``, ``'cuda'``, or ``'cpu'``.
    pretrained : bool
        Whether to load pretrained weights.

    Returns
    -------
    torch.nn.Module
        ViT encoder backbone.
    """
    try:
        from terratorch.registry import BACKBONE_REGISTRY
    except ImportError:
        raise ImportError(
            "terratorch is required for backbone loading: pip install terratorch"
        )

    dev = resolve_device(device)

    # Map model_id → terratorch registry name
    arch_name = _registry_name(model_id)
    backbone = BACKBONE_REGISTRY.build(arch_name, pretrained=pretrained)
    backbone.to(dev)
    backbone.eval()
    logger.info("Loaded backbone '%s' on %s", arch_name, dev)
    return backbone


def _registry_name(model_id: str) -> str:
    """Map HuggingFace model id to a terratorch backbone registry name."""
    mapping = {
        "ibm-nasa-geospatial/Prithvi-EO-2.0-600M": "prithvi_eo_v2_600",
        "ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL": "prithvi_eo_v2_600_TL",
        "ibm-nasa-geospatial/Prithvi-EO-2.0-300M": "prithvi_eo_v2_300",
        "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL": "prithvi_eo_v2_300_TL",
    }
    name = mapping.get(model_id)
    if name is None:
        raise ValueError(f"Unknown model_id: {model_id}")
    return name


# ---------------------------------------------------------------------------
# Normalization helpers (re-exported for convenience)
# ---------------------------------------------------------------------------

def get_normalization() -> Dict[str, Any]:
    """Return dict with ``mean`` and ``std`` lists for the 6 Prithvi bands."""
    return {"mean": list(BAND_MEAN), "std": list(BAND_STD)}
