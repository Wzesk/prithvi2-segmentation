"""
Prithvi-EO 2.0 configuration: band mappings, normalization, model architecture.

Values sourced from the official HuggingFace model card:
https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M
"""

# ---------------------------------------------------------------------------
# HuggingFace model identifiers
# ---------------------------------------------------------------------------
MODEL_ID_600M = "ibm-nasa-geospatial/Prithvi-EO-2.0-600M"
MODEL_ID_600M_TL = "ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL"
MODEL_ID_300M = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"

DEFAULT_MODEL_ID = MODEL_ID_600M

# ---------------------------------------------------------------------------
# Prithvi-EO 2.0-600M architecture parameters (from config.json)
# ---------------------------------------------------------------------------
MODEL_CONFIG = {
    "img_size": 224,
    "num_frames": 1,           # single-frame by default; override for multi-temporal
    "patch_size": [1, 14, 14], # (temporal, height, width)
    "in_chans": 6,
    "embed_dim": 1280,
    "depth": 32,
    "num_heads": 16,
    "decoder_embed_dim": 512,
    "decoder_depth": 8,
    "decoder_num_heads": 16,
    "mlp_ratio": 4,
    "coords_encoding": [],     # empty for non-TL version
    "coords_scale_learn": False,
    "mask_ratio": 0.75,
    "norm_pix_loss": False,
}

# ---------------------------------------------------------------------------
# Band definitions
#
# Prithvi was pretrained on NASA HLS V2 data.  The six HLS bands map to
# Sentinel-2 bands as follows:
#
#   HLS name   S2 band   Wavelength   Description
#   --------   -------   ----------   -----------
#   B02        B02       490 nm       Blue
#   B03        B03       560 nm       Green
#   B04        B04       665 nm       Red
#   B05        B8A       865 nm       Narrow NIR
#   B06        B11       1610 nm      SWIR-1
#   B07        B12       2190 nm      SWIR-2
# ---------------------------------------------------------------------------

# HLS band names used by the model
HLS_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07"]

# Corresponding Sentinel-2 band names
S2_BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]

# Mapping: Sentinel-2 band name → position in the 6-channel Prithvi input
S2_TO_PRITHVI_INDEX = {
    "B02": 0,
    "B03": 1,
    "B04": 2,
    "B8A": 3,
    "B11": 4,
    "B12": 5,
}

# ---------------------------------------------------------------------------
# Pipeline TIFF band ordering
#
# Current pipeline stores 6-band TIFFs (no B12):
#   [B02, B03, B04, B08, B8A, B11]
#
# After adding B12 it becomes 7-band:
#   [B02, B03, B04, B08, B8A, B11, B12]
#
# Indices into the 7-band TIFF that select the 6 Prithvi input bands:
# ---------------------------------------------------------------------------
TIFF_7BAND_ORDER = ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]
PRITHVI_INDICES_FROM_7BAND = [0, 1, 2, 4, 5, 6]  # skip B08 (index 3)

TIFF_6BAND_ORDER = ["B02", "B03", "B04", "B08", "B8A", "B11"]
PRITHVI_INDICES_FROM_6BAND = [0, 1, 2, 4, 5]  # only 5 bands — cannot fill B12

# ---------------------------------------------------------------------------
# Normalization constants (HLS reflectance units, ×10 000 scale)
# From the model's config.json pretrained_cfg section.
# ---------------------------------------------------------------------------
BAND_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
BAND_STD  = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]

NO_DATA = -9999
NO_DATA_FLOAT = 0.0001

# ---------------------------------------------------------------------------
# Processing defaults
# ---------------------------------------------------------------------------
DEFAULT_PATCH_SIZE = 224       # pixels
DEFAULT_PATCH_OVERLAP = 28     # pixels (one patch-width stride = 14)
DEFAULT_CLOUD_THRESHOLD = 0.3  # fraction of cloudy pixels in patch to mask it
DEFAULT_DEVICE = "auto"        # "auto", "cuda", "cpu"

# ---------------------------------------------------------------------------
# Output directories (created under the site folder)
# ---------------------------------------------------------------------------
CLOUDLESS_DIR = "TARGETS/prithvi2_cloudless"
MASK_DIR = "MASK"

# ---------------------------------------------------------------------------
# Segmentation training defaults
# ---------------------------------------------------------------------------
TRAIN_DEFAULTS = {
    "epochs": 50,
    "lr": 1e-4,
    "batch_size": 8,
    "freeze_encoder": True,
    "unfreeze_last_n": 0,
    "weight_decay": 0.01,
    "warmup_epochs": 5,
    "dice_weight": 0.5,
    "bce_weight": 0.5,
    "val_split": 0.2,
}
