# Prithvi-EO 2.0 — Multispectral Segmentation & Cloud Infill

An alternative pipeline path that uses the [IBM-NASA Prithvi-EO 2.0](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M) geospatial foundation model for both **cloud removal** and **land/water segmentation**, operating directly on full multispectral Sentinel-2 GeoTIFFs.

## Pipeline Comparison

```
CURRENT:   Download(6-band) → Coreg → VPint Cloud Remove → RGB/NIR → Upsample → Normalize → YOLO/SAM2 → Boundary …
PRITHVI2:  Download(7-band) → Coreg → Prithvi2 Cloud Infill → Prithvi2 Segment → Boundary …
```

The Prithvi path bypasses four intermediate steps (RGB/NIR creation, upsampling, normalization, and separate cloud removal) by working natively with 6-band multispectral data.

## Model

**Prithvi-EO 2.0-600M** — a Vision Transformer pretrained with masked autoencoder (MAE) on 4.2M HLS V2 samples.

| Parameter | Value |
|-----------|-------|
| Architecture | ViT-MAE with 3D patch embeddings |
| Parameters | ~600M |
| Input bands | Blue (B02), Green (B03), Red (B04), Narrow NIR (B8A), SWIR-1 (B11), SWIR-2 (B12) |
| Patch size | 224 × 224 pixels |
| ViT token | 14 × 14 pixels → 16 × 16 = 256 tokens per frame |
| Temporal frames | Up to 4 (single-frame supported) |

## Directory Structure

```
prithvi2/
├── config.py              # Band mappings, normalization constants, model config
├── model.py               # Model loading from HuggingFace / terratorch
├── data_loader.py         # GeoTIFF I/O, band selection, tiling, reassembly
├── cloud_infill.py        # MAE-based cloud reconstruction
├── segment.py             # Segmentation head + inference + mask_from_folder
├── train_segment.py       # Fine-tuning with existing YOLO masks as labels
├── pipeline_adapter.py    # Interface bridging prithvi2 ↔ pipeline orchestrator
├── requirements.txt
├── README.md
└── weights/               # Model checkpoints (gitignored)
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Cloud infill (standalone)

```python
from cloud_infill import batch_cloud_infill

paths = batch_cloud_infill("/path/to/site/TARGETS/")
# Output: /path/to/site/TARGETS/prithvi2_cloudless/*.tif
```

### 3. Segmentation (standalone)

```python
from segment import Prithvi2Seg

seg = Prithvi2Seg(checkpoint="weights/segment_finetuned.pt")
masks = seg.mask_from_folder("/path/to/site/TARGETS/prithvi2_cloudless/")
# Output: /path/to/site/MASK/*_mask.png
```

### 4. Combined pipeline path

```python
from pipeline_adapter import Prithvi2Pipeline

pipe = Prithvi2Pipeline(seg_checkpoint="weights/segment_finetuned.pt")
result = pipe.run("/path/to/geotools_sites/Fenfushi")
```

### 5. Fine-tuning the segmentation head

```bash
python train_segment.py \
    --sites /path/to/geotools_sites/Fenfushi /path/to/geotools_sites/Nauset \
    --epochs 50 --lr 1e-4 --batch-size 8 \
    --output weights/segment_finetuned.pt
```

### 6. Via the main pipeline CLI

```bash
python littoral_pipeline.py --site Fenfushi --seg-model prithvi2
```

## Band Mapping

The pipeline downloads 7-band Sentinel-2 TIFFs: `[B02, B03, B04, B08, B8A, B11, B12]`.

Prithvi-EO 2.0 expects 6 HLS bands. The mapping is:

| HLS Band | S2 Band | TIFF Index | Description |
|----------|---------|------------|-------------|
| B02 | B02 | 0 | Blue (490 nm) |
| B03 | B03 | 1 | Green (560 nm) |
| B04 | B04 | 2 | Red (665 nm) |
| B05 | B8A | 4 | Narrow NIR (865 nm) |
| B06 | B11 | 5 | SWIR-1 (1610 nm) |
| B07 | B12 | 6 | SWIR-2 (2190 nm) |

B08 (broad NIR, index 3) is skipped — Prithvi uses the narrow NIR (B8A) instead.

## Cloud Infill Approach

The MAE was pretrained to reconstruct randomly masked patches. For cloud infill:

1. **Detect clouds** using SEnSeIv2 (reused from `littoral_cloud_impute`) or a brightness fallback
2. **Convert** pixel-level cloud mask → ViT token-level mask (14×14 per token)
3. **Encode** visible (clear) tokens through the ViT encoder
4. **Decode** to reconstruct masked (cloudy) tokens
5. **Blend** predicted pixels into the original, replacing only cloud-occluded regions
6. **Save** as a cloud-free GeoTIFF preserving all geospatial metadata

## Normalization

Values are in HLS reflectance units (0–10000 scale):

| Band | Mean | Std |
|------|------|-----|
| Blue (B02) | 1087 | 2248 |
| Green (B03) | 1342 | 2179 |
| Red (B04) | 1433 | 2178 |
| Narrow NIR (B8A) | 2734 | 1850 |
| SWIR-1 (B11) | 1958 | 1242 |
| SWIR-2 (B12) | 1363 | 1049 |

## References

- [Prithvi-EO-2.0 GitHub](https://github.com/NASA-IMPACT/Prithvi-EO-2.0)
- [HuggingFace Model Card (600M)](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M)
- [TerraTorch (fine-tuning framework)](https://github.com/IBM/terratorch)
- [arXiv paper](https://arxiv.org/abs/2412.02732)
