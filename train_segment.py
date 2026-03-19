"""
Fine-tuning script for Prithvi-EO 2.0 land/water segmentation.

Uses existing YOLO-generated binary masks as training labels and the
corresponding multispectral GeoTIFFs as inputs.  The encoder is frozen
(optionally un-freezing the last N layers) and a lightweight segmentation
head is trained on top.

Usage
-----
    python train_segment.py \\
        --sites /path/to/geotools_sites/Fenfushi /path/to/geotools_sites/Nauset \\
        --epochs 50 --lr 1e-4 --batch-size 8 \\
        --output weights/segment_finetuned.pt
"""

import argparse
import glob
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from config import (
    DEFAULT_MODEL_ID,
    DEFAULT_DEVICE,
    DEFAULT_PATCH_SIZE,
    DEFAULT_PATCH_OVERLAP,
    MODEL_CONFIG,
    TRAIN_DEFAULTS,
)
from model import resolve_device
from segment import load_segmenter, Prithvi2Segmenter
from data_loader import read_tiff, select_bands, normalize, tile_image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ShorelineSegDataset(Dataset):
    """Paired multispectral patches ↔ binary mask patches.

    Directory layout expected per site:
        site_dir/
            TARGETS/          ← multispectral GeoTIFFs (6 or 7 band)
            MASK/             ← binary PNGs (land=255, water=0)

    Masks are at a potentially different resolution (upsampled) so they are
    downscaled to match the TIF spatial dimensions before patch extraction.
    """

    def __init__(
        self,
        site_dirs: List[str],
        patch_size: int = DEFAULT_PATCH_SIZE,
        overlap: int = DEFAULT_PATCH_OVERLAP,
        tiff_subdir: str = "TARGETS",
        mask_subdir: str = "MASK",
    ):
        self.patch_size = patch_size
        self.overlap = overlap
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []

        for site in site_dirs:
            self._load_site(site, tiff_subdir, mask_subdir)

        logger.info(
            "Dataset: %d patch pairs from %d sites", len(self.samples), len(site_dirs)
        )

    def _load_site(self, site_dir: str, tiff_sub: str, mask_sub: str):
        tiff_dir = os.path.join(site_dir, tiff_sub)
        mask_dir = os.path.join(site_dir, mask_sub)
        if not os.path.isdir(tiff_dir) or not os.path.isdir(mask_dir):
            logger.warning("Missing TARGETS or MASK in %s — skipping", site_dir)
            return

        tiff_files = {
            os.path.splitext(f)[0]: os.path.join(tiff_dir, f)
            for f in os.listdir(tiff_dir)
            if f.lower().endswith((".tif", ".tiff"))
        }
        mask_files = {
            self._stem_from_mask(f): os.path.join(mask_dir, f)
            for f in os.listdir(mask_dir)
            if f.lower().endswith(".png")
        }

        # Match by date stem
        matched = 0
        for stem, tiff_path in tiff_files.items():
            mask_path = mask_files.get(stem)
            if mask_path is None:
                continue
            try:
                self._extract_patches(tiff_path, mask_path)
                matched += 1
            except Exception as exc:
                logger.debug("Skipping %s: %s", stem, exc)

        logger.info("Site %s: %d matched TIF↔MASK pairs", os.path.basename(site_dir), matched)

    @staticmethod
    def _stem_from_mask(filename: str) -> str:
        """Extract date stem from mask filename.

        The pipeline names masks like ``20200101_nir_x4_mask.png`` or just
        ``20200101_mask.png``.  Strip suffixes to get the date stem.
        """
        stem = os.path.splitext(filename)[0]
        for suffix in ("_mask", "_nir_x4_mask", "_nir_mask"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        return stem

    def _extract_patches(self, tiff_path: str, mask_path: str):
        from PIL import Image

        data, _ = read_tiff(tiff_path)
        data = select_bands(data)
        _, tH, tW = data.shape

        mask_pil = Image.open(mask_path).convert("L")
        # Resize mask to match TIF resolution
        mask_pil = mask_pil.resize((tW, tH), Image.NEAREST)
        mask = np.array(mask_pil, dtype=np.float32) / 255.0  # 0 or 1

        # Normalize spectral data
        normed = normalize(data)

        # Tile both
        spec_patches, tile_info = tile_image(normed, self.patch_size, self.overlap)
        mask_3d = mask[np.newaxis, :, :]  # (1, H, W)
        mask_patches, _ = tile_image(mask_3d, self.patch_size, self.overlap)

        for sp, mp in zip(spec_patches, mask_patches):
            # Skip patches that are nearly all water or all land (less informative)
            land_frac = mp.mean()
            if 0.02 < land_frac < 0.98:
                self.samples.append((sp, mp[0]))  # mp[0] → (ps, ps)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        spec, mask = self.samples[idx]
        return torch.tensor(spec, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class DiceBCELoss(nn.Module):
    """Combined Binary Cross-Entropy + Dice loss."""

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dw = dice_weight
        self.bw = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
        # Dice
        smooth = 1.0
        pflat = probs.reshape(-1)
        tflat = targets.reshape(-1)
        intersection = (pflat * tflat).sum()
        dice = 1 - (2 * intersection + smooth) / (pflat.sum() + tflat.sum() + smooth)
        return self.bw * bce + self.dw * dice


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    iou = tp / max(tp + fp + fn, 1e-8)
    dice = 2 * tp / max(2 * tp + fp + fn, 1e-8)
    acc = (preds == targets).float().mean().item()
    return {"iou": iou, "dice": dice, "accuracy": acc}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    segmenter: Prithvi2Segmenter,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = TRAIN_DEFAULTS["epochs"],
    lr: float = TRAIN_DEFAULTS["lr"],
    weight_decay: float = TRAIN_DEFAULTS["weight_decay"],
    warmup_epochs: int = TRAIN_DEFAULTS["warmup_epochs"],
    dice_weight: float = TRAIN_DEFAULTS["dice_weight"],
    bce_weight: float = TRAIN_DEFAULTS["bce_weight"],
    device: str = DEFAULT_DEVICE,
    output_path: str = "weights/segment_finetuned.pt",
) -> Dict:
    """Train the segmentation head.

    Returns
    -------
    dict
        Training history with loss / metric curves.
    """
    dev = resolve_device(device)
    segmenter.to(dev)
    segmenter.encoder.eval()  # keep encoder in eval mode (BN, dropout)
    segmenter.head.train()

    criterion = DiceBCELoss(dice_weight, bce_weight)
    optimizer = torch.optim.AdamW(
        segmenter.head.parameters(), lr=lr, weight_decay=weight_decay,
    )

    # Simple linear warmup + cosine decay
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_iou = -1.0
    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}

    for epoch in range(epochs):
        # --- Train ---
        segmenter.head.train()
        running_loss = 0.0
        n_batches = 0

        for spec, mask in train_loader:
            spec = spec.to(dev)
            mask = mask.to(dev).unsqueeze(1)  # (B, 1, H, W)

            # Forward through frozen encoder + trainable head
            x5d = spec.unsqueeze(2)  # (B, C, 1, H, W)
            with torch.no_grad():
                latent, _, _ = segmenter.encoder.forward_encoder(x5d, mask_ratio=0.0)
            expected_tokens = (spec.shape[2] // 14) * (spec.shape[3] // 14)
            if latent.shape[1] > expected_tokens:
                latent = latent[:, -expected_tokens:]

            logits = segmenter.head(latent, target_h=spec.shape[2], target_w=spec.shape[3])
            loss = criterion(logits, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_train_loss = running_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train_loss)

        # --- Validate ---
        segmenter.head.eval()
        val_loss = 0.0
        val_metrics = {"iou": 0.0, "dice": 0.0, "accuracy": 0.0}
        n_val = 0

        with torch.no_grad():
            for spec, mask in val_loader:
                spec = spec.to(dev)
                mask = mask.to(dev).unsqueeze(1)

                x5d = spec.unsqueeze(2)
                latent, _, _ = segmenter.encoder.forward_encoder(x5d, mask_ratio=0.0)
                expected_tokens = (spec.shape[2] // 14) * (spec.shape[3] // 14)
                if latent.shape[1] > expected_tokens:
                    latent = latent[:, -expected_tokens:]

                logits = segmenter.head(latent, target_h=spec.shape[2], target_w=spec.shape[3])
                val_loss += criterion(logits, mask).item()

                m = compute_metrics(logits, mask)
                for k in val_metrics:
                    val_metrics[k] += m[k]
                n_val += 1

        avg_val_loss = val_loss / max(n_val, 1)
        for k in val_metrics:
            val_metrics[k] /= max(n_val, 1)

        history["val_loss"].append(avg_val_loss)
        history["val_iou"].append(val_metrics["iou"])
        history["val_dice"].append(val_metrics["dice"])

        scheduler.step()

        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  IoU=%.4f  Dice=%.4f",
            epoch + 1, epochs, avg_train_loss, avg_val_loss,
            val_metrics["iou"], val_metrics["dice"],
        )

        # Save best
        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            torch.save(segmenter.head.state_dict(), output_path)
            logger.info("  ↳ Saved best head (IoU=%.4f) → %s", best_val_iou, output_path)

    logger.info("Training complete.  Best val IoU = %.4f", best_val_iou)
    return history


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Prithvi2 segmentation head")
    parser.add_argument("--sites", nargs="+", required=True, help="Site directories (each with TARGETS/ and MASK/)")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--epochs", type=int, default=TRAIN_DEFAULTS["epochs"])
    parser.add_argument("--lr", type=float, default=TRAIN_DEFAULTS["lr"])
    parser.add_argument("--batch-size", type=int, default=TRAIN_DEFAULTS["batch_size"])
    parser.add_argument("--weight-decay", type=float, default=TRAIN_DEFAULTS["weight_decay"])
    parser.add_argument("--warmup-epochs", type=int, default=TRAIN_DEFAULTS["warmup_epochs"])
    parser.add_argument("--val-split", type=float, default=TRAIN_DEFAULTS["val_split"])
    parser.add_argument("--output", default="weights/segment_finetuned.pt", help="Path to save best head weights")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_PATCH_OVERLAP)
    parser.add_argument("--tiff-subdir", default="TARGETS", help="Subdirectory for input TIFFs")
    parser.add_argument("--mask-subdir", default="MASK", help="Subdirectory for mask PNGs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Build dataset
    dataset = ShorelineSegDataset(
        args.sites,
        patch_size=args.patch_size,
        overlap=args.overlap,
        tiff_subdir=args.tiff_subdir,
        mask_subdir=args.mask_subdir,
    )

    if len(dataset) == 0:
        logger.error("No training patches found — check site directories.")
        sys.exit(1)

    # Split
    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    logger.info("Train: %d patches  Val: %d patches", n_train, n_val)

    # Build model
    segmenter = load_segmenter(
        checkpoint=None,
        model_id=args.model_id,
        device=args.device,
        freeze_encoder=True,
    )

    # Train
    history = train(
        segmenter,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        device=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
