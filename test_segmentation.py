"""
Test Prithvi2 segmentation on validation images from the prithvi2_dataset.

Loads the fine-tuned segmentation head, runs inference on val images,
computes IoU/Dice metrics, and saves visual results (RGB + predicted mask +
ground truth) to test_results/.

Usage:
    python test_segmentation.py
    python test_segmentation.py --n-samples 20 --checkpoint weights/segment_finetuned.pt
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch

# Workaround: cuDNN conv3d fails on some driver/hardware combos (V100 + PyTorch 2.7)
torch.backends.cudnn.enabled = False

import rasterio
from PIL import Image

from config import DEFAULT_MODEL_ID, DEFAULT_DEVICE, BAND_MEAN, BAND_STD
from model import resolve_device, load_mae
from segment import load_segmenter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_DIR = "/home/walter_littor_al/seg-training/prithvi2_dataset"
OUTPUT_DIR = "/home/walter_littor_al/prithvi2/test_results"


def compute_metrics(pred: np.ndarray, gt: np.ndarray):
    """Compute IoU, Dice, accuracy between binary masks (0/1)."""
    tp = ((pred == 1) & (gt == 1)).sum()
    fp = ((pred == 1) & (gt == 0)).sum()
    fn = ((pred == 0) & (gt == 1)).sum()
    tn = ((pred == 0) & (gt == 0)).sum()
    iou = tp / max(tp + fp + fn, 1)
    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {"iou": float(iou), "dice": float(dice), "accuracy": float(acc)}


def normalize_band_to_uint8(band: np.ndarray) -> np.ndarray:
    """Percentile stretch a single band to 0-255."""
    valid = band[band > 0]
    if len(valid) == 0:
        return np.zeros_like(band, dtype=np.uint8)
    p2, p98 = np.percentile(valid, [2, 98])
    if p98 <= p2:
        p2, p98 = 0, 1
    stretched = np.clip((band.astype(np.float32) - p2) / (p98 - p2) * 255, 0, 255)
    return stretched.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Test Prithvi2 segmentation")
    parser.add_argument("--checkpoint", default="weights/segment_finetuned.pt")
    parser.add_argument("--dataset", default=DATASET_DIR)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--n-samples", type=int, default=15, help="Number of val samples to test")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load model
    logger.info("Loading segmenter with checkpoint: %s", args.checkpoint)
    segmenter = load_segmenter(args.checkpoint, device=args.device)
    dev = resolve_device(args.device)
    logger.info("Model loaded on %s", dev)

    # Gather val samples
    tiff_dir = os.path.join(args.dataset, "tiffs", "val")
    mask_dir = os.path.join(args.dataset, "masks", "val")
    stems = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(tiff_dir)
        if f.endswith(".tif")
    )

    if args.n_samples < len(stems):
        # Pick evenly spaced samples for diversity
        indices = np.linspace(0, len(stems) - 1, args.n_samples, dtype=int)
        stems = [stems[i] for i in indices]

    logger.info("Testing on %d validation images", len(stems))

    all_metrics = []
    for stem in stems:
        tiff_path = os.path.join(tiff_dir, f"{stem}.tif")
        mask_path = os.path.join(mask_dir, f"{stem}.tif")

        # Read 6-band tiff
        with rasterio.open(tiff_path) as src:
            data = src.read().astype(np.float32)  # (6, H, W)

        # Read ground truth mask
        with rasterio.open(mask_path) as src:
            gt_mask = src.read(1)  # (H, W) uint8, 0 or 1

        # Normalize using Prithvi stats
        mean = np.array(BAND_MEAN)[:, None, None]
        std = np.array(BAND_STD)[:, None, None]
        normed = (data - mean) / std

        # Run inference
        x = torch.tensor(normed, dtype=torch.float32).unsqueeze(0).to(dev)  # (1, 6, H, W)
        with torch.no_grad():
            prob = segmenter.predict(x)  # (1, 1, H, W)
        prob_np = prob.cpu().numpy()[0, 0]  # (H, W)

        # Threshold
        pred_mask = (prob_np >= args.threshold).astype(np.uint8)

        # Metrics
        m = compute_metrics(pred_mask, gt_mask)
        m["stem"] = stem
        all_metrics.append(m)
        logger.info("  %s  IoU=%.3f  Dice=%.3f  Acc=%.3f", stem, m["iou"], m["dice"], m["accuracy"])

        # Save visual result: RGB | Prediction | Ground Truth side by side
        # RGB from bands B04(idx2), B03(idx1), B02(idx0)
        rgb = np.stack([
            normalize_band_to_uint8(data[2]),  # Red (B04)
            normalize_band_to_uint8(data[1]),  # Green (B03)
            normalize_band_to_uint8(data[0]),  # Blue (B02)
        ], axis=-1)

        h, w = gt_mask.shape
        pred_vis = np.stack([pred_mask * 255, pred_mask * 255, pred_mask * 255], axis=-1).astype(np.uint8)
        gt_vis = np.stack([gt_mask * 255, gt_mask * 255, gt_mask * 255], axis=-1).astype(np.uint8)

        # Overlay: red where prediction differs from GT
        diff = np.zeros((h, w, 3), dtype=np.uint8)
        fp_mask = (pred_mask == 1) & (gt_mask == 0)  # false positive → red
        fn_mask = (pred_mask == 0) & (gt_mask == 1)  # false negative → blue
        tp_mask = (pred_mask == 1) & (gt_mask == 1)  # true positive → green
        diff[fp_mask] = [255, 0, 0]
        diff[fn_mask] = [0, 0, 255]
        diff[tp_mask] = [0, 255, 0]

        # Combine: RGB | Prediction | GT | Diff
        combined = np.concatenate([rgb, pred_vis, gt_vis, diff], axis=1)
        Image.fromarray(combined).save(os.path.join(args.output, f"{stem}.png"))

    # Summary
    ious = [m["iou"] for m in all_metrics]
    dices = [m["dice"] for m in all_metrics]
    accs = [m["accuracy"] for m in all_metrics]

    summary = {
        "n_samples": len(all_metrics),
        "mean_iou": float(np.mean(ious)),
        "mean_dice": float(np.mean(dices)),
        "mean_accuracy": float(np.mean(accs)),
        "std_iou": float(np.std(ious)),
        "min_iou": float(np.min(ious)),
        "max_iou": float(np.max(ious)),
        "per_sample": all_metrics,
    }

    summary_path = os.path.join(args.output, "test_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Test Results ({len(all_metrics)} samples)")
    print(f"{'='*50}")
    print(f"  Mean IoU:      {summary['mean_iou']:.4f} ± {summary['std_iou']:.4f}")
    print(f"  Mean Dice:     {summary['mean_dice']:.4f}")
    print(f"  Mean Accuracy: {summary['mean_accuracy']:.4f}")
    print(f"  IoU range:     [{summary['min_iou']:.4f}, {summary['max_iou']:.4f}]")
    print(f"\nResults saved to: {args.output}")
    print(f"  test_summary.json  - metrics")
    print(f"  *.png              - visual comparisons (RGB | Pred | GT | Diff)")


if __name__ == "__main__":
    main()
