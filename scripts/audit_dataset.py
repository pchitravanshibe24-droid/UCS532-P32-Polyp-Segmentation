import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.data.pairing import find_image_mask_pairs
from src.utils import ensure_dir, load_config


def create_contact_sheet(
    pairs: List[dict],
    output_path: Path,
    max_samples: int = 6
) -> None:
    """Generate and save a visual contact sheet of image, mask, and overlay triplets.
    
    Args:
        pairs: List of valid pair audit dictionaries.
        output_path: Destination PNG file path.
        max_samples: Number of pairs to display (rows).
    """
    if not pairs:
        return
        
    num_samples = min(len(pairs), max_samples)
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)
        
    for i in range(num_samples):
        p = pairs[i]
        img_bgr = cv2.imread(p["image_path"])
        mask_gray = cv2.imread(p["mask_path"], cv2.IMREAD_GRAYSCALE)
        
        if img_bgr is None or mask_gray is None:
            continue
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        bin_mask = (mask_gray > 127).astype(np.uint8)
        
        # Create translucent red overlay for polyp
        overlay = img_rgb.copy()
        overlay[bin_mask == 1, 0] = np.clip(overlay[bin_mask == 1, 0] * 0.5 + 127, 0, 255)
        overlay[bin_mask == 1, 1] = np.clip(overlay[bin_mask == 1, 1] * 0.5, 0, 255)
        overlay[bin_mask == 1, 2] = np.clip(overlay[bin_mask == 1, 2] * 0.5, 0, 255)
        
        # Plot Image
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f"Image: {p['pair_id']}", fontsize=9)
        axes[i, 0].axis("off")
        
        # Plot Mask
        axes[i, 1].imshow(bin_mask, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title(f"Mask (FG: {p['foreground_ratio']*100:.1f}%)", fontsize=9)
        axes[i, 1].axis("off")
        
        # Plot Overlay
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Ground Truth Overlay", fontsize=9)
        axes[i, 2].axis("off")
        
    plt.tight_layout()
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Contact sheet saved to: {output_path}")


def run_audit(
    images_dir: Path,
    masks_dir: Path,
    report_output: Path,
    contact_sheet_output: Optional[Path] = None,
    mask_threshold: int = 127
) -> dict:
    """Run full dataset audit and print summary report."""
    print("=" * 70)
    print("Running Dataset Pairing & Integrity Audit")
    print(f"Images directory: {images_dir}")
    print(f"Masks directory:  {masks_dir}")
    print("=" * 70)
    
    valid_pairs, summary = find_image_mask_pairs(images_dir, masks_dir, mask_threshold)
    
    # Save JSON report
    ensure_dir(report_output.parent)
    with open(report_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("\n--- Audit Summary ---")
    print(f"Total Images Found:        {summary['total_images_discovered']}")
    print(f"Total Masks Found:         {summary['total_masks_discovered']}")
    print(f"Valid Matched Pairs:       {summary['valid_pairs_count']}")
    print(f"Unmatched Images:          {summary['unmatched_images_count']}")
    print(f"Unmatched Masks:           {summary['unmatched_masks_count']}")
    print(f"Corrupt/Mismatched Pairs:  {summary['corrupt_or_mismatched_count']}")
    
    if valid_pairs:
        widths = [p["width"] for p in valid_pairs]
        heights = [p["height"] for p in valid_pairs]
        fg_ratios = [p["foreground_ratio"] for p in valid_pairs]
        empty_count = sum(1 for p in valid_pairs if p["is_empty_mask"])
        
        print(f"Resolution Range:          {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
        print(f"Polyp Area Fraction (mean):{np.mean(fg_ratios)*100:.2f}% (range: {np.min(fg_ratios)*100:.1f}% - {np.max(fg_ratios)*100:.1f}%)")
        print(f"Empty Masks Count:         {empty_count} ({empty_count / len(valid_pairs) * 100:.1f}%)")
        
        if contact_sheet_output:
            create_contact_sheet(valid_pairs, contact_sheet_output)
            
    if summary["unmatched_images_count"] > 0:
        print("\n[WARNING] Unmatched images detected:")
        for img in summary["unmatched_images"][:5]:
            print(f"  - {img}")
            
    if summary["unmatched_masks_count"] > 0:
        print("\n[WARNING] Unmatched masks detected:")
        for m in summary["unmatched_masks"][:5]:
            print(f"  - {m}")
            
    print(f"\nAudit report saved to: {report_output}")
    print("=" * 70)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit dataset integrity and pairing.")
    parser.add_argument("--data-root", type=str, default=None, help="Root folder containing images/ and masks/.")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path.")
    parser.add_argument("--output-report", type=str, default="outputs/metrics/audit_report.json")
    parser.add_argument("--contact-sheet", type=str, default="outputs/figures/dataset_contact_sheet.png")
    args = parser.parse_args()
    
    if args.config:
        cfg = load_config(args.config)
        images_dir = Path(cfg["paths"]["images_dir"])
        masks_dir = Path(cfg["paths"]["masks_dir"])
        report_path = Path(cfg["paths"].get("metrics_dir", "outputs/metrics")) / "audit_report.json"
        contact_path = Path(cfg["paths"].get("figures_dir", "outputs/figures")) / "dataset_contact_sheet.png"
    elif args.data_root:
        root = Path(args.data_root)
        images_dir = root / "images"
        masks_dir = root / "masks"
        report_path = Path(args.output_report)
        contact_path = Path(args.contact_sheet)
    else:
        # Default to synthetic
        images_dir = Path("data/synthetic/images")
        masks_dir = Path("data/synthetic/masks")
        report_path = Path(args.output_report)
        contact_path = Path(args.contact_sheet)
        
    run_audit(images_dir, masks_dir, report_path, contact_path)
