"""Loss functions for polyp segmentation: BCEWithLogitsLoss and Soft Dice Loss."""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import SegmentationOutput


class SoftDiceLoss(nn.Module):
    """Numerically stable Soft Dice Loss with Laplace smoothing."""

    def __init__(self, smooth: float = 1.0e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Soft Dice Loss.
        
        Args:
            probs: Sigmoid probability tensor [B, 1, H, W] in range [0, 1].
            targets: Binary ground truth tensor [B, 1, H, W] in {0, 1}.
            
        Returns:
            Scalar Dice loss (1 - Dice).
        """
        # Flatten spatial dimensions per sample in the batch
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = torch.sum(probs_flat * targets_flat, dim=1)
        cardinality = torch.sum(probs_flat, dim=1) + torch.sum(targets_flat, dim=1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return torch.mean(1.0 - dice)


class CombinedLoss(nn.Module):
    """Weighted sum of BCEWithLogitsLoss and Soft Dice Loss."""

    def __init__(self, lambda_bce: float = 0.5, lambda_dice: float = 0.5, dice_smooth: float = 1.0e-6):
        super().__init__()
        self.lambda_bce = lambda_bce
        self.lambda_dice = lambda_dice
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.dice_fn = SoftDiceLoss(smooth=dice_smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss from unthresholded logits and ground truth.
        
        Args:
            logits: Raw network output [B, 1, H, W].
            targets: Binary ground truth [B, 1, H, W].
            
        Returns:
            Tuple of:
                - total_loss: Scalar torch.Tensor with attached gradients.
                - loss_components: Dictionary with detached float values for logging.
        """
        bce_loss = self.bce_fn(logits, targets)
        probs = torch.sigmoid(logits)
        dice_loss = self.dice_fn(probs, targets)

        total_loss = self.lambda_bce * bce_loss + self.lambda_dice * dice_loss

        components = {
            "loss_total": float(total_loss.item()),
            "loss_bce": float(bce_loss.item()),
            "loss_dice": float(dice_loss.item()),
        }
        return total_loss, components


class BCEIoULoss(nn.Module):
    """BCE plus soft spatial IoU, summed across all supervised outputs."""

    def forward(self, output, targets):
        predictions = output.logits if isinstance(output, SegmentationOutput) else (output,)
        bce_terms, iou_terms = [], []
        for logits in predictions:
            bce_terms.append(F.binary_cross_entropy_with_logits(logits, targets))
            probabilities = logits.sigmoid()
            intersection = (probabilities * targets).sum(dim=(-2, -1))
            union = (probabilities + targets).sum(dim=(-2, -1)) - intersection
            iou_terms.append((1 - (intersection + 1) / (union + 1)).mean())
        bce = torch.stack(bce_terms).sum()
        iou = torch.stack(iou_terms).sum()
        total = bce + iou
        return total, {"loss_total": total.item(), "loss_bce": bce.item(), "loss_iou": iou.item()}


def build_loss(config: dict) -> nn.Module:
    """Build CombinedLoss from configuration dictionary."""
    loss_cfg = config.get("loss", {})
    if loss_cfg.get("type") == "bce_iou" or config.get("model", {}).get("architecture") == "uacanet":
        return BCEIoULoss()
    return CombinedLoss(
        lambda_bce=loss_cfg.get("lambda_bce", 0.5),
        lambda_dice=loss_cfg.get("lambda_dice", 0.5),
        dice_smooth=loss_cfg.get("dice_smooth", 1.0e-6),
    )
