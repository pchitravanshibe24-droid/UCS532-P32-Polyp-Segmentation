"""Standard U-Net architecture with configurable Monte Carlo Dropout blocks."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive [Conv2d -> BatchNorm2d -> ReLU] layers."""

    def __init__(self, in_channels: int, out_channels: int, dropout_prob: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_prob > 0.0:
            layers.append(nn.Dropout2d(p=dropout_prob))
            
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Classic 4-stage U-Net architecture for medical image segmentation with MC Dropout support."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: Optional[List[int]] = None,
        dropout_prob: float = 0.2,
        dropout_locations: Optional[List[str]] = None,
    ):
        """Initialize U-Net.

        Args:
            in_channels: Number of input image channels (default: 3 for RGB).
            out_channels: Number of output channels (default: 1 for binary logits).
            features: List of feature dimensions across 4 encoder stages (default: [64, 128, 256, 512]).
            dropout_prob: Dropout probability for Monte Carlo passes.
            dropout_locations: List of block names where dropout is applied (e.g. ['bottleneck', 'decoder_upper']).
        """
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        if dropout_locations is None:
            dropout_locations = ["bottleneck", "decoder_upper"]

        self.features = features
        self.dropout_prob = dropout_prob
        self.dropout_locations = set(dropout_locations)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1. Encoder (Downsampling)
        prev_channels = in_channels
        for i, feat in enumerate(features):
            use_drop = ("encoder" in self.dropout_locations) and (self.dropout_prob > 0.0)
            self.downs.append(DoubleConv(prev_channels, feat, dropout_prob=self.dropout_prob if use_drop else 0.0))
            prev_channels = feat

        # 2. Bottleneck
        use_bottleneck_drop = ("bottleneck" in self.dropout_locations) and (self.dropout_prob > 0.0)
        bottleneck_channels = features[-1] * 2
        self.bottleneck = DoubleConv(
            features[-1],
            bottleneck_channels,
            dropout_prob=self.dropout_prob if use_bottleneck_drop else 0.0,
        )

        # 3. Decoder (Upsampling)
        rev_features = list(reversed(features))
        prev_dec_channels = bottleneck_channels

        for i, feat in enumerate(rev_features):
            # Upsampling layer
            self.ups.append(nn.ConvTranspose2d(prev_dec_channels, feat, kernel_size=2, stride=2))
            
            # DoubleConv after skip concatenation
            is_upper = (i < len(rev_features) // 2)
            use_dec_drop = (
                ("decoder" in self.dropout_locations)
                or ("decoder_upper" in self.dropout_locations and is_upper)
            ) and (self.dropout_prob > 0.0)
            
            self.ups.append(DoubleConv(feat * 2, feat, dropout_prob=self.dropout_prob if use_dec_drop else 0.0))
            prev_dec_channels = feat

        # 4. Final Output Logit Layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, in_channels, H, W]
            
        Returns:
            Logits tensor [B, out_channels, H, W]
        """
        skip_connections = []

        # Downward path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Invert skips for upward path
        skip_connections = skip_connections[::-1]

        # Upward path
        for idx in range(0, len(self.ups), 2):
            up_conv = self.ups[idx]
            double_conv = self.ups[idx + 1]
            skip = skip_connections[idx // 2]

            x = up_conv(x)

            # Handle odd dimension padding if necessary
            if x.shape[2:] != skip.shape[2:]:
                diff_y = skip.size(2) - x.size(2)
                diff_x = skip.size(3) - x.size(3)
                x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

            concat_skip = torch.cat((skip, x), dim=1)
            x = double_conv(concat_skip)

        return self.final_conv(x)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config: dict) -> UNet:
    """Build UNet model instance from configuration dictionary."""
    model_cfg = config.get("model", {})
    return UNet(
        in_channels=model_cfg.get("in_channels", 3),
        out_channels=model_cfg.get("out_channels", 1),
        features=model_cfg.get("features", [64, 128, 256, 512]),
        dropout_prob=model_cfg.get("dropout_prob", 0.2),
        dropout_locations=model_cfg.get("dropout_locations", ["bottleneck", "decoder_upper"]),
    )
