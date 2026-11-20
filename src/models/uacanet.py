"""Compact implementation of Kim, Lee and Kim's UACANet (ACM MM 2021)."""

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


class SegmentationOutput(NamedTuple):
    logits: tuple[torch.Tensor, ...]
    uncertainty: torch.Tensor


def resize(tensor, size):
    return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)


def conv(inputs, outputs, kernel=3, *, stride=1, dilation=1, activation=True):
    layers = [
        nn.Conv2d(inputs, outputs, kernel, stride, dilation=dilation,
                  padding=tuple(k // 2 * dilation for k in (kernel if isinstance(kernel, tuple) else (kernel, kernel))),
                  bias=False),
        nn.GroupNorm(1, outputs),
    ]
    if activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def region_maps(probability):
    """Return foreground, background and uncertain maps in that order."""
    offset = probability - 0.5
    return torch.cat((offset.clamp_min(0), (-offset).clamp_min(0), 0.5 - offset.abs()), dim=1)


def aggregate_context(features, regions):
    """Compute one weighted sum of pixel feature vectors per region."""
    return torch.bmm(regions.flatten(2), features.flatten(2).transpose(1, 2))


class AxialAttention(nn.Module):
    def __init__(self, channels, along_height):
        super().__init__()
        projected = max(1, channels // 8)
        self.query = nn.Conv2d(channels, projected, 1)
        self.key = nn.Conv2d(channels, projected, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.along_height = along_height
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, features):
        # Move the attended axis last and fold the orthogonal axis into channels.
        x = features.transpose(-1, -2) if self.along_height else features
        query = self.query(x).flatten(1, 2).transpose(1, 2)
        key = self.key(x).flatten(1, 2)
        value = self.value(x).flatten(1, 2)
        weights = (query @ key).softmax(dim=-1)
        attended = (value @ weights.transpose(1, 2)).reshape_as(x)
        if self.along_height:
            attended = attended.transpose(-1, -2)
        return features + self.scale * attended


class ParallelAxialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.height = AxialAttention(channels, along_height=True)
        self.width = AxialAttention(channels, along_height=False)

    def forward(self, features):
        return self.height(features) + self.width(features)


class PAAEncoder(nn.Module):
    def __init__(self, inputs, channels):
        super().__init__()
        self.projection = conv(inputs, channels, 1, activation=False)
        self.branches = nn.ModuleList([
            nn.Sequential(
                conv(inputs, channels, 1, activation=False),
                conv(channels, channels, (1, kernel), activation=False),
                conv(channels, channels, (kernel, 1), activation=False),
                ParallelAxialAttention(channels),
                conv(channels, channels, dilation=kernel, activation=False),
            )
            for kernel in (3, 5, 7)
        ])
        self.merge = conv(4 * channels, channels, activation=False)
        self.residual = conv(inputs, channels, 1, activation=False)

    def forward(self, features):
        branches = [self.projection(features), *(branch(features) for branch in self.branches)]
        return F.relu(self.merge(torch.cat(branches, dim=1)) + self.residual(features))


class PAADecoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fusion = nn.Sequential(
            conv(3 * channels, channels, activation=False),
            ParallelAxialAttention(channels),
            *(conv(channels, channels, activation=False) for _ in range(3)),
        )
        self.saliency = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, pyramid):
        size = pyramid[0].shape[-2:]
        features = self.fusion(torch.cat([resize(x, size) for x in pyramid], dim=1))
        return features, self.saliency(features)


class UACA(nn.Module):
    def __init__(self, channels, use_uncertain_context=True):
        super().__init__()
        self.use_uncertain_context = use_uncertain_context
        self.query = nn.Sequential(conv(2 * channels, channels), conv(channels, channels))
        self.key = nn.Sequential(nn.Linear(2 * channels, channels), nn.ReLU(),
                                 nn.Linear(channels, channels), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(2 * channels, channels), nn.ReLU(),
                                   nn.Linear(channels, channels), nn.ReLU())
        self.context_projection = conv(channels, channels)
        self.fusion = nn.Sequential(conv(3 * channels, channels), conv(channels, channels))
        self.residual_saliency = nn.Conv2d(channels, 1, 1)
        self.scale = channels ** -0.5

    def forward(self, features, saliency):
        saliency = resize(saliency, features.shape[-2:])
        regions = region_maps(saliency.sigmoid())
        used_regions = regions if self.use_uncertain_context else regions[:, :2]
        context = aggregate_context(features, used_regions)
        query = self.query(features)
        query_pixels = query.flatten(2).transpose(1, 2)
        scores = self.scale * (query_pixels @ self.key(context).transpose(1, 2))
        attended = scores.softmax(dim=-1) @ self.value(context)
        attended = self.context_projection(attended.transpose(1, 2).reshape_as(query))
        refined = self.fusion(torch.cat((features, attended), dim=1))
        return refined, saliency + self.residual_saliency(refined), regions[:, 2:3]


class UACANet(nn.Module):
    """Three encoder scales, initial PAA saliency and three UACA refinements."""

    def __init__(self, channels=16, backbone_width=16, use_uncertain_context=True):
        super().__init__()
        self.stem = conv(3, backbone_width, stride=2)
        widths = (backbone_width, 2 * backbone_width, 3 * backbone_width)
        self.backbone = nn.ModuleList([
            nn.Sequential(conv(inputs, outputs, stride=2), conv(outputs, outputs))
            for inputs, outputs in zip((backbone_width, *widths), widths)
        ])
        self.encoders = nn.ModuleList([PAAEncoder(width, channels) for width in widths])
        self.decoder = PAADecoder(channels)
        self.refinements = nn.ModuleList([UACA(channels, use_uncertain_context) for _ in widths])

    def forward(self, image):
        features = self.stem(image)
        pyramid = []
        for stage, encoder in zip(self.backbone, self.encoders):
            features = stage(features)
            pyramid.append(encoder(features))
        features, saliency = self.decoder(pyramid)
        predictions = [saliency]
        for skip, refine in zip(reversed(pyramid), self.refinements):
            merged = torch.cat((skip, resize(features, skip.shape[-2:])), dim=1)
            features, saliency, uncertainty = refine(merged, saliency)
            predictions.append(saliency)
        size = image.shape[-2:]
        return SegmentationOutput(tuple(resize(x, size) for x in predictions), resize(uncertainty, size))
