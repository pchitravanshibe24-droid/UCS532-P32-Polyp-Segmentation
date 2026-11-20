"""Model architectures and factories."""

from src.models.uacanet import SegmentationOutput, UACANet
from src.models.unet import DoubleConv, UNet, build_model as build_unet


def build_model(config):
    model = config.get("model", {})
    architecture = model.get("architecture", "unet")
    if architecture == "unet":
        return build_unet(config)
    if architecture == "uacanet":
        return UACANet(
            channels=model.get("channels", 16),
            backbone_width=model.get("backbone_width", 16),
            use_uncertain_context=model.get("use_uncertain_context", True),
        )
    raise ValueError(f"Unsupported architecture: {architecture}")


def final_logits(output):
    return output.logits[-1] if isinstance(output, SegmentationOutput) else output

__all__ = ["UNet", "DoubleConv", "build_model"]
