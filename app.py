"""Offline Gradio demo for locally trained UACANet checkpoints."""

import argparse
import os
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import cv2
import gradio as gr
import numpy as np
import torch

from src.inference import color_heatmap, load_model_from_checkpoint, run_inference_on_image
from src.models import UACANet

PRIMARY = Path("checkpoints/kvasir/uacanet/best_model.pt")
SMOKE = Path("checkpoints/smoke/uacanet/best_model.pt")


def export_outputs(result):
    images = {
        "segmentation.png": result["binary_mask"] * 255,
        "probability_16bit.png": np.rint(result["mean_probability"] * 65535).astype(np.uint16),
        "uaca_uncertainty_16bit.png": np.rint(result["uncertainty_map"] * 65535).astype(np.uint16),
        "uncertainty_heatmap.png": cv2.cvtColor(color_heatmap(2 * result["uncertainty_map"]), cv2.COLOR_RGB2BGR),
        "uncertainty_overlay.png": cv2.cvtColor(result["uncertainty_overlay"], cv2.COLOR_RGB2BGR),
    }
    output = Path(tempfile.mkdtemp(prefix="p32-demo-")) / "p32_prediction.zip"
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name, image in images.items():
            okay, encoded = cv2.imencode(".png", image)
            if not okay:
                raise OSError(f"Unable to encode {name}")
            archive.writestr(name, encoded.tobytes())
        archive.writestr("README.txt", "Divide 16-bit map pixels by 65535. UACA ranges from 0 to 0.5.\n")
    return str(output)


def predict_gradio(input_image, checkpoint_choice):
    if input_image is None:
        raise gr.Error("Upload an RGB image to run segmentation.")
    if not checkpoint_choice:
        raise gr.Error("Train a UACANet checkpoint using bash run_all.sh.")
    model, checkpoint = load_model_from_checkpoint(checkpoint_choice)
    if not isinstance(model, UACANet):
        raise gr.Error("Select a trained UACANet checkpoint.")
    config = checkpoint["config"]
    torch.set_num_threads(config["project"].get("threads", 4))
    threshold = checkpoint.get("threshold", config["inference"].get("evaluation_threshold", 0.5))
    result = run_inference_on_image(model, input_image, tuple(config["data"]["image_size"]), threshold)
    uncertainty = result["uncertainty_map"]
    return (
        result["binary_mask"] * 255,
        np.rint(result["mean_probability"] * 255).astype(np.uint8),
        color_heatmap(2 * uncertainty),
        result["uncertainty_overlay"],
        f"Validation threshold {threshold:.2f} · Predicted area {result['binary_mask'].mean():.1%} · "
        f"Mean UACA uncertainty {uncertainty.mean():.3f}",
        export_outputs(result),
    )


def create_demo(checkpoint=None):
    candidates = [(label, str(path)) for label, path in (
        ("Kvasir-SEG UACANet", PRIMARY), ("Synthetic smoke UACANet", SMOKE)
    ) if path.is_file()]
    if checkpoint and str(checkpoint) not in {value for _, value in candidates}:
        candidates.insert(0, ("Local UACANet checkpoint", str(checkpoint)))
    selection = str(checkpoint) if checkpoint else (candidates[0][1] if candidates else None)

    with gr.Blocks(title="P32 Polyp Segmentation") as demo:
        gr.Markdown("# Polyp segmentation\nCourse research demo. Use a local image and a trained checkpoint.")
        with gr.Row():
            source = gr.Image(type="numpy", image_mode="RGB", sources=["upload"], label="Colonoscopy image")
            with gr.Column():
                model = gr.Dropdown(choices=candidates, value=selection, label="Checkpoint")
                run = gr.Button("Segment image", variant="primary")
                stats = gr.Markdown()
                download = gr.File(label="Download outputs")
        with gr.Row():
            mask = gr.Image(label="Segmentation", interactive=False)
            probability = gr.Image(label="Polyp probability", interactive=False)
        with gr.Row():
            uncertainty = gr.Image(label="UACA uncertainty (0 to 0.5)", interactive=False)
            overlay = gr.Image(label="Uncertainty overlay", interactive=False)
        run.click(predict_gradio, [source, model], [mask, probability, uncertainty, overlay, stats, download])
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    create_demo(args.checkpoint).launch(server_name="127.0.0.1", server_port=args.port, share=False)
