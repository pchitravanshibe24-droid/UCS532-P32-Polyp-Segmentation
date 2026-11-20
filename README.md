# P32 · Polyp Segmentation with Explicit Pixel-Level Uncertainty

CS-UCS532 Open Challenge. Given one RGB colonoscopy frame, segment the visible polyp and show where the previous saliency estimate supplied ambiguous context to the last refinement stage.

## Reproduce

Python 3.11 and Bash are required. Git Bash works on Windows. Install the pinned environment once. To match the recorded CPU build, install PyTorch from its CPU wheel index first.

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
bash run_all.sh
```

The last command verifies and extracts the official dataset archive, creates the split, trains all three networks, freezes each validation-selected checkpoint and threshold, evaluates the held-out test set, exports diagnostic figures, and runs the corruption study. The main table appears in `outputs/kvasir/tables/main_table.csv`. It is generated from scratch. Downloads and run outputs stay local. For an offline benchmark rerun, place the official archive at `data/kvasir-seg.zip` before starting.

A fresh local clone reproduced the recorded tables and training histories byte for byte. An independent Windows run completed the same command and preserved the model ranking. It selected different validation thresholds and produced different numerical scores. The [hosted run](https://github.com/pchitravanshibe24-droid/UCS532-P32-Polyp-Segmentation/actions/runs/44798481575), [hosted table](results/tables/hosted_main_table.csv), and [metric deltas](results/logs/hosted-comparison.log) record the cross-environment comparison.

```bash
bash run_all.sh --smoke
python -m pytest -q
python app.py
```

The smoke run generates 20 synthetic pairs and exercises train–reload–evaluate on CPU. Its results are pipeline checks. The demo binds to localhost and uses the trained Kvasir UACANet by default when available. After initial setup it runs without an internet connection. To use the weight file from the extracted course ZIP, pass `--checkpoint results/checkpoints/uacanet/best_model.pt` or the full extracted path. Upload a local RGB image to display and download segmentation, polyp probabilities, UACA ambiguity, and an overlay.

## Dataset and protocol

- [Kvasir-SEG dataset](https://datasets.simula.no/kvasir-seg/) and [official archive](https://datasets.simula.no/downloads/kvasir-seg.zip). All 1,000 valid matched images and masks were used. The split is 700 train, 150 validation and 150 test. Split and training seed 42 is printed in logs. The archive checksum and split fingerprints are in the evidence folder.
- Images and masks are paired by case-insensitive filename stem. Unmatched, corrupt or size-mismatched input fails the run. Identical image and mask pairs are deduplicated before splitting. Conflicting labels on identical image files fail the run. The split is image-level, with no shared exact image hashes. Patient groups and near-duplicate exclusion are unverified.
- Network input is 96 × 96 with RGB scaled to [0, 1]. Masks use nearest-neighbor resizing and are binarized. Geometric augmentation is paired and appearance augmentation affects only the training RGB image.
- All variants train for up to 12 epochs from random initialization with AdamW, initial learning rate 0.001, batch size 12 and the same split, seed, augmentation and BCE + soft IoU per head. Compact UACANet supervises four heads, U-Net supervises one. Early stopping is disabled for the recorded run.
- Select the epoch and threshold by validation macro Dice on the fixed grid 0.30, 0.40, 0.50, 0.60, 0.70. Threshold ties prefer closest to 0.50 and then lower threshold. Epoch ties keep the first. Test masks use native image dimensions, with probabilities bilinearly resized before thresholding. The final metrics are per-image macro means and population standard deviations. MAE is the pixel mean of the absolute difference between probability and the binary label. F1 equals Dice for these binary masks.
- The CPU reference is Windows, AMD Ryzen 5 5500U, four PyTorch threads, Python 3.11.16 and PyTorch 2.14.1+cpu. The log records measured runtimes. `bash run_all.sh --device cuda` also accepts a CUDA build. CUDA results can differ and were unverified on this host.

## Models and results

- **Compact UACANet** implements parallel axial encoder/decoder branches, a coarse saliency head, and three UACA refinements. It follows the paper’s foreground, background and uncertainty weighting and query/key/value aggregation. The real-data run uses a small convolutional backbone, GroupNorm, smaller inputs, no pretraining, and an unweighted spatial soft-IoU loss.
- **U-Net** is the implementation already in this repository and serves as the deterministic baseline.
- **Two-region ablation** retrains the same compact UACANet after removing the uncertain context token from attention. The remaining foreground/background weights and parameter shapes are retained. Its treatment effect is conditional on this split and seed.
- **Extension** checks a test slice based on the median training-mask area. The robustness study pairs each clean prediction with fixed moderate blur and lower brightness for the same test image. Uncertainty association is measured descriptively using quantile bins and error AUROC. UACA ambiguity is based on previous saliency near 0.5. Predictive entropy uses final segmentation probabilities. MC Dropout remains available only as a separate option for compatible U-Net checkpoints.

<!-- RESULTS_TABLE -->

| Model | Dice | IoU | Precision | Recall | F1 | MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Compact UACANet | 0.4977 | 0.3784 | 0.5573 | 0.5983 | 0.4977 | 0.1564 |
| U-Net | 0.6032 | 0.4879 | 0.6852 | 0.6455 | 0.6032 | 0.1278 |
| UACANet without uncertain context | 0.4912 | 0.3675 | 0.5164 | 0.6335 | 0.4912 | 0.1579 |

Recorded data preparation and training took 561.61 s. Evaluation and stress analysis took 115.96 s.

## Evidence and credits

- See [results](results/) for per-image tables, training history, split metadata, run logs and derived figures. The local `outputs/kvasir/*/figures/test_panels` folders contain the full image-by-image displays. Run the reproduction command to regenerate local checkpoints. The course ZIP includes the three benchmark checkpoints for offline use.
- Taehun Kim, Hyemin Lee and Daijin Kim, [UACANet, ACM MM 2021](https://doi.org/10.1145/3474085.3475375). The architecture and implementation details were adapted from the [authors’ MIT-licensed code](https://github.com/plemeri/UACANet). Its [license notice](LICENSES/UACANet-MIT.txt) is included. No upstream checkpoint was used.
- Olaf Ronneberger, Philipp Fischer and Thomas Brox, [U-Net, MICCAI 2015](https://arxiv.org/abs/1505.04597). The baseline reuses this project’s existing U-Net code. Debesh Jha et al., [Kvasir-SEG, MMM 2020](https://doi.org/10.1007/978-3-030-37734-2_37), provided the dataset.
- Yarin Gal and Zoubin Ghahramani, [MC Dropout, ICML 2016](https://proceedings.mlr.press/v48/gal16.html), informed the legacy optional comparator.

## AI declaration

OpenAI Codex was used to audit and modify the code, adapt the architecture, add tests and analysis, run and verify experiments, and draft the report and slides from recorded outputs. The team must review the work, confirm individual contributions, and sign its declarations before submission.
