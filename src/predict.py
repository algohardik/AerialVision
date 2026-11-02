from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

try:
    from .inference_utils import load_image_file, make_overlay, predict_array
    from .models import build_model
except ImportError:  # Supports: python src/predict.py
    from inference_utils import load_image_file, make_overlay, predict_array
    from models import build_model


def infer_model_type(model_path: Path, requested_type: str) -> str:
    """Infer the architecture from the checkpoint path unless explicitly set."""
    if requested_type != "auto":
        return requested_type

    normalized = str(model_path).lower().replace("-", "_")
    if "resnet50_unet" in normalized:
        if "finetuned" in normalized or "fine_tuned" in normalized:
            return "resnet50_unet_finetuned"
        return "resnet50_unet"
    return "unet"


def load_checkpoint(model_path: Path, model_type: str, patch_size: int) -> tf.keras.Model:
    """Load U-Net normally and rebuild ResNet50 U-Net before restoring weights.

    Keras 3 may reject deserializing the Lambda preprocessing layer contained in
    the saved ResNet50 .keras model. Rebuilding the known architecture and
    restoring model.weights.h5 avoids Lambda deserialization.
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    if model_type == "unet":
        return tf.keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False,
        )

    fine_tuned = model_type == "resnet50_unet_finetuned"
    model = build_model(
        model_name="resnet50_unet",
        input_shape=(patch_size, patch_size, 3),
        weights="none",
        freeze_encoder=not fine_tuned,
    )

    with zipfile.ZipFile(model_path, "r") as archive:
        weight_files = [
            name for name in archive.namelist()
            if name.endswith("model.weights.h5")
        ]
        if not weight_files:
            raise FileNotFoundError(
                f"No model.weights.h5 file was found inside checkpoint: {model_path}"
            )

        # Keep the temporary HDF5 file alive until Keras has loaded its weights.
        with tempfile.TemporaryDirectory(prefix="aerialvision_weights_") as temp_dir:
            weights_path = Path(temp_dir) / "checkpoint.weights.h5"
            with archive.open(weight_files[0], "r") as source:
                with weights_path.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
            model.load_weights(weights_path)

    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sliding-window building-footprint segmentation on an image."
    )
    parser.add_argument("--image", type=Path, required=True,
                        help="Input RGB image or 3-band GeoTIFF")
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Path to a trained .keras checkpoint")
    parser.add_argument(
        "--model-type",
        choices=("auto", "unet", "resnet50_unet", "resnet50_unet_finetuned"),
        default="auto",
        help="Architecture; auto-detects from the checkpoint path (default: auto)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("predictions"))
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Probability threshold (default: 0.60 for fine-tuned ResNet50 U-Net, otherwise 0.50)",
    )
    args = parser.parse_args()

    model_path = args.model_path.expanduser().resolve()
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if args.patch_size < 1:
        parser.error("--patch-size must be positive")
    if not 1 <= args.stride <= args.patch_size:
        parser.error("--stride must be between 1 and --patch-size")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    model_type = infer_model_type(model_path, args.model_type)
    threshold = args.threshold
    if threshold is None:
        threshold = 0.60 if model_type == "resnet50_unet_finetuned" else 0.50
    if not 0.0 <= threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    print(f"Loading {model_type} checkpoint: {model_path}")
    model = load_checkpoint(model_path, model_type, args.patch_size)
    print(f"Using probability threshold: {threshold:.2f}")

    image = load_image_file(image_path)
    probabilities, mask = predict_array(
        model,
        image,
        patch_size=args.patch_size,
        stride=args.stride,
        threshold=threshold,
        batch_size=args.batch_size,
    )
    overlay = make_overlay(image, mask)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    Image.fromarray((mask * 255).astype(np.uint8)).save(
        args.output_dir / f"{stem}_building_mask.png"
    )
    Image.fromarray((probabilities * 255).astype(np.uint8)).save(
        args.output_dir / f"{stem}_probability.png"
    )
    Image.fromarray(np.asarray(overlay, dtype=np.uint8)).save(
        args.output_dir / f"{stem}_overlay.png"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Input image")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Building mask")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(args.output_dir / f"{stem}_result.png", dpi=150)
    plt.close(fig)

    print(f"Saved predictions to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
