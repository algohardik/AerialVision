from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from inference_utils import load_image_file, make_overlay, predict_array


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sliding-window building segmentation on an image.")
    parser.add_argument("--image", type=Path, required=True, help="Input RGB image or GeoTIFF")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("predictions"))
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    image = load_image_file(args.image)
    model = tf.keras.models.load_model(args.model_path, compile=False)
    probabilities, mask = predict_array(model, image, args.patch_size, args.stride, args.threshold)
    overlay = make_overlay(image, mask)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    Image.fromarray(mask * 255).save(args.output_dir / f"{stem}_building_mask.png")
    Image.fromarray((probabilities * 255).astype(np.uint8)).save(args.output_dir / f"{stem}_probability.png")
    Image.fromarray(overlay).save(args.output_dir / f"{stem}_overlay.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image); axes[0].set_title("Input image")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Building mask")
    axes[2].imshow(overlay); axes[2].set_title("Overlay")
    for axis in axes: axis.axis("off")
    fig.tight_layout()
    fig.savefig(args.output_dir / f"{stem}_result.png", dpi=150)
    plt.close(fig)
    print(f"Saved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
