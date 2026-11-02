
import argparse
import json
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from models import build_model


def load_resnet_checkpoint(model_path):
    """Rebuild ResNet50 U-Net and load weights from a Keras archive."""

    model = build_model(
        "resnet50_unet",
        input_shape=(256, 256, 3),
        weights="none",
        freeze_encoder=False
    )

    with zipfile.ZipFile(model_path, "r") as archive:
        with tempfile.NamedTemporaryFile(
            suffix=".weights.h5", delete=False
        ) as temp:
            temp_path = temp.name
            with archive.open("model.weights.h5") as source:
                temp.write(source.read())

    model.load_weights(temp_path)
    Path(temp_path).unlink(missing_ok=True)

    return model


def load_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((256, 256))
    return np.asarray(image, dtype=np.float32)


def load_mask(path):
    mask = Image.open(path).convert("L")
    mask = mask.resize((256, 256), resample=Image.Resampling.NEAREST)
    return (np.asarray(mask, dtype=np.float32) > 127).astype(np.uint8)


def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()

    return intersection / union if union else 1.0


def main():
    parser = argparse.ArgumentParser(
        description="Choose a probability threshold using validation data."
    )

    parser.add_argument("--manifest", default="data/processed/spacenet_paris/manifest.csv")
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default="results/best_threshold.json")

    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)

    print("Manifest columns:", manifest.columns.tolist())

    # Use validation split only.
    val_df = manifest[
        manifest["split"].astype(str).str.lower().isin(["val", "valid", "validation"])
    ].copy()

    if val_df.empty:
        raise ValueError("No validation rows found in manifest.")

    # Identify image and mask columns.
    image_col = next(
        (c for c in ["image_path", "image", "image_file"] if c in val_df.columns),
        None
    )

    mask_col = next(
        (c for c in ["mask_path", "mask", "mask_file"] if c in val_df.columns),
        None
    )

    if image_col is None or mask_col is None:
        raise ValueError(
            "Could not identify image/mask columns. "
            "Manifest columns are: " + str(val_df.columns.tolist())
        )

    print("Validation samples:", len(val_df))
    print("Loading fine-tuned ResNet50 U-Net...")

    model = load_resnet_checkpoint(args.model_path)

    images = []
    masks = []

    for _, row in val_df.iterrows():
        image_path = Path(row[image_col])
        mask_path = Path(row[mask_col])

        if not image_path.is_absolute():
            image_path = Path(args.manifest).parent / image_path

        if not mask_path.is_absolute():
            mask_path = Path(args.manifest).parent / mask_path

        images.append(load_image(image_path))
        masks.append(load_mask(mask_path))

    X = np.asarray(images, dtype=np.float32)
    Y = np.asarray(masks, dtype=np.uint8)

    print("Validation image shape:", X.shape)
    print("Generating predictions...")

    probabilities = model.predict(
        X,
        batch_size=args.batch_size,
        verbose=1
    )

    probabilities = np.squeeze(probabilities, axis=-1)

    results = []

    for threshold in args.thresholds:
        predictions = probabilities >= threshold

        iou = calculate_iou(Y.astype(bool), predictions)

        results.append({
            "threshold": float(threshold),
            "validation_iou": float(iou)
        })

        print(
            f"Threshold: {threshold:.2f} | "
            f"Validation IoU: {iou:.5f}"
        )

    best = max(results, key=lambda x: x["validation_iou"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "best_threshold": best["threshold"],
        "best_validation_iou": best["validation_iou"],
        "results": results
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    print("\nBest threshold:", best["threshold"])
    print("Best validation IoU:", best["validation_iou"])
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
