from __future__ import annotations

import argparse
import json
import zipfile
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from input_pipeline import read_manifest
from models import build_model
from paths import MODEL_DIR, PROCESSED_DIR, REPORT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved model on the held-out test split.")
    parser.add_argument("--manifest", type=Path, default=PROCESSED_DIR / "spacenet_paris" / "manifest.csv")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR / "evaluation")
    parser.add_argument("--visualize", type=int, default=12)
    args = parser.parse_args()

    frame = read_manifest(args.manifest.resolve())
    test_rows = frame[frame["split"] == "test"].reset_index(drop=True)
    if test_rows.empty:
        raise SystemExit("No test rows found. Re-run preprocessing with at least 3 source scenes.")
    dataset_root = args.manifest.resolve().parent
    if "resnet50_unet" in str(args.model_path):
        print("Rebuilding ResNet50 U-Net architecture...")

        model = build_model(
            model_name="resnet50_unet",
            input_shape=(args.patch_size, args.patch_size, 3),
            weights="none",
            freeze_encoder=True,
        )

        print("Extracting trained weights from Keras archive...")

        extracted_weights = Path("/content/resnet50_unet.weights.h5")

        with zipfile.ZipFile(args.model_path, "r") as archive:
            weight_files = [
                name for name in archive.namelist()
                if name.endswith("model.weights.h5")
            ]

            if not weight_files:
                raise FileNotFoundError(
                    "Could not find model.weights.h5 inside checkpoint."
                )

            with archive.open(weight_files[0]) as source:
                with extracted_weights.open("wb") as destination:
                    shutil.copyfileobj(source, destination)

        model.load_weights(extracted_weights)
        print("ResNet50 U-Net trained weights loaded!")

    else:
        model = tf.keras.models.load_model(
            args.model_path,
            compile=False,
            safe_mode=False
        )

    tp = fp = fn = tn = 0
    per_image_dice = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    selected = set(rng.choice(len(test_rows), size=min(args.visualize, len(test_rows)), replace=False).tolist())

    for start in range(0, len(test_rows), args.batch_size):
        batch_rows = test_rows.iloc[start:start + args.batch_size]
        images, masks = [], []
        for row in batch_rows.itertuples(index=False):
            image = np.asarray(Image.open(dataset_root / row.image).convert("RGB"), dtype=np.float32)
            mask = np.asarray(Image.open(dataset_root / row.mask).convert("L"), dtype=np.uint8) > 127
            images.append(image)
            masks.append(mask)
        x = np.stack(images)
        y = np.stack(masks)
        probabilities = model.predict(x, verbose=0)[..., 0]
        predictions = probabilities >= args.threshold
        tp += int(np.logical_and(predictions, y).sum())
        fp += int(np.logical_and(predictions, np.logical_not(y)).sum())
        fn += int(np.logical_and(np.logical_not(predictions), y).sum())
        tn += int(np.logical_and(np.logical_not(predictions), np.logical_not(y)).sum())

        for local_index in range(len(batch_rows)):
            global_index = start + local_index
            intersection = np.logical_and(predictions[local_index], y[local_index]).sum()
            denominator = predictions[local_index].sum() + y[local_index].sum()
            per_image_dice.append(float((2 * intersection + 1e-7) / (denominator + 1e-7)))
            if global_index in selected:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(x[local_index].astype(np.uint8)); axes[0].set_title("Image")
                axes[1].imshow(y[local_index], cmap="gray"); axes[1].set_title("Ground truth")
                axes[2].imshow(predictions[local_index], cmap="gray"); axes[2].set_title("Prediction")
                for axis in axes: axis.axis("off")
                fig.tight_layout()
                fig.savefig(args.output_dir / f"test_example_{global_index}.png", dpi=140)
                plt.close(fig)

    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    metrics = {
        "test_patches": int(len(test_rows)),
        "threshold": args.threshold,
        "pixel_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "dice": dice,
        "mean_per_patch_dice": float(np.mean(per_image_dice)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "note": "Pixel metrics are aggregated over all test pixels; mean_per_patch_dice gives each patch equal weight.",
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Visual examples saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
