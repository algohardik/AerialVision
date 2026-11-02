from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import tensorflow as tf

from input_pipeline import make_dataset, read_manifest
from losses import bce_dice_loss, dice_coefficient
from models import build_model
from paths import LOG_DIR, MODEL_DIR, PROCESSED_DIR, REPORT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AerialVision U-Net models with TensorFlow/Keras.")
    parser.add_argument("--manifest", type=Path, default=PROCESSED_DIR / "spacenet_paris" / "manifest.csv")
    parser.add_argument("--model", choices=["unet", "resnet50_unet"], default="unet")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--unfreeze-encoder", action="store_true", help="Fine-tune ResNet50 encoder; use a lower learning rate")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive.")

    manifest = args.manifest.resolve()
    dataset_root = manifest.parent
    frame = read_manifest(manifest)
    train_ds = make_dataset(frame, dataset_root, "train", args.patch_size, args.batch_size, True)
    val_ds = make_dataset(frame, dataset_root, "val", args.patch_size, args.batch_size, False)

    run_name = args.run_name or f"{args.model}"
    output_dir = MODEL_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = LOG_DIR / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    freeze_encoder = not args.unfreeze_encoder
    model = build_model(
        args.model,
        input_shape=(args.patch_size, args.patch_size, 3),
        weights=args.weights,
        freeze_encoder=freeze_encoder,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=bce_dice_loss,
        metrics=[
            dice_coefficient,
            tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name="iou"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    model.summary()

    best_path = output_dir / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_path), monitor="val_iou", mode="max", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_iou", mode="max", patience=8, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / "training_history.csv")),
        tf.keras.callbacks.TensorBoard(log_dir=str(log_dir)),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    final_path = output_dir / "final_model.keras"
    model.save(final_path)
    run_config = vars(args)
    run_config["manifest"] = str(manifest)
    run_config["freeze_encoder"] = freeze_encoder
    run_config["train_patches"] = int((frame["split"] == "train").sum())
    run_config["val_patches"] = int((frame["split"] == "val").sum())
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2, default=str)

    print(f"Best checkpoint: {best_path}")
    print(f"Final model: {final_path}")
    print(f"Training completed after {len(history.history.get('loss', []))} epochs.")


if __name__ == "__main__":
    main()
