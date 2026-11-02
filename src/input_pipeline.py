from __future__ import annotations

from pathlib import Path
import pandas as pd
import tensorflow as tf


def read_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run src/prepare_dataset.py first.")
    frame = pd.read_csv(manifest_path)
    required = {"image", "mask", "split", "scene_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    return frame


def _decode_pair(image_path, mask_path, patch_size: int):
    image_bytes = tf.io.read_file(image_path)
    mask_bytes = tf.io.read_file(mask_path)
    image = tf.io.decode_png(image_bytes, channels=3)
    mask = tf.io.decode_png(mask_bytes, channels=1)
    image = tf.cast(image, tf.float32)  # Keep 0..255; models handle their own preprocessing.
    mask = tf.cast(mask > 127, tf.float32)
    image = tf.image.resize(image, [patch_size, patch_size], method="bilinear")
    mask = tf.image.resize(mask, [patch_size, patch_size], method="nearest")
    return image, mask


def _augment(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    mask = tf.image.rot90(mask, k)
    image = tf.image.random_brightness(image, max_delta=12.0)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, mask


def make_dataset(frame: pd.DataFrame, dataset_root: Path, split: str, patch_size: int, batch_size: int, training: bool):
    rows = frame[frame["split"] == split]
    if rows.empty:
        raise ValueError(f"No samples found for split '{split}' in manifest.")
    image_paths = [str(dataset_root / path) for path in rows["image"].tolist()]
    mask_paths = [str(dataset_root / path) for path in rows["mask"].tolist()]

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    if training:
        dataset = dataset.shuffle(min(len(image_paths), 4096), reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda image_path, mask_path: _decode_pair(image_path, mask_path, patch_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if training:
        dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
