from __future__ import annotations

from pathlib import Path
import numpy as np
import rasterio
from PIL import Image
import tensorflow as tf

try:
    from .data_utils import patch_starts
except ImportError:  # Supports running src/predict.py as a script
    from data_utils import patch_starts


def load_image_file(file_path: str | Path) -> np.ndarray:
    path = Path(file_path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(path) as src:
            if src.count < 3:
                raise ValueError("The uploaded GeoTIFF must contain at least three bands.")
            arr = src.read([1, 2, 3]).transpose(1, 2, 0)
        return _to_uint8(arr)
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_image_bytes(data: bytes, filename: str) -> np.ndarray:
    suffix = Path(filename).suffix.lower()
    if suffix in {".tif", ".tiff"}:
        from rasterio.io import MemoryFile
        with MemoryFile(data) as memfile:
            with memfile.open() as src:
                if src.count < 3:
                    raise ValueError("The uploaded GeoTIFF must contain at least three bands.")
                arr = src.read([1, 2, 3]).transpose(1, 2, 0)
        return _to_uint8(arr)
    from io import BytesIO
    return np.asarray(Image.open(BytesIO(data)).convert("RGB"), dtype=np.uint8)


def _to_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array
    channels = []
    for index in range(array.shape[-1]):
        band = array[..., index].astype(np.float32)
        finite = band[np.isfinite(band)]
        if not finite.size:
            channels.append(np.zeros(band.shape, dtype=np.uint8))
            continue
        low, high = np.percentile(finite, [2, 98])
        if high <= low:
            high = low + 1
        channels.append((np.clip((band - low) / (high - low), 0, 1) * 255).astype(np.uint8))
    return np.stack(channels, axis=-1)


def predict_array(
    model: tf.keras.Model,
    image: np.ndarray,
    patch_size: int = 256,
    stride: int = 128,
    threshold: float = 0.5,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window inference with overlap averaging. Returns probability map and binary mask."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected an RGB image with shape (height, width, 3).")
    if stride > patch_size or stride < 1:
        raise ValueError("stride must be between 1 and patch_size.")

    height, width = image.shape[:2]
    ys = patch_starts(height, patch_size, stride)
    xs = patch_starts(width, patch_size, stride)
    prob_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    patches, positions = [], []

    def flush_batch():
        nonlocal patches, positions, prob_sum, weight_sum
        if not patches:
            return
        predicted = model.predict(np.stack(patches).astype(np.float32), verbose=0)[..., 0]
        for probability, (y, x, valid_h, valid_w) in zip(predicted, positions):
            prob_sum[y:y + valid_h, x:x + valid_w] += probability[:valid_h, :valid_w]
            weight_sum[y:y + valid_h, x:x + valid_w] += 1.0
        patches, positions = [], []

    for y in ys:
        for x in xs:
            tile = image[y:min(y + patch_size, height), x:min(x + patch_size, width)]
            valid_h, valid_w = tile.shape[:2]
            if valid_h != patch_size or valid_w != patch_size:
                padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                padded[:valid_h, :valid_w] = tile
                tile = padded
            patches.append(tile)
            positions.append((y, x, valid_h, valid_w))
            if len(patches) >= batch_size:
                flush_batch()
    flush_batch()

    probabilities = prob_sum / np.maximum(weight_sum, 1.0)
    mask = (probabilities >= threshold).astype(np.uint8)
    return probabilities, mask


def make_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image_float = image.astype(np.float32)
    overlay = image_float.copy()
    # Red highlight for building predictions.
    red = np.zeros_like(image_float)
    red[..., 0] = 255.0
    selected = mask.astype(bool)
    overlay[selected] = (1.0 - alpha) * image_float[selected] + alpha * red[selected]
    return np.clip(overlay, 0, 255).astype(np.uint8)
