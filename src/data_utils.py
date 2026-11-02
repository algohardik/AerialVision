from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom


def extract_image_id(path: Path) -> str | None:
    match = re.search(r"_img(\d+)", path.stem, flags=re.IGNORECASE)
    return match.group(1) if match else None


def discover_spacenet_pairs(raw_root: Path) -> list[tuple[Path, Path]]:
    """Find matching RGB-PanSharpen TIFF and building GeoJSON files recursively."""
    image_files = list(raw_root.rglob("RGB-PanSharpen/*.tif"))
    if not image_files:
        # Also support an extracted folder where TIFFs are in a differently named RGB dir.
        image_files = [
            p for p in raw_root.rglob("*.tif")
            if "RGB-PanSharpen" in {part for part in p.parts}
        ]

    label_files = list(raw_root.rglob("geojson/buildings/*.geojson"))
    if not label_files:
        label_files = list(raw_root.rglob("buildings/*.geojson"))

    labels_by_id: dict[str, Path] = {}
    for label in label_files:
        image_id = extract_image_id(label)
        if image_id is not None:
            labels_by_id[image_id] = label

    pairs: list[tuple[Path, Path]] = []
    for image in image_files:
        image_id = extract_image_id(image)
        if image_id is not None and image_id in labels_by_id:
            pairs.append((image, labels_by_id[image_id]))

    # Deterministic order and no duplicate image IDs.
    unique: dict[str, tuple[Path, Path]] = {}
    for image, label in sorted(pairs, key=lambda item: str(item[0])):
        unique[extract_image_id(image) or str(image)] = (image, label)
    return list(unique.values())


def _geojson_source_crs(document: dict) -> str:
    """SpaceNet GeoJSON is normally CRS84/EPSG:4326; honor a declared CRS if present."""
    crs_obj = document.get("crs")
    if isinstance(crs_obj, dict):
        properties = crs_obj.get("properties", {})
        name = properties.get("name") or properties.get("href")
        if name:
            name = str(name)
            if "EPSG" in name.upper():
                code = re.search(r"(\d{4,6})", name)
                if code:
                    return f"EPSG:{code.group(1)}"
            if "CRS84" in name.upper() or "4326" in name:
                return "EPSG:4326"
    return "EPSG:4326"


def read_rgb_geotiff(path: Path) -> tuple[np.ndarray, object]:
    """Read first three bands and return uint8 RGB plus rasterio metadata."""
    with rasterio.open(path) as src:
        if src.count < 3:
            raise ValueError(f"Expected at least 3 raster bands in {path}, found {src.count}")
        array = src.read([1, 2, 3]).transpose(1, 2, 0)
        profile = {
            "transform": src.transform,
            "crs": src.crs,
            "height": src.height,
            "width": src.width,
            "profile": src.profile.copy(),
        }

    if array.dtype == np.uint8:
        rgb = array
    else:
        # Robust percentile stretch for non-8-bit input. Percentiles are per band.
        channels = []
        for channel in range(array.shape[-1]):
            band = array[..., channel].astype(np.float32)
            finite = band[np.isfinite(band)]
            if finite.size == 0:
                channels.append(np.zeros_like(band, dtype=np.uint8))
                continue
            low, high = np.percentile(finite, [2, 98])
            if high <= low:
                high = low + 1.0
            scaled = np.clip((band - low) / (high - low), 0, 1) * 255.0
            channels.append(scaled.astype(np.uint8))
        rgb = np.stack(channels, axis=-1)
    return rgb, profile


def rasterize_buildings(label_path: Path, shape: tuple[int, int], transform, target_crs) -> np.ndarray:
    """Rasterize GeoJSON building polygons into a 0/1 mask aligned to the image."""
    with label_path.open("r", encoding="utf-8") as file:
        document = json.load(file)

    source_crs = _geojson_source_crs(document)
    geometries = []
    for feature in document.get("features", []):
        geometry = feature.get("geometry")
        if not geometry:
            continue
        if target_crs is None:
            raise ValueError("Image GeoTIFF has no CRS; cannot align GeoJSON building polygons.")
        try:
            projected = transform_geom(source_crs, target_crs, geometry, precision=-1)
            geometries.append((projected, 1))
        except Exception as exc:
            raise ValueError(f"Could not transform geometry in {label_path}: {exc}") from exc

    if not geometries:
        return np.zeros(shape, dtype=np.uint8)

    mask = rasterize(
        geometries,
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )
    return mask


def patch_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def create_patches(image: np.ndarray, mask: np.ndarray, patch_size: int, stride: int) -> Iterable[tuple[np.ndarray, np.ndarray, int, int]]:
    height, width = mask.shape
    for y in patch_starts(height, patch_size, stride):
        for x in patch_starts(width, patch_size, stride):
            image_patch = image[y:y + patch_size, x:x + patch_size]
            mask_patch = mask[y:y + patch_size, x:x + patch_size]
            if image_patch.shape[0] != patch_size or image_patch.shape[1] != patch_size:
                padded_image = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                padded_mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
                padded_image[:image_patch.shape[0], :image_patch.shape[1]] = image_patch
                padded_mask[:mask_patch.shape[0], :mask_patch.shape[1]] = mask_patch
                image_patch, mask_patch = padded_image, padded_mask
            yield image_patch, mask_patch, y, x
