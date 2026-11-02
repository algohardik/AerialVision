"""Create a tiny synthetic GeoTIFF + GeoJSON dataset to smoke-test preprocessing.

This is NOT real satellite imagery and must NOT be used for project performance claims.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from paths import RAW_DIR


def pixel_to_world(transform, col, row):
    x, y = transform * (col, row)
    return [float(x), float(y)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=RAW_DIR / "synthetic_demo")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.scenes < 3:
        raise ValueError("Use at least 3 scenes so train/validation/test splits are possible.")

    image_dir = args.output_dir / "SN2_buildings_train_AOI_3_Paris" / "AOI_3_Paris_Train" / "RGB-PanSharpen"
    label_dir = args.output_dir / "SN2_buildings_train_AOI_3_Paris" / "AOI_3_Paris_Train" / "geojson" / "buildings"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    height = width = 256
    transform = from_origin(-1.0, 1.0, 0.0001, 0.0001)

    for scene in range(args.scenes):
        # Simulated roads/background and bright building roofs.
        image = rng.normal(90, 18, size=(height, width, 3)).clip(0, 255).astype(np.uint8)
        image[::48, :, :] = [65, 65, 65]
        image[:, ::48, :] = [65, 65, 65]
        features = []
        building_count = int(rng.integers(8, 20))
        for _ in range(building_count):
            x1 = int(rng.integers(8, width - 35))
            y1 = int(rng.integers(8, height - 35))
            bw = int(rng.integers(8, 28))
            bh = int(rng.integers(8, 28))
            x2, y2 = min(width - 2, x1 + bw), min(height - 2, y1 + bh)
            image[y1:y2, x1:x2] = [190, 180, 160]
            ring = [
                pixel_to_world(transform, x1, y1),
                pixel_to_world(transform, x2, y1),
                pixel_to_world(transform, x2, y2),
                pixel_to_world(transform, x1, y2),
                pixel_to_world(transform, x1, y1),
            ]
            features.append({
                "type": "Feature",
                "properties": {"building_id": len(features) + 1},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            })

        stem = f"AOI_3_Paris_img{scene:05d}"
        image_path = image_dir / f"RGB-PanSharpen_{stem}.tif"
        label_path = label_dir / f"buildings_{stem}.geojson"
        with rasterio.open(
            image_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(image.transpose(2, 0, 1))
        geojson = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": features,
        }
        label_path.write_text(json.dumps(geojson), encoding="utf-8")

    print(f"Created {args.scenes} synthetic image/label pairs under {args.output_dir}")
    print("This is test data only, not real satellite imagery.")


if __name__ == "__main__":
    main()
