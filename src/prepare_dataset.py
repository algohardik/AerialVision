from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from data_utils import create_patches, discover_spacenet_pairs, rasterize_buildings, read_rgb_geotiff
from paths import PROCESSED_DIR, RAW_DIR


def split_scenes(pairs, seed: int, train_ratio: float, val_ratio: float):
    items = list(pairs)
    random.Random(seed).shuffle(items)
    n = len(items)
    if n < 3:
        raise ValueError(f"At least 3 matched image/label scenes are needed; found {n}.")
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_train + n_val >= n:
        n_train, n_val = n - 2, 1
    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SpaceNet2 GeoTIFF + GeoJSON to paired segmentation patches.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="Directory containing the extracted SpaceNet archive")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR / "spacenet_paris")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256, help="Use 128 for 50%% overlap; 256 avoids duplicate patches")
    parser.add_argument("--max-scenes", type=int, default=0, help="Limit source image scenes for a quick pilot; 0 uses all matched scenes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    if args.patch_size < 64 or args.stride < 1 or args.stride > args.patch_size:
        raise ValueError("Use patch-size >= 64 and 1 <= stride <= patch-size.")
    if not (0 < args.train_ratio < 1 and 0 < args.val_ratio < 1 and args.train_ratio + args.val_ratio < 1):
        raise ValueError("train-ratio and val-ratio must be positive and sum to less than 1.")

    pairs = discover_spacenet_pairs(args.raw_dir)
    if not pairs:
        raise SystemExit(
            f"No image/GeoJSON pairs found under {args.raw_dir}. Expected RGB-PanSharpen/*.tif and "
            "geojson/buildings/*.geojson from the extracted SpaceNet2 archive. See README.md."
        )

    if args.max_scenes > 0:
        pairs = pairs[:args.max_scenes]
    print(f"Found {len(pairs)} matched source scenes.")
    splits = split_scenes(pairs, args.seed, args.train_ratio, args.val_ratio)

    for split in splits:
        (args.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_dir / "masks" / split).mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / "manifest.csv"
    manifest_rows = []
    counts = {name: 0 for name in splits}

    for split_name, split_pairs in splits.items():
        for image_path, label_path in tqdm(split_pairs, desc=f"Preparing {split_name}"):
            scene_id = image_path.stem
            try:
                rgb, metadata = read_rgb_geotiff(image_path)
                mask = rasterize_buildings(
                    label_path,
                    shape=rgb.shape[:2],
                    transform=metadata["transform"],
                    target_crs=metadata["crs"],
                )
            except Exception as exc:
                print(f"\nSkipping {image_path.name}: {exc}", file=sys.stderr)
                continue

            if rgb.shape[:2] != mask.shape:
                print(f"Skipping {image_path.name}: image/mask dimensions differ.", file=sys.stderr)
                continue

            for patch_number, (image_patch, mask_patch, y, x) in enumerate(
                create_patches(rgb, mask, args.patch_size, args.stride)
            ):
                patch_name = f"{scene_id}_y{y}_x{x}_p{patch_number:02d}.png"
                image_out = args.output_dir / "images" / split_name / patch_name
                mask_out = args.output_dir / "masks" / split_name / patch_name
                Image.fromarray(image_patch, mode="RGB").save(image_out, optimize=True)
                Image.fromarray((mask_patch * 255).astype(np.uint8), mode="L").save(mask_out, optimize=True)
                manifest_rows.append({
                    "image": image_out.relative_to(args.output_dir).as_posix(),
                    "mask": mask_out.relative_to(args.output_dir).as_posix(),
                    "split": split_name,
                    "scene_id": scene_id,
                    "foreground_pixels": int(mask_patch.sum()),
                    "patch_y": y,
                    "patch_x": x,
                })
                counts[split_name] += 1

    if not manifest_rows:
        raise SystemExit("No patches were created. Check the GeoJSON CRS and source archive structure.")

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\nPreprocessing complete.")
    print(f"Manifest: {manifest_path}")
    print(f"Patch counts: {counts}")
    print("Source-scene split was performed before patch extraction to reduce spatial leakage between splits.")


if __name__ == "__main__":
    main()
