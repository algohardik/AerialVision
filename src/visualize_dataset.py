from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from paths import PROCESSED_DIR


def main():
    parser = argparse.ArgumentParser(description="Visualize random image/mask pairs from the processed dataset.")
    parser.add_argument("--manifest", type=Path, default=PROCESSED_DIR / "spacenet_paris" / "manifest.csv")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("reports/dataset_samples.png"))
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    rows = frame[frame.split == args.split]
    if rows.empty:
        raise SystemExit(f"No examples in split {args.split}.")
    sample = rows.sample(min(args.count, len(rows)), random_state=42)
    root = args.manifest.parent

    fig, axes = plt.subplots(len(sample), 2, figsize=(9, 4 * len(sample)), squeeze=False)
    for row_index, row in enumerate(sample.itertuples(index=False)):
        image = np.asarray(Image.open(root / row.image).convert("RGB"))
        mask = np.asarray(Image.open(root / row.mask).convert("L"))
        axes[row_index, 0].imshow(image); axes[row_index, 0].set_title("Satellite image")
        axes[row_index, 1].imshow(mask, cmap="gray"); axes[row_index, 1].set_title("Building mask")
        axes[row_index, 0].axis("off"); axes[row_index, 1].axis("off")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
