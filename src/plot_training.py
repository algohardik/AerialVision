from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from training_history.csv")
    parser.add_argument("history", type=Path, help="Path to training_history.csv")
    parser.add_argument("--output", type=Path, default=Path("reports/training_curves.png"))
    args = parser.parse_args()
    frame = pd.read_csv(args.history)
    metrics = [name for name in ["loss", "iou", "dice_coefficient", "precision", "recall"] if name in frame.columns]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 3.2 * len(metrics)), squeeze=False)
    for index, metric in enumerate(metrics):
        axes[index, 0].plot(frame[metric], label=f"train_{metric}")
        val_name = f"val_{metric}"
        if val_name in frame:
            axes[index, 0].plot(frame[val_name], label=val_name)
        axes[index, 0].set_xlabel("Epoch")
        axes[index, 0].set_ylabel(metric)
        axes[index, 0].legend()
        axes[index, 0].grid(True, alpha=0.25)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
