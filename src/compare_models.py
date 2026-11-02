from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

EXPERIMENTS = [
    {
        "experiment": "Baseline U-Net",
        "architecture": "U-Net from scratch",
        "metrics": "unet/metrics.json",
        "label": "U-Net",
    },
    {
        "experiment": "Transfer learning",
        "architecture": "ImageNet ResNet50 (frozen)",
        "metrics": "resnet50_unet/metrics.json",
        "label": "ResNet50\n(frozen)",
    },
    {
        "experiment": "Fine-tuning (threshold 0.50)",
        "architecture": "ImageNet ResNet50 (fine-tuned)",
        "metrics": "resnet50_unet_finetuned/metrics.json",
        "label": "Fine-tuned\n(0.50)",
    },
    {
        "experiment": "Fine-tuning (validation-selected threshold)",
        "architecture": "ImageNet ResNet50 (fine-tuned)",
        "metrics": "resnet50_unet_finetuned_threshold_060/metrics.json",
        "label": "Fine-tuned\n(0.60)",
    },
]

CSV_FIELDS = [
    "experiment",
    "architecture",
    "threshold",
    "test_patches",
    "iou",
    "dice",
    "precision",
    "recall",
    "pixel_accuracy",
    "mean_per_patch_dice",
]


def load_comparison_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for experiment in EXPERIMENTS:
        metrics_path = RESULTS / experiment["metrics"]
        if not metrics_path.is_file():
            raise FileNotFoundError(
                f"Missing metrics file: {metrics_path}. Run the corresponding evaluation first."
            )
        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        row = {
            "experiment": experiment["experiment"],
            "architecture": experiment["architecture"],
            "threshold": metrics["threshold"],
            "test_patches": metrics["test_patches"],
            "iou": metrics["iou"],
            "dice": metrics["dice"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "pixel_accuracy": metrics["pixel_accuracy"],
            "mean_per_patch_dice": metrics["mean_per_patch_dice"],
        }
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    output_path = RESULTS / "model_comparison.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {output_path.relative_to(ROOT)}")


def write_metric_dashboard(rows: list[dict[str, object]]) -> None:
    labels = [experiment["label"] for experiment in EXPERIMENTS]
    metric_specs = [
        ("iou", "Test IoU"),
        ("dice", "Test Dice"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for axis, (key, title) in zip(axes.flat, metric_specs):
        values = [float(row[key]) for row in rows]
        bars = axis.bar(x, values)
        axis.set_title(title, fontsize=12, weight="bold")
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks(x, labels, fontsize=8)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=8)
    fig.suptitle("AerialVision — Held-out Test Performance", fontsize=15, weight="bold")
    fig.tight_layout()
    output_path = RESULTS / "model_comparison.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path.relative_to(ROOT)}")


def write_final_iou_dice_chart(rows: list[dict[str, object]]) -> None:
    labels = [experiment["label"] for experiment in EXPERIMENTS]
    x = np.arange(len(rows))
    width = 0.36
    iou = [float(row["iou"]) for row in rows]
    dice = [float(row["dice"]) for row in rows]

    fig, axis = plt.subplots(figsize=(11, 5.5))
    bars_iou = axis.bar(x - width / 2, iou, width, label="IoU")
    bars_dice = axis.bar(x + width / 2, dice, width, label="Dice")
    axis.set_title("AerialVision — Final Model Comparison", fontsize=15, weight="bold")
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(x, labels)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend()
    axis.bar_label(bars_iou, labels=[f"{value:.3f}" for value in iou], padding=2, fontsize=8)
    axis.bar_label(bars_dice, labels=[f"{value:.3f}" for value in dice], padding=2, fontsize=8)
    fig.tight_layout()
    output_path = RESULTS / "final_model_comparison.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path.relative_to(ROOT)}")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = load_comparison_rows()
    write_csv(rows)
    write_metric_dashboard(rows)
    write_final_iou_dice_chart(rows)
    print("\nComparison data:")
    for row in rows:
        print(
            f"{row['experiment']}: threshold={float(row['threshold']):.2f}, "
            f"IoU={float(row['iou']):.4f}, Dice={float(row['dice']):.4f}"
        )


if __name__ == "__main__":
    main()
