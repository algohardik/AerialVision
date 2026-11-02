# AerialVision — Building Footprint Segmentation with TensorFlow

A reproducible computer-vision project that segments building footprints from high-resolution aerial/satellite imagery. It uses the **SpaceNet 2 Building Detection** dataset, converts GeoJSON footprint polygons into pixel masks, trains TensorFlow/Keras segmentation models, evaluates them with overlap metrics, and provides command-line and Streamlit inference. The best reported experiment is a fine-tuned ResNet50 U-Net with **0.5891 test IoU** and **0.7414 test Dice** at a probability threshold of 0.60.

> This repository contains the complete code and dataset download/preprocessing pipeline. The raw SpaceNet imagery is **not bundled** because even the recommended Paris training archive is approximately 5.3 GB compressed. Download it using the steps below.

## What this project demonstrates

- Geospatial image/label handling: GeoTIFF, GeoJSON, CRS transformation and polygon rasterization.
- Data engineering: image tiling, mask generation, dataset manifests and scene-level train/validation/test splitting.
- Deep learning: U-Net built from scratch and U-Net with a ResNet50 encoder.
- Training: BCE + Dice loss, augmentation, early stopping, learning-rate scheduling and checkpoints.
- Evaluation: pixel-level precision, recall, IoU/Jaccard, Dice and qualitative error analysis.
- Deployment: sliding-window inference and a Streamlit demo.

## 1. Dataset

### SpaceNet 2 — AOI 3 (Paris)

The default dataset is SpaceNet 2 AOI 3 (Paris). Its official training archive is listed as approximately **5.3 GB** and contains 200 m × 200 m image tiles with building-footprint annotations. The official dataset page lists the archive and the SpaceNet Jaccard/IoU evaluation metric:

- Dataset page: [https://spacenet.ai/spacenet-buildings-dataset-v2/](https://www.google.com/url?q=https%3A%2F%2Fspacenet.ai%2Fspacenet-buildings-dataset-v2%2F)
- Download archive: `s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_3_Paris.tar.gz`
- Dataset citation: Van Etten, A., Lindenbaum, D., & Bacastow, T. M. (2018). *SpaceNet: A Remote Sensing Dataset and Challenge Series*. arXiv:1807.01232.
- Dataset license: SpaceNet dataset is published under CC BY-SA 4.0. Review the official page and license before redistribution.

The official page states that the dataset is hosted as an AWS public dataset and that an AWS account is required. Install and configure AWS CLI according to AWS instructions. If you already have the archive, place it in `data/raw/` with the original filename.

### Download and extract (Windows PowerShell / macOS / Linux)

From the repository root:

```
python download_data.py --aoi Paris --extract

```

Other AOIs are supported:

```
python download_data.py --aoi Khartoum --extract
python download_data.py --aoi Vegas --extract
python download_data.py --aoi Shanghai --extract

```

Paris is recommended for the first run because its official training archive is smaller than the Vegas and Shanghai archives. Extraction requires additional free disk space. Do not commit raw data or generated patches to GitHub.

If downloading manually, save the archive under `data/raw/`, then run:

```
python download_data.py --aoi Paris --extract

```

## 2. Environment setup

### Recommended local environment

Use **Python 3.10**. TensorFlow 2.15.1 is pinned to keep the project environment reproducible and to avoid NumPy 2.x compatibility issues.

Windows PowerShell:

```
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/check_setup.py

```

macOS/Linux:

```
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/check_setup.py

```

For a GPU, use a compatible Linux/Colab environment and verify `tf.config.list_physical_devices('GPU')`. TensorFlow GPU installation depends on the operating system, CUDA/cuDNN and driver versions. Do not assume that a normal Windows pip install will enable NVIDIA GPU acceleration. CPU training is possible but can be slow.

### Google Colab

1. Upload/clone this repository into Colab.
2. Use a GPU runtime.
3. Install the extra packages:

```
!pip install -r requirements-colab.txt
!python src/check_setup.py

```

4. Download/extract SpaceNet to the repository's `data/raw/` directory, or mount Drive and set `--raw-dir` to the extracted folder.

The ResNet50 ImageNet weights may be downloaded automatically the first time the ResNet50 model is built. Use `--weights none` for an offline/randomly initialized encoder, although ImageNet pretraining is recommended for the transfer-learning experiment.

## 3. Quick pipeline smoke test (without SpaceNet download)

Before downloading the full dataset, you can create 30 tiny **synthetic** GeoTIFF/GeoJSON pairs and test whether preprocessing works:

```
python src/create_demo_data.py --scenes 30
python src/prepare_dataset.py --raw-dir data/raw/synthetic_demo --output-dir data/processed/synthetic_demo
python src/visualize_dataset.py --manifest data/processed/synthetic_demo/manifest.csv --split train --count 4 --output reports/synthetic_samples.png

```

This only tests file discovery, CRS alignment, polygon rasterization and patch generation. It is not satellite data, should not be used to train the final model, and must not be used for performance claims.

## 4. Prepare the dataset

The preprocessing script discovers matching `RGB-PanSharpen/*.tif` images and `geojson/buildings/*.geojson` building labels recursively. It transforms vector geometries into the raster CRS, rasterizes the building footprints, then produces 256 × 256 paired image/mask PNG patches.

First, use a limited number of source scenes to validate the pipeline:

```
python src/prepare_dataset.py --max-scenes 100 --patch-size 256 --stride 256

```

When that succeeds, preprocess the full AOI:

```
python src/prepare_dataset.py --patch-size 256 --stride 256

```

For overlap between neighboring patches, use `--stride 128`. This produces more training examples but also increases disk usage and near-duplicate patches. Keep the same patch size for model training and inference.

Output structure:

```
data/processed/spacenet_paris/
├── manifest.csv
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── masks/
    ├── train/
    ├── val/
    └── test/

```

The split is made at the **source-scene level before patch extraction**, so patches from one source GeoTIFF cannot appear in both training and validation/test. It is still a random scene split from one AOI; it does not establish cross-city generalization.

Inspect sample image/mask pairs:

```
python src/visualize_dataset.py --split train --count 6

```

The output is saved to `reports/dataset_samples.png`.

## 5. Train the baseline U-Net

Start with the baseline model:

```
python src/train.py --model unet --epochs 40 --batch-size 8 --learning-rate 0.001 --run-name unet

```

The best model is saved to:

```
models/unet/best_model.keras

```

The baseline U-Net includes an input rescaling layer, encoder/decoder blocks and skip connections.

## 6. Train U-Net with a ResNet50 encoder

This is the transfer-learning model. It uses the ImageNet-pretrained ResNet50 encoder (frozen by default) and a trainable segmentation decoder:

```
python src/train.py --model resnet50_unet --weights imagenet --epochs 40 --batch-size 8 --learning-rate 0.0001 --run-name resnet50_unet

```

To train the encoder from the beginning of this run as well, use a smaller learning rate:

```
python src/train.py --model resnet50_unet --weights imagenet --unfreeze-encoder --epochs 30 --batch-size 4 --learning-rate 0.00001 --run-name resnet50_unet_finetuned

```

These are separate experiments. Compare them on the same held-out test split; do not choose the model based on the test set during repeated tuning. Use validation IoU for checkpoint selection.

### Training artifacts

Each run saves:

- `best_model.keras` — checkpoint with the highest validation IoU.
- `final_model.keras` — final epoch model.
- `training_history.csv` — epoch-by-epoch metrics.
- `run_config.json` — training arguments and sample counts.
- TensorBoard logs under `logs/<run-name>/`.

Plot learning curves:

```
python src/plot_training.py models/unet/training_history.csv --output reports/unet_curves.png

```

## 7. Evaluate on the held-out test split

### Select the probability threshold using validation data

The fine-tuned ResNet50 U-Net was evaluated on the validation split across thresholds from 0.30 to 0.70. The best validation pixel IoU was obtained at **0.60** (validation IoU: **0.5911**). This threshold was fixed before evaluating the held-out test split.

```bash
python src/tune_threshold.py \
  --manifest data/processed/spacenet_paris/manifest.csv \
  --model-path models/resnet50_unet_finetuned/best_model.keras \
  --thresholds 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 \
  --batch-size 8 \
  --output results/resnet50_unet_finetuned/best_threshold.json
```

### Held-out test evaluation

Evaluate each architecture on the same held-out test split. The baseline U-Net and frozen-encoder ResNet50 U-Net results used a threshold of 0.50. The fine-tuned ResNet50 U-Net uses the validation-selected threshold of 0.60:

```bash
python src/evaluate.py \
  --manifest data/processed/spacenet_paris/manifest.csv \
  --model-path models/unet/best_model.keras \
  --threshold 0.50 \
  --batch-size 8 \
  --output-dir results/unet \
  --visualize 12

python src/evaluate.py \
  --manifest data/processed/spacenet_paris/manifest.csv \
  --model-path models/resnet50_unet/best_model.keras \
  --threshold 0.50 \
  --batch-size 8 \
  --output-dir results/resnet50_unet \
  --visualize 12

python src/evaluate.py \
  --manifest data/processed/spacenet_paris/manifest.csv \
  --model-path models/resnet50_unet_finetuned/best_model.keras \
  --threshold 0.60 \
  --batch-size 8 \
  --output-dir results/resnet50_unet_finetuned_threshold_060 \
  --visualize 12
```

Each evaluation writes `metrics.json` and example figures containing the input image, ground-truth mask and predicted mask.

Metrics:

- **IoU/Jaccard:** TP / (TP + FP + FN)
- **Dice/F1 overlap:** 2TP / (2TP + FP + FN)
- **Precision:** TP / (TP + FP)
- **Recall:** TP / (TP + FN)
- Pixel accuracy is included but should not be the primary metric because background pixels dominate many image tiles.

The evaluation code reports global pixel metrics across the whole test split and mean per-patch Dice. These are pixel-level segmentation metrics, not SpaceNet's official instance/polygon matching score. Do not describe them as official SpaceNet challenge scores.

## 8. Predict on a new image

```
# Frozen ResNet50 U-Net (defaults to threshold 0.50)
python src/predict.py \
  --image path/to/aerial_image.tif \
  --model-path models/resnet50_unet/best_model.keras \
  --output-dir predictions

# Fine-tuned ResNet50 U-Net (defaults to validation-selected threshold 0.60)
python src/predict.py \
  --image path/to/aerial_image.tif \
  --model-path models/resnet50_unet_finetuned/best_model.keras \
  --output-dir predictions

# Pass --threshold to override the default for any checkpoint.

```

The script performs overlapping sliding-window inference and saves:

- `*_building_mask.png`
- `*_probability.png`
- `*_overlay.png`
- `*_result.png`

Input can be a 3-band RGB GeoTIFF, PNG or JPEG. For an ordinary PNG/JPEG, the model can still run, but the image should be reasonably similar to the training imagery in resolution, color and viewpoint.

## 9. Run the Streamlit demo

After training the models, launch the application from the repository root:

```bash
streamlit run app.py
```

The app provides a model selector for U-Net, ResNet50 U-Net and fine-tuned ResNet50 U-Net. The fine-tuned model is selected by default and uses the validation-selected probability threshold of **0.60**; the other models default to **0.50**. The threshold, sliding-window stride and overlay opacity can be adjusted in the sidebar. Upload an RGB image or 3-band GeoTIFF to view and download the predicted building mask and segmentation overlay.

## 10. Experiments and results

All three architectures were trained and evaluated using 256 × 256 patches from SpaceNet 2 AOI 3 (Paris), with a scene-level train/validation/test split. The fine-tuned ResNet50 U-Net threshold was selected using validation IoU and then evaluated on the held-out test split.

### Held-out test-set comparison

| Experiment | Encoder | Threshold | Test IoU | Test Dice | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| Baseline U-Net | U-Net from scratch | 0.50 | 0.5074 | 0.6732 | 0.6021 | 0.7634 |
| Transfer learning | ImageNet ResNet50, frozen | 0.50 | 0.5723 | 0.7280 | 0.6351 | 0.8528 |
| Fine-tuning | ImageNet ResNet50, fine-tuned | 0.50 | 0.5800 | 0.7341 | 0.6444 | 0.8529 |
| **Fine-tuning (validation-selected threshold)** | **ImageNet ResNet50, fine-tuned** | **0.60** | **0.5891** | **0.7414** | **0.6755** | **0.8215** |

The best result in these experiments was achieved by the fine-tuned ResNet50 U-Net at threshold **0.60**, with test IoU **0.5891** and Dice **0.7414**. Compared with the same checkpoint evaluated at threshold 0.50, threshold 0.60 improved IoU by 0.0091 and Dice by 0.0073. Precision increased while recall decreased, reflecting a more conservative building prediction threshold.

These results are pixel-level metrics aggregated over the held-out test pixels. They are not the official SpaceNet instance-level building detection score. The evaluation uses one AOI, so it does not demonstrate generalization to other cities.

### Regenerate comparison artifacts

After all four evaluation metric files exist, regenerate the committed comparison table and charts with:

```bash
python src/compare_models.py
```

The script reads the four `metrics.json` files under `results/` and regenerates `results/model_comparison.csv`, `results/model_comparison.png`, and `results/final_model_comparison.png`.

### Further experiments

1. Compare BCE-only, Dice-only and BCE + Dice loss.
2. Examine false positives (roads, parking lots, bright roofs) and false negatives (small or shadowed buildings).
3. Measure inference latency per 256 × 256 patch and model file size.
4. Evaluate on a geographically distinct AOI, if its imagery and labels are available and can be processed consistently.

## 11. Troubleshooting

**No image/GeoJSON pairs found**

- Confirm the archive extracted successfully.
- The expected SpaceNet folders contain `RGB-PanSharpen/` and `geojson/buildings/`.
- Pass the parent extracted directory using `--raw-dir` if you extracted elsewhere.

**Not enough disk space**

- Start with `--max-scenes 50` or `100`.
- Use `--stride 256` rather than overlapping patches.
- Delete intermediate processed patches only after backing up any outputs you need.

**TensorFlow reports no GPU**

- Confirm the runtime has a supported NVIDIA driver and CUDA/cuDNN combination.
- In Colab, choose Runtime → Change runtime type → GPU.
- `python src/check_setup.py` prints the TensorFlow devices it can access.

**Out-of-memory during training**

- Reduce `--batch-size` to 4, 2 or 1.
- Keep patch size at 256 for the first experiment.
- Start with baseline U-Net before ResNet50 U-Net.

**ImageNet weight download fails**

- Retry with internet enabled, or use `--weights none` (less desirable for the transfer-learning experiment).

**Masks appear empty or displaced**

- Inspect the source GeoTIFF CRS and GeoJSON CRS.
- Inspect `reports/dataset_samples.png`.
- Do not train until image/mask alignment is visually verified. Incorrect CRS assumptions can silently corrupt labels.

## Project structure

The following is the tracked GitHub repository layout. `results/` is committed because it contains compact evaluation metrics, comparison tables, and selected prediction visualizations. Large source data, trained checkpoints, and runtime artifacts are generated locally and intentionally excluded by `.gitignore`.

```text
AerialVision/
├── .gitignore
├── README.md
├── app.py
├── download_data.py
├── requirements.txt
├── requirements-colab.txt
├── results/
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── final_model_comparison.png
│   ├── unet/
│   │   ├── metrics.json
│   │   └── test_example_*.png
│   ├── resnet50_unet/
│   │   ├── metrics.json
│   │   └── test_example_*.png
│   ├── resnet50_unet_finetuned/
│   │   ├── best_threshold.json
│   │   ├── metrics.json
│   │   └── test_example_*.png
│   └── resnet50_unet_finetuned_threshold_060/
│       ├── metrics.json
│       └── test_example_*.png
└── src/
    ├── __init__.py
    ├── check_setup.py
    ├── compare_models.py
    ├── create_demo_data.py
    ├── data_utils.py
    ├── evaluate.py
    ├── inference_utils.py
    ├── input_pipeline.py
    ├── losses.py
    ├── models.py
    ├── paths.py
    ├── plot_training.py
    ├── predict.py
    ├── prepare_dataset.py
    ├── train.py
    ├── tune_threshold.py
    └── visualize_dataset.py
```

The following directories are created locally as needed and are **not committed**:

- `data/raw/` and `data/processed/` — downloaded SpaceNet archives and generated image/mask patches.
- `models/` — trained `.keras` checkpoints.
- `reports/` and `logs/` — local inspection plots and TensorBoard logs.
- `predictions/` — outputs produced by command-line inference.

