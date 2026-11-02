
from io import BytesIO
from pathlib import Path
import tempfile
import zipfile
import shutil

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

from src.inference_utils import load_image_bytes, make_overlay, predict_array
from src.models import build_model


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"


# ---------------- MODEL OPTIONS ----------------

MODEL_OPTIONS = {
    "U-Net": {
        "path": MODEL_DIR / "unet" / "best_model.keras",
        "type": "unet",
    },
    "ResNet50 U-Net": {
        "path": MODEL_DIR / "resnet50_unet" / "best_model.keras",
        "type": "resnet50_unet",
    },
    "Fine-tuned ResNet50 U-Net": {
        "path": MODEL_DIR / "resnet50_unet_finetuned" / "best_model.keras",
        "type": "resnet50_unet_finetuned",
    },
}


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="AerialVision | Building Segmentation",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ AerialVision")
st.caption("Satellite Building Footprint Segmentation")

st.write(
    "Upload an aerial or satellite image to detect building footprints "
    "using deep learning segmentation models."
)


# ---------------- MODEL LOADING ----------------

@st.cache_resource
def load_selected_model(model_path: str, model_type: str):

    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    # Load the standard U-Net checkpoint.
    if model_type == "unet":

        model = tf.keras.models.load_model(
            path,
            compile=False,
            safe_mode=False,
        )

    else:

        # Rebuild the ResNet50 U-Net architecture.
        input_size = 256

        model = build_model(
            model_name="resnet50_unet",
            input_shape=(input_size, input_size, 3),
            weights="none",
            freeze_encoder=(model_type == "resnet50_unet"),
        )

        # Extract trained weights from the .keras archive.
        temporary_path = None

        try:

            with zipfile.ZipFile(path, "r") as archive:

                weight_files = [
                    name for name in archive.namelist()
                    if name.endswith("model.weights.h5")
                ]

                if not weight_files:
                    raise FileNotFoundError(
                        f"No model.weights.h5 found in {path}"
                    )

                with tempfile.NamedTemporaryFile(
                    suffix=".weights.h5",
                    delete=False,
                ) as temporary_file:

                    temporary_path = Path(temporary_file.name)

                    with archive.open(weight_files[0]) as source:
                        shutil.copyfileobj(source, temporary_file)

            model.load_weights(temporary_path)

        finally:

            if temporary_path and temporary_path.exists():
                temporary_path.unlink()

    return model


# ---------------- SIDEBAR SETTINGS ----------------

st.sidebar.header("Inference Settings")

selected_name = st.sidebar.selectbox(
    "Select segmentation model",
    list(MODEL_OPTIONS.keys()),
    index=2,
)

selected_model = MODEL_OPTIONS[selected_name]

model_path = selected_model["path"]
model_type = selected_model["type"]


# Check whether checkpoint exists.
if not model_path.exists():

    st.error(f"Model checkpoint not found: {model_path}")
    st.stop()


# Load selected model.
try:

    with st.spinner(f"Loading {selected_name}..."):

        model = load_selected_model(
            str(model_path),
            model_type,
        )

except Exception as error:

    st.error(f"Could not load model: {error}")
    st.stop()


expected_size = int(model.input_shape[1] or 256)


st.sidebar.success("Model loaded successfully")

st.sidebar.write(f"**Selected:** {selected_name}")

st.sidebar.write(
    f"**Input size:** {expected_size} × {expected_size}"
)


# ---------------- THRESHOLD SETTINGS ----------------

# Validation-set optimized threshold for fine-tuned ResNet50 U-Net.
# Other models retain the original default threshold.

default_threshold = (
    0.60
    if model_type == "resnet50_unet_finetuned"
    else 0.50
)

st.sidebar.write(
    f"**Default threshold:** {default_threshold:.2f}"
)

threshold = st.sidebar.slider(
    "Building probability threshold",
    min_value=0.10,
    max_value=0.90,
    value=default_threshold,
    step=0.05,
    help=(
        "The fine-tuned ResNet50 U-Net uses a default threshold "
        "of 0.60, selected using validation-set IoU."
    ),
)


# Sliding-window stride.
stride = st.sidebar.selectbox(
    "Sliding-window stride",
    [64, 128, expected_size],
    index=1,
)


# Overlay opacity.
alpha = st.sidebar.slider(
    "Overlay opacity",
    min_value=0.10,
    max_value=0.90,
    value=0.45,
    step=0.05,
)


# ---------------- IMAGE UPLOAD ----------------

st.subheader("Upload Satellite Image")

uploaded = st.file_uploader(
    "Choose an RGB image or 3-band GeoTIFF",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
)


if uploaded is not None:

    try:

        # Read uploaded image.
        image = load_image_bytes(
            uploaded.getvalue(),
            uploaded.name,
        )

        st.write(
            f"Image dimensions: {image.shape[1]} × {image.shape[0]}"
        )


        # ---------------- PREDICTION ----------------

        with st.spinner("Detecting building footprints..."):

            probabilities, mask = predict_array(
                model,
                image,
                patch_size=expected_size,
                stride=min(stride, expected_size),
                threshold=threshold,
            )


            # Generate overlay.
            overlay = make_overlay(
                image,
                mask,
                alpha=alpha,
            )


        st.success("Segmentation completed!")


        # ---------------- VISUALIZATION ----------------

        st.subheader("Segmentation Results")

        col1, col2, col3 = st.columns(3)


        col1.image(
            image,
            caption="Original Satellite Image",
            use_container_width=True,
        )


        col2.image(
            mask * 255,
            caption="Predicted Building Mask",
            clamp=True,
            use_container_width=True,
        )


        col3.image(
            overlay,
            caption="Building Segmentation Overlay",
            use_container_width=True,
        )


        # ---------------- PREDICTION METRICS ----------------

        st.subheader("Prediction Summary")

        coverage = 100.0 * float(mask.mean())

        metric1, metric2, metric3 = st.columns(3)


        metric1.metric(
            "Building Coverage",
            f"{coverage:.2f}%",
        )


        metric2.metric(
            "Building Pixels",
            f"{int(mask.sum()):,}",
        )


        metric3.metric(
            "Total Pixels",
            f"{mask.size:,}",
        )


        st.caption(
            f"Model: {selected_name} | "
            f"Threshold: {threshold:.2f} | "
            f"Stride: {min(stride, expected_size)}"
        )


        # ---------------- DOWNLOAD BUILDING MASK ----------------

        mask_buffer = BytesIO()

        Image.fromarray(
            (mask * 255).astype(np.uint8)
        ).save(
            mask_buffer,
            format="PNG",
        )


        st.download_button(
            "Download Building Mask",
            data=mask_buffer.getvalue(),
            file_name="building_mask.png",
            mime="image/png",
        )


        # ---------------- DOWNLOAD OVERLAY ----------------

        overlay_buffer = BytesIO()

        Image.fromarray(
            np.asarray(overlay, dtype=np.uint8)
        ).save(
            overlay_buffer,
            format="PNG",
        )


        st.download_button(
            "Download Segmentation Overlay",
            data=overlay_buffer.getvalue(),
            file_name="building_overlay.png",
            mime="image/png",
        )


    except Exception as error:

        st.error(f"Could not process this image: {error}")


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "AerialVision | Deep Learning Based Building Footprint Segmentation | "
    "TensorFlow and Keras"
)
