from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, Model


def conv_block(x, filters: int, name: str):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    return layers.Activation("relu", name=f"{name}_relu2")(x)


def up_block(x, skip, filters: int, name: str):
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name=f"{name}_upsample")(x)
    x = layers.Concatenate(name=f"{name}_concat")([x, skip])
    return conv_block(x, filters, name)


def build_unet(input_shape=(256, 256, 3), base_filters: int = 32) -> Model:
    inputs = layers.Input(shape=input_shape, name="image")
    x = layers.Rescaling(1.0 / 255.0, name="input_rescaling")(inputs)

    c1 = conv_block(x, base_filters, "enc1")
    p1 = layers.MaxPooling2D(name="pool1")(c1)
    c2 = conv_block(p1, base_filters * 2, "enc2")
    p2 = layers.MaxPooling2D(name="pool2")(c2)
    c3 = conv_block(p2, base_filters * 4, "enc3")
    p3 = layers.MaxPooling2D(name="pool3")(c3)
    c4 = conv_block(p3, base_filters * 8, "enc4")
    p4 = layers.MaxPooling2D(name="pool4")(c4)
    bridge = conv_block(p4, base_filters * 16, "bridge")

    x = up_block(bridge, c4, base_filters * 8, "dec4")
    x = up_block(x, c3, base_filters * 4, "dec3")
    x = up_block(x, c2, base_filters * 2, "dec2")
    x = up_block(x, c1, base_filters, "dec1")
    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="building_probability")(x)
    return Model(inputs, outputs, name="AerialVision_UNet")


def build_resnet50_unet(
    input_shape=(256, 256, 3),
    weights: str | None = "imagenet",
    freeze_encoder: bool = True,
) -> Model:
    inputs = layers.Input(shape=input_shape, name="image")
    # SpaceNet RGB patches are stored as uint8. Keras applications expects 0..255
    # before ResNet50 preprocessing (BGR channel conversion + ImageNet means).
    resnet_input = layers.Lambda(
        lambda tensor: tf.keras.applications.resnet50.preprocess_input(
            tf.cast(tensor, tf.float32)
        ),
        name="resnet_preprocessing",
    )(inputs)
    encoder = tf.keras.applications.ResNet50(
        include_top=False,
        weights=weights,
        input_tensor=resnet_input,
    )
    encoder.trainable = not freeze_encoder

    # Stable feature taps from ResNet50 at 1/2, 1/4, 1/8, 1/16 and 1/32 resolution.
    s1 = encoder.get_layer("conv1_relu").output
    s2 = encoder.get_layer("conv2_block3_out").output
    s3 = encoder.get_layer("conv3_block4_out").output
    s4 = encoder.get_layer("conv4_block6_out").output
    bridge = encoder.get_layer("conv5_block3_out").output

    x = up_block(bridge, s4, 256, "res_dec4")
    x = up_block(x, s3, 128, "res_dec3")
    x = up_block(x, s2, 64, "res_dec2")
    x = up_block(x, s1, 32, "res_dec1")
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="res_final_upsample")(x)
    x = conv_block(x, 16, "res_final")
    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="building_probability")(x)
    return Model(inputs, outputs, name="AerialVision_ResNet50_UNet")


def build_model(model_name: str, input_shape=(256, 256, 3), weights: str = "imagenet", freeze_encoder: bool = True):
    if model_name == "unet":
        return build_unet(input_shape=input_shape)
    if model_name == "resnet50_unet":
        selected_weights = None if weights.lower() in {"none", "null", "random"} else weights
        return build_resnet50_unet(
            input_shape=input_shape,
            weights=selected_weights,
            freeze_encoder=freeze_encoder,
        )
    raise ValueError(f"Unknown model: {model_name}")
