"""
DeepFER – model.py
===================
Defines THREE model architectures for Facial Emotion Recognition:
  1. Custom CNN         – built from scratch, lightweight, grayscale input
  2. MobileNetV2        – transfer learning, good speed/accuracy trade-off
  3. VGG16              – transfer learning, higher accuracy baseline

All builders return compiled tf.keras.Model objects ready to call .fit().

Author : DeepFER Team
Version: 1.0.0
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.applications import MobileNetV2, VGG16
import os

# ── Shared constants ───────────────────────────────────────────────────────────
NUM_CLASSES   = 7
IMG_H, IMG_W  = 48, 48
L2_REG        = 1e-4              # L2 weight decay to combat overfitting


# ──────────────────────────────────────────────────────────────────────────────
# 1.  CUSTOM CNN
# ──────────────────────────────────────────────────────────────────────────────

def build_custom_cnn(
    input_shape: tuple = (IMG_H, IMG_W, 1),
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = 0.5,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Custom 4-block CNN architecture designed for 48×48 grayscale face images.

    Architecture
    ------------
    Block 1: Conv(32)  -> BN -> ReLU -> Conv(32)  -> BN -> ReLU -> MaxPool -> Dropout(0.25)
    Block 2: Conv(64)  -> BN -> ReLU -> Conv(64)  -> BN -> ReLU -> MaxPool -> Dropout(0.25)
    Block 3: Conv(128) -> BN -> ReLU -> Conv(128) -> BN -> ReLU -> MaxPool -> Dropout(0.25)
    Block 4: Conv(256) -> BN -> ReLU ->                          MaxPool -> Dropout(0.25)
    Head   : Flatten -> Dense(512) -> BN -> ReLU -> Dropout(0.5) -> Dense(7, softmax)

    Design choices:
    • BatchNormalization after each Conv speeds convergence and acts as regulariser.
    • Two consecutive Conv layers per block increase receptive field without deep pooling.
    • Dropout applied after each block + before final dense to prevent overfitting.
    • L2 regularisation on Conv and Dense kernels.

    Parameters
    ----------
    input_shape   : (H, W, C) – default (48, 48, 1) for grayscale FER.
    num_classes   : Number of emotion classes.
    dropout_rate  : Dropout probability for the dense head.
    learning_rate : Adam initial LR.

    Returns
    -------
    Compiled tf.keras.Model.
    """
    reg = regularizers.l2(L2_REG)

    inp = layers.Input(shape=input_shape, name='input')

    # ── Block 1 ───────────────────────────────────────────────────────────────
    x = layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=reg, name='b1_conv1')(inp)
    x = layers.BatchNormalization(name='b1_bn1')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=reg, name='b1_conv2')(x)
    x = layers.BatchNormalization(name='b1_bn2')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2), name='b1_pool')(x)       # 48->24
    x = layers.Dropout(0.25, name='b1_drop')(x)

    # ── Block 2 ───────────────────────────────────────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=reg, name='b2_conv1')(x)
    x = layers.BatchNormalization(name='b2_bn1')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=reg, name='b2_conv2')(x)
    x = layers.BatchNormalization(name='b2_bn2')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2), name='b2_pool')(x)       # 24->12
    x = layers.Dropout(0.25, name='b2_drop')(x)

    # ── Block 3 ───────────────────────────────────────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=reg, name='b3_conv1')(x)
    x = layers.BatchNormalization(name='b3_bn1')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=reg, name='b3_conv2')(x)
    x = layers.BatchNormalization(name='b3_bn2')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2), name='b3_pool')(x)       # 12->6
    x = layers.Dropout(0.25, name='b3_drop')(x)

    # ── Block 4 ───────────────────────────────────────────────────────────────
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=reg, name='b4_conv1')(x)
    x = layers.BatchNormalization(name='b4_bn1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2), name='b4_pool')(x)       # 6->3
    x = layers.Dropout(0.25, name='b4_drop')(x)

    # ── Classification head ───────────────────────────────────────────────────
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(512, kernel_regularizer=reg, name='fc1')(x)
    x = layers.BatchNormalization(name='fc1_bn')(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate, name='fc1_drop')(x)
    out = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = models.Model(inputs=inp, outputs=out, name='DeepFER_CustomCNN')
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 2.  MobileNetV2 – Transfer Learning (Recommended for real-time)
# ──────────────────────────────────────────────────────────────────────────────

def build_mobilenet_model(
    input_shape: tuple = (96, 96, 3),    # MobileNet prefers RGB ≥ 96px
    num_classes: int = NUM_CLASSES,
    fine_tune_from: int = 100,            # Freeze layers below this index
    learning_rate: float = 1e-4,
    dropout_rate: float = 0.4,
) -> tf.keras.Model:
    """
    MobileNetV2 fine-tuned for facial emotion recognition.

    Strategy
    --------
    Phase 1 – Feature extraction: freeze the entire MobileNetV2 base, train
              only the new classification head for a few epochs so the random
              weights don't propagate large gradients into the pre-trained base.
    Phase 2 – Fine-tuning: unfreeze layers above *fine_tune_from* and train
              with a very small LR to adapt high-level representations to FER.

    Parameters
    ----------
    input_shape    : (H, W, 3) – RGB input; must be ≥ 96px for MobileNetV2.
    num_classes    : Number of emotion classes.
    fine_tune_from : Layer index above which to unfreeze during fine-tuning.
    learning_rate  : Initial learning rate (used for head-only phase).
    dropout_rate   : Dropout before the output Dense layer.

    Returns
    -------
    (model, base_model) – both returned so caller can manipulate base_model
    for Phase-2 fine-tuning.
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
    )
    # ── Phase 1: freeze entire base ───────────────────────────────────────────
    base_model.trainable = False

    inp  = layers.Input(shape=input_shape, name='input')

    # MobileNetV2 expects pixels in [-1, 1]; our pipeline delivers [0, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inp * 255.0)

    x = base_model(x, training=False)                        # (B, 3, 3, 1280)
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(L2_REG), name='fc1')(x)
    x = layers.BatchNormalization(name='fc1_bn')(x)
    x = layers.Dropout(dropout_rate, name='fc1_drop')(x)
    out = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = models.Model(inputs=inp, outputs=out, name='DeepFER_MobileNetV2')
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, base_model


def unfreeze_mobilenet(model: tf.keras.Model,
                       base_model: tf.keras.Model,
                       fine_tune_from: int = 100,
                       new_lr: float = 1e-5) -> None:
    """
    Unfreeze the top layers of *base_model* for Phase-2 fine-tuning.
    Call this AFTER Phase-1 training has converged.

    Parameters
    ----------
    model         : Full compiled model returned by build_mobilenet_model().
    base_model    : The MobileNetV2 base (second return value).
    fine_tune_from: Unfreeze layers whose index ≥ fine_tune_from.
    new_lr        : Very small LR to avoid destroying pre-trained weights.
    """
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_from]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=new_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"  MobileNetV2 layers {fine_tune_from}+ unfrozen.  LR -> {new_lr}")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  VGG16 – Transfer Learning (Higher accuracy baseline)
# ──────────────────────────────────────────────────────────────────────────────

def build_vgg16_model(
    input_shape: tuple = (48, 48, 3),
    num_classes: int = NUM_CLASSES,
    fine_tune_at_block: int = 4,         # Unfreeze from block4_conv1
    learning_rate: float = 1e-4,
    dropout_rate: float = 0.5,
) -> tuple:
    """
    VGG16 fine-tuned for facial emotion recognition.

    Parameters
    ----------
    input_shape      : (H, W, 3) – RGB input.
    num_classes      : Number of emotion classes.
    fine_tune_at_block: VGG16 block from which to unfreeze.
                        4 = unfreeze block4 + block5 only.
    learning_rate    : Initial LR for head-only phase.
    dropout_rate     : Dropout probability.

    Returns
    -------
    (model, base_model)
    """
    base_model = VGG16(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
    )
    # Freeze all base layers initially
    base_model.trainable = False

    inp = layers.Input(shape=input_shape, name='input')
    # VGG16 expects BGR mean-subtracted pixels; preprocess_input handles that.
    x = tf.keras.applications.vgg16.preprocess_input(inp * 255.0)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(L2_REG), name='fc1')(x)
    x = layers.BatchNormalization(name='fc1_bn')(x)
    x = layers.Dropout(dropout_rate, name='fc1_drop')(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(L2_REG), name='fc2')(x)
    x = layers.Dropout(dropout_rate * 0.6, name='fc2_drop')(x)
    out = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = models.Model(inputs=inp, outputs=out, name='DeepFER_VGG16')
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, base_model


def unfreeze_vgg16(model: tf.keras.Model,
                   base_model: tf.keras.Model,
                   fine_tune_at_block: int = 4,
                   new_lr: float = 1e-5) -> None:
    """Unfreeze VGG16 layers at and above the given block for fine-tuning."""
    base_model.trainable = True
    freeze_until = f'block{fine_tune_at_block}_conv1'
    set_trainable = False

    for layer in base_model.layers:
        if layer.name == freeze_until:
            set_trainable = True
        layer.trainable = set_trainable

    model.compile(
        optimizer=optimizers.Adam(learning_rate=new_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"  VGG16 layers from block{fine_tune_at_block} unfrozen.  LR -> {new_lr}")


# ── Utility: summary + count parameters ───────────────────────────────────────

def model_summary(model: tf.keras.Model) -> None:
    """Print model summary with total / trainable parameter counts."""
    model.summary(line_length=90)
    total     = model.count_params()
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"\n  Total params     : {total:,}")
    print(f"  Trainable params : {trainable:,}")
    print(f"  Frozen params    : {total - trainable:,}\n")


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  1. Custom CNN")
    print("=" * 60)
    cnn = build_custom_cnn()
    model_summary(cnn)

    print("=" * 60)
    print("  2. MobileNetV2")
    print("=" * 60)
    mob, mob_base = build_mobilenet_model(input_shape=(96, 96, 3))
    model_summary(mob)

    print("=" * 60)
    print("  3. VGG16")
    print("=" * 60)
    vgg, vgg_base = build_vgg16_model(input_shape=(48, 48, 3))
    model_summary(vgg)
