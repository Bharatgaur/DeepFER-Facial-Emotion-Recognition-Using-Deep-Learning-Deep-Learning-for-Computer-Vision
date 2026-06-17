"""
DeepFER – augmentation.py
==========================
Defines augmentation pipelines (Keras layers + tf.data) that are applied
ONLY to the training split to improve model generalization.

Why augmentation?
-----------------
Facial expression datasets are relatively small (tens of thousands of images).
Without augmentation the CNN memorises the training set (overfitting).
By generating plausibly-varied views at every epoch the model learns
*emotion-invariant* features rather than specific pixel patterns.

Author : DeepFER Team
Version: 1.0.0
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Keras Sequential Augmentation Layer ───────────────────────────────────────

def build_augmentation_layer(
    rotation_range: float = 0.15,
    zoom_range: float = 0.15,
    horizontal_flip: bool = True,
    brightness_range: tuple = (0.80, 1.20),
    contrast_range: float = 0.10,
    translation_range: float = 0.10,
    noise_stddev: float = 0.02,
) -> tf.keras.Sequential:
    """
    Build and return a Keras Sequential model composed entirely of
    preprocessing / augmentation layers.

    These layers are *training-only* – during inference (model.predict,
    model.evaluate) they pass data through unchanged.

    Parameters
    ----------
    rotation_range    : Max fraction of 2*pi to rotate (e.g. 0.15 -> approx 27 degrees).
    zoom_range        : Max fractional zoom amount.
    horizontal_flip   : Randomly mirror left and right (valid for faces).
    brightness_range  : (lower, upper) multiplier for brightness jitter.
    contrast_range    : Contrast factor for random contrast.
    translation_range : Max fractional translation (height & width).
    noise_stddev      : Std-dev of Gaussian noise added per pixel.

    Returns
    -------
    tf.keras.Sequential  –  augmentation pipeline.
    """
    layers = [
        # Random horizontal flip (most natural augmentation for faces)
        tf.keras.layers.RandomFlip("horizontal") if horizontal_flip
        else tf.keras.layers.Lambda(lambda x: x),

        # Random rotation (slight head tilt)
        tf.keras.layers.RandomRotation(factor=rotation_range, fill_mode='nearest'),

        # Random zoom (simulates subject distance variation)
        tf.keras.layers.RandomZoom(
            height_factor=(-zoom_range, zoom_range),
            width_factor=(-zoom_range, zoom_range),
            fill_mode='nearest',
        ),

        # Random translation (subject not always centred)
        tf.keras.layers.RandomTranslation(
            height_factor=translation_range,
            width_factor=translation_range,
            fill_mode='nearest',
        ),

        # Brightness jitter (lighting condition variation)
        tf.keras.layers.RandomBrightness(factor=brightness_range),

        # Contrast jitter (camera / monitor variation)
        tf.keras.layers.RandomContrast(factor=contrast_range),
    ]

    aug_model = tf.keras.Sequential(layers, name='augmentation')
    return aug_model


# ── tf.data-compatible augmentation function ──────────────────────────────────

def augment_fn(image: tf.Tensor, label: tf.Tensor,
               aug_layer: tf.keras.Sequential) -> tuple:
    """
    Apply *aug_layer* to a single (image, label) pair.
    Use with dataset.map() for on-the-fly augmentation.
    """
    image = aug_layer(image, training=True)
    # Clip to valid range after brightness / contrast shifts
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def apply_augmentation_to_dataset(
    dataset: tf.data.Dataset,
    aug_layer: tf.keras.Sequential,
    buffer_size: int = 1000,
) -> tf.data.Dataset:
    """
    Wrap a tf.data.Dataset with augmentation.
    Augmentation is applied element-wise so each epoch sees a fresh sample.

    Parameters
    ----------
    dataset     : Normalised training dataset (images in [0,1]).
    aug_layer   : Augmentation pipeline from build_augmentation_layer().
    buffer_size : Shuffle buffer size.

    Returns
    -------
    Augmented tf.data.Dataset.
    """
    AUTOTUNE = tf.data.AUTOTUNE
    augmented = dataset.map(
        lambda x, y: augment_fn(x, y, aug_layer),
        num_parallel_calls=AUTOTUNE,
    )
    return augmented.shuffle(buffer_size).prefetch(AUTOTUNE)


# ── Visualisation helper ───────────────────────────────────────────────────────

def visualise_augmentation(
    original_img: np.ndarray,
    aug_layer: tf.keras.Sequential,
    n_variants: int = 9,
    emotion_name: str = 'Sample',
    save_path: str | None = None,
) -> None:
    """
    Plot one original image alongside *n_variants* augmented versions.

    Parameters
    ----------
    original_img : np.ndarray (H, W, 1) or (H, W, 3), float32 [0,1].
    aug_layer    : Augmentation pipeline.
    n_variants   : Number of augmented copies to show.
    emotion_name : Title annotation.
    save_path    : If given, save the figure to this path.
    """
    cols = n_variants + 1
    fig, axes = plt.subplots(1, cols, figsize=(cols * 2, 2.5))
    fig.suptitle(f'Augmentation demo – "{emotion_name}"', fontsize=13, y=1.02)

    cmap = 'gray' if original_img.shape[-1] == 1 else None

    # Original
    axes[0].imshow(original_img.squeeze(), cmap=cmap)
    axes[0].set_title('Original', fontsize=9)
    axes[0].axis('off')

    # Augmented variants
    img_batch = tf.expand_dims(original_img, axis=0)          # (1,H,W,C)
    for i in range(1, cols):
        aug = aug_layer(img_batch, training=True).numpy()[0]   # (H,W,C)
        aug = np.clip(aug, 0, 1)
        axes[i].imshow(aug.squeeze(), cmap=cmap)
        axes[i].set_title(f'Aug #{i}', fontsize=9)
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved augmentation plot to {save_path}")
    plt.close()


# ── Compute class weights for imbalanced dataset ──────────────────────────────

def compute_class_weights(y_train: np.ndarray, num_classes: int = 7) -> dict:
    """
    Compute inverse-frequency class weights so that minority classes
    (e.g. 'disgust' with only ~436 samples) are not underweighted during
    training.

    Returns
    -------
    dict  {class_index: weight_float}
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.arange(num_classes)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train,
    )
    return dict(zip(classes, weights))


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocessing import load_image, EMOTION_LABELS

    TRAIN_DIR = os.path.join(
        os.path.dirname(__file__), '..', 'raw_data', 'images', 'images', 'train'
    )
    SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'visuals')
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load one sample from each class and visualise augmentation
    aug = build_augmentation_layer()
    for cls in EMOTION_LABELS:
        cls_dir = os.path.join(TRAIN_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        sample = os.listdir(cls_dir)[0]
        img = load_image(os.path.join(cls_dir, sample))
        save_path = os.path.join(SAVE_DIR, f'aug_{cls}.png')
        visualise_augmentation(img, aug, n_variants=6,
                                emotion_name=cls, save_path=save_path)
        print(f"  Augmented sample saved for class: {cls}")
