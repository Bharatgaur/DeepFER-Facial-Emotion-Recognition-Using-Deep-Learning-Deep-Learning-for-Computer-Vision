"""
DeepFER – preprocessing.py
============================
Handles all data loading, resizing, normalization, label-encoding, and
train/validation/test splitting for the Facial Emotion Recognition pipeline.

Author : DeepFER Team
Version: 1.0.0
"""

import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE      = (48, 48)          # Target spatial resolution (H × W)
CHANNELS      = 1                 # Grayscale for FER48 dataset
NUM_CLASSES   = 7
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ── Helper ────────────────────────────────────────────────────────────────────

def load_image(path: str, img_size: tuple = IMG_SIZE, grayscale: bool = True) -> np.ndarray:
    """
    Load a single image from *path*, resize to *img_size*, and return a
    float32 NumPy array normalised to [0, 1].

    Parameters
    ----------
    path      : Absolute / relative path to the image file.
    img_size  : (height, width) tuple.
    grayscale : Convert to grayscale if True (recommended for FER datasets).

    Returns
    -------
    np.ndarray of shape (H, W, 1) if grayscale else (H, W, 3), dtype float32.
    """
    mode = 'L' if grayscale else 'RGB'
    img  = Image.open(path).convert(mode).resize((img_size[1], img_size[0]))
    arr  = np.array(img, dtype=np.float32) / 255.0       # Normalise to [0, 1]
    if grayscale:
        arr = arr[..., np.newaxis]                        # Add channel dim -> (H,W,1)
    return arr


def load_dataset(data_dir: str,
                 img_size: tuple = IMG_SIZE,
                 grayscale: bool = True,
                 verbose: bool = True) -> tuple:
    """
    Walk *data_dir* (expected layout: data_dir/<emotion_class>/<image>),
    load every image, and return (X, y_str) arrays.

    Parameters
    ----------
    data_dir  : Root folder containing one sub-folder per emotion class.
    img_size  : Target (H, W) to resize every image.
    grayscale : Use grayscale channel.
    verbose   : Print per-class image counts.

    Returns
    -------
    X      : np.ndarray shape (N, H, W, C), float32, values in [0, 1].
    labels : list of str  – raw string labels aligned with X.
    """
    images, labels = [], []
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'}

    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        class_images = [
            f for f in os.listdir(class_dir)
            if os.path.splitext(f)[1].lower() in valid_ext
        ]
        if verbose:
            print(f"  [{class_name:>10}]  {len(class_images):>5} images")

        for fname in class_images:
            fpath = os.path.join(class_dir, fname)
            try:
                img = load_image(fpath, img_size=img_size, grayscale=grayscale)
                images.append(img)
                labels.append(class_name)
            except Exception as exc:
                print(f"    Skipping {fpath}: {exc}")

    X = np.array(images, dtype=np.float32)
    return X, labels


def encode_labels(labels: list) -> tuple:
    """
    Encode string labels to integer indices.

    Returns
    -------
    y_encoded : np.ndarray of int (N,)
    le        : fitted sklearn LabelEncoder (use le.classes_ for mapping)
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    return y_encoded, le


def split_data(X: np.ndarray,
               y: np.ndarray,
               val_size: float = 0.15,
               test_size: float = 0.10,
               random_state: int = 42) -> tuple:
    """
    Stratified split into train / validation / test sets.

    Parameters
    ----------
    val_size  : Fraction of the *whole* dataset held out for validation.
    test_size : Fraction of the *whole* dataset held out for test.

    Returns
    -------
    (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    test_frac_of_remaining = test_size / (1.0 - val_size)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=test_frac_of_remaining,
        stratify=y_train_val,
        random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def one_hot_encode(y: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Convert integer labels to one-hot vectors."""
    return tf.keras.utils.to_categorical(y, num_classes=num_classes)


def get_tf_datasets(train_dir: str,
                    val_dir: str,
                    img_size: tuple = IMG_SIZE,
                    batch_size: int = 64,
                    augment: bool = True) -> tuple:
    """
    Build tf.data.Dataset pipelines directly from directory structure.
    Preferred approach for large datasets (avoids loading all data into RAM).

    Parameters
    ----------
    train_dir  : Folder with sub-folders per class (training split).
    val_dir    : Folder with sub-folders per class (validation split).
    img_size   : (H, W) target size.
    batch_size : Mini-batch size.
    augment    : Whether to apply augmentation layers on the training set.

    Returns
    -------
    (train_ds, val_ds, class_names)
    """
    color_mode = 'grayscale'   # Change to 'rgb' for transfer-learning models

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        color_mode=color_mode,
        label_mode='categorical',
        shuffle=True,
        seed=42,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        color_mode=color_mode,
        label_mode='categorical',
        shuffle=False,
        seed=42,
    )

    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE

    # Normalise pixel values 0-255 -> 0.0-1.0
    norm = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (norm(x), y), num_parallel_calls=AUTOTUNE)
    val_ds   = val_ds.map(lambda x, y: (norm(x), y),   num_parallel_calls=AUTOTUNE)

    # Performance tuning
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    DATA_ROOT  = os.path.join(os.path.dirname(__file__), '..', 'raw_data', 'images', 'images')
    TRAIN_DIR  = os.path.join(DATA_ROOT, 'train')
    VAL_DIR    = os.path.join(DATA_ROOT, 'validation')

    print("\nLoading training images...")
    X_train, y_train_str = load_dataset(TRAIN_DIR, verbose=True)
    y_train_enc, le = encode_labels(y_train_str)

    print(f"\nX_train shape : {X_train.shape}")
    print(f"   Classes       : {list(le.classes_)}")
    print(f"   Label range   : {y_train_enc.min()} – {y_train_enc.max()}")

    print("\nBuilding tf.data pipelines...")
    train_ds, val_ds, classes = get_tf_datasets(TRAIN_DIR, VAL_DIR)
    print(f"   Class names   : {classes}")
    for batch_x, batch_y in train_ds.take(1):
        print(f"   Batch X shape : {batch_x.shape}")
        print(f"   Batch Y shape : {batch_y.shape}")
        print(f"   Pixel range   : [{batch_x.numpy().min():.3f}, {batch_x.numpy().max():.3f}]")
