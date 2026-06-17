"""
DeepFER - train.py
===================
End-to-end training script.  Supports:
  • Custom CNN  (grayscale 48x48)
  • MobileNetV2 (two-phase: head-only -> fine-tune)
  • VGG16       (two-phase: head-only -> fine-tune)

Usage (CLI):
    python train.py --model cnn      --epochs 60 --batch 64 --lr 0.001
    python train.py --model mobilenet --epochs 30 --batch 32 --lr 0.0001
    python train.py --model vgg16    --epochs 30 --batch 32 --lr 0.0001

Author : DeepFER Team
Version: 1.0.0
"""

import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
from datetime import datetime

# Make src/ importable
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import get_tf_datasets, IMG_SIZE, NUM_CLASSES
from augmentation import build_augmentation_layer, apply_augmentation_to_dataset, compute_class_weights
from model import (build_custom_cnn, build_mobilenet_model, unfreeze_mobilenet,
                   build_vgg16_model, unfreeze_vgg16)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.join(os.path.dirname(__file__), '..')
TRAIN_DIR  = os.path.join(ROOT, 'raw_data', 'images', 'images', 'train')
VAL_DIR    = os.path.join(ROOT, 'raw_data', 'images', 'images', 'validation')
MODEL_DIR  = os.path.join(ROOT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# ── Callbacks ─────────────────────────────────────────────────────────────────

def build_callbacks(model_name: str, patience: int = 10) -> list:
    """
    Return a list of Keras callbacks:
    • ModelCheckpoint – saves best weights by val_accuracy.
    • EarlyStopping   – halts training when val_loss stops improving.
    • ReduceLROnPlateau – halves LR when val_loss plateaus for 5 epochs.
    • TensorBoard     – optional; writes logs for TensorBoard UI.

    Parameters
    ----------
    model_name : Used to name checkpoint files.
    patience   : EarlyStopping patience (epochs with no improvement).

    Returns
    -------
    list of tf.keras.callbacks.Callback
    """
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    ckpt_path = os.path.join(MODEL_DIR, f'{model_name}_best_{ts}.keras')

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1,
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1,
    )
    log_dir = os.path.join(ROOT, 'logs', f'{model_name}_{ts}')
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
    )

    print(f"\n  Best checkpoint saved to: {ckpt_path}")
    return [checkpoint, early_stop, reduce_lr, tensorboard], ckpt_path


# ── Dataset helpers ────────────────────────────────────────────────────────────

def prepare_datasets(model_type: str,
                     batch_size: int) -> tuple:
    """
    Build tf.data pipelines appropriate for the chosen model type.
    - CNN      -> grayscale 48x48
    - MobileNet -> RGB 96x96
    - VGG16    -> RGB 48x48

    Returns
    -------
    (train_ds, val_ds, class_names, class_weights_dict)
    """
    if model_type == 'cnn':
        color_mode = 'grayscale'
        img_size   = (48, 48)
    elif model_type == 'mobilenet':
        color_mode = 'rgb'
        img_size   = (96, 96)
    else:  # vgg16
        color_mode = 'rgb'
        img_size   = (48, 48)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=img_size,
        batch_size=batch_size,
        color_mode=color_mode,
        label_mode='categorical',
        shuffle=True,
        seed=42,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        VAL_DIR,
        image_size=img_size,
        batch_size=batch_size,
        color_mode=color_mode,
        label_mode='categorical',
        shuffle=False,
        seed=42,
    )

    class_names = train_ds.class_names
    AUTOTUNE    = tf.data.AUTOTUNE

    # Normalise 0-255 -> 0-1
    norm       = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds   = train_ds.map(lambda x, y: (norm(x), y), num_parallel_calls=AUTOTUNE)
    val_ds     = val_ds.map(lambda x, y: (norm(x), y),   num_parallel_calls=AUTOTUNE)

    # Augmentation pipeline applied to training only
    aug_layer  = build_augmentation_layer()
    train_ds   = apply_augmentation_to_dataset(train_ds, aug_layer)

    val_ds     = val_ds.cache().prefetch(AUTOTUNE)

    # Compute class weights (handles class imbalance, especially 'disgust')
    # Approximate weights using known counts
    counts = np.array([3993, 436, 4103, 7164, 4982, 4938, 3205], dtype=np.float32)
    class_ids = np.concatenate([np.full(int(c), i) for i, c in enumerate(counts)])
    cw = compute_class_weights(class_ids, num_classes=NUM_CLASSES)

    return train_ds, val_ds, class_names, cw


# ── Main training routine ──────────────────────────────────────────────────────

def train_custom_cnn(epochs: int = 60, batch_size: int = 64, lr: float = 1e-3):
    """Train the custom CNN model end-to-end."""
    print("\n" + "=" * 60)
    print("  Training: Custom CNN")
    print("=" * 60)

    train_ds, val_ds, class_names, cw = prepare_datasets('cnn', batch_size)

    model = build_custom_cnn(
        input_shape=(48, 48, 1),
        num_classes=NUM_CLASSES,
        learning_rate=lr,
    )
    model.summary()

    cbs, ckpt_path = build_callbacks('custom_cnn', patience=12)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cbs,
        class_weight=cw,
        verbose=1,
    )

    _save_history(history, 'custom_cnn')
    print(f"\n  Custom CNN training complete. Best model saved to: {ckpt_path}")
    return model, history


def train_mobilenet(epochs_head: int = 15,
                    epochs_finetune: int = 20,
                    batch_size: int = 32,
                    lr_head: float = 1e-4,
                    lr_finetune: float = 1e-5):
    """
    Two-phase MobileNetV2 training:
    Phase 1 – head only (few epochs)
    Phase 2 – unfreeze top layers, fine-tune with low LR
    """
    print("\n" + "=" * 60)
    print("  Training: MobileNetV2 (Transfer Learning)")
    print("=" * 60)

    train_ds, val_ds, class_names, cw = prepare_datasets('mobilenet', batch_size)
    model, base_model = build_mobilenet_model(
        input_shape=(96, 96, 3),
        num_classes=NUM_CLASSES,
        learning_rate=lr_head,
    )
    model.summary()

    # Phase 1: head only
    print("\n  ── Phase 1: Training head only ──")
    cbs1, _ = build_callbacks('mobilenet_phase1', patience=8)
    hist1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs_head, callbacks=cbs1,
        class_weight=cw, verbose=1,
    )

    # Phase 2: fine-tune
    print("\n  ── Phase 2: Fine-tuning top layers ──")
    unfreeze_mobilenet(model, base_model, fine_tune_from=100, new_lr=lr_finetune)
    cbs2, ckpt2 = build_callbacks('mobilenet_finetune', patience=10)
    hist2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs_finetune, callbacks=cbs2,
        class_weight=cw, verbose=1,
    )

    # Merge histories
    history = _merge_histories(hist1, hist2)
    _save_history_dict(history, 'mobilenet')
    print(f"\n  MobileNetV2 training complete. Best model saved to: {ckpt2}")
    return model, history


def train_vgg16(epochs_head: int = 15,
                epochs_finetune: int = 20,
                batch_size: int = 32,
                lr_head: float = 1e-4,
                lr_finetune: float = 1e-5):
    """Two-phase VGG16 training."""
    print("\n" + "=" * 60)
    print("  Training: VGG16 (Transfer Learning)")
    print("=" * 60)

    train_ds, val_ds, class_names, cw = prepare_datasets('vgg16', batch_size)
    model, base_model = build_vgg16_model(
        input_shape=(48, 48, 3),
        num_classes=NUM_CLASSES,
        learning_rate=lr_head,
    )
    model.summary()

    # Phase 1
    print("\n  ── Phase 1: Training head only ──")
    cbs1, _ = build_callbacks('vgg16_phase1', patience=8)
    hist1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs_head, callbacks=cbs1,
        class_weight=cw, verbose=1,
    )

    # Phase 2
    print("\n  ── Phase 2: Fine-tuning block4+ layers ──")
    unfreeze_vgg16(model, base_model, fine_tune_at_block=4, new_lr=lr_finetune)
    cbs2, ckpt2 = build_callbacks('vgg16_finetune', patience=10)
    hist2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs_finetune, callbacks=cbs2,
        class_weight=cw, verbose=1,
    )

    history = _merge_histories(hist1, hist2)
    _save_history_dict(history, 'vgg16')
    print(f"\n  VGG16 training complete. Best model saved to: {ckpt2}")
    return model, history


# ── Utility helpers ────────────────────────────────────────────────────────────

def _save_history(history, name: str):
    """Persist Keras History object as JSON for later plotting."""
    path = os.path.join(MODEL_DIR, f'{name}_history.json')
    with open(path, 'w') as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f, indent=2)
    print(f"  History saved to: {path}")


def _save_history_dict(hist_dict: dict, name: str):
    path = os.path.join(MODEL_DIR, f'{name}_history.json')
    with open(path, 'w') as f:
        json.dump(hist_dict, f, indent=2)
    print(f"  History saved to: {path}")


def _merge_histories(h1, h2) -> dict:
    """Concatenate two Keras History objects into one plain dict."""
    merged = {}
    for key in h1.history:
        merged[key] = [float(v) for v in h1.history[key]] + \
                      [float(v) for v in h2.history.get(key, [])]
    return merged


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepFER – Training Script')
    parser.add_argument('--model',  default='cnn',
                        choices=['cnn', 'mobilenet', 'vgg16'],
                        help='Model architecture to train')
    parser.add_argument('--epochs', type=int, default=60,
                        help='Max training epochs')
    parser.add_argument('--batch',  type=int, default=64,
                        help='Mini-batch size')
    parser.add_argument('--lr',     type=float, default=1e-3,
                        help='Initial learning rate')
    args = parser.parse_args()

    tf.random.set_seed(42)

    if args.model == 'cnn':
        train_custom_cnn(epochs=args.epochs, batch_size=args.batch, lr=args.lr)
    elif args.model == 'mobilenet':
        train_mobilenet(batch_size=args.batch,
                        lr_head=args.lr, lr_finetune=args.lr / 10)
    else:
        train_vgg16(batch_size=args.batch,
                    lr_head=args.lr, lr_finetune=args.lr / 10)
