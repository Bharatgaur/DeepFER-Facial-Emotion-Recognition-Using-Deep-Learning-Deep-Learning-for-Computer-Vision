"""
DeepFER – evaluate.py
======================
Comprehensive model evaluation:
  • Accuracy / Precision / Recall / F1-score (per-class + macro)
  • Confusion matrix (normalised heatmap)
  • Training vs Validation loss & accuracy curves
  • Prediction examples on random validation samples

Author : DeepFER Team
Version: 1.0.0
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_recall_fscore_support)

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import get_tf_datasets, IMG_SIZE

ROOT      = os.path.join(os.path.dirname(__file__), '..')
VAL_DIR   = os.path.join(ROOT, 'raw_data', 'images', 'images', 'validation')
VISUAL_DIR = os.path.join(ROOT, 'visuals')
MODEL_DIR  = os.path.join(ROOT, 'models')
os.makedirs(VISUAL_DIR, exist_ok=True)

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
PALETTE        = sns.color_palette('husl', n_colors=7)


# ── Core evaluation ────────────────────────────────────────────────────────────

def evaluate_model(model: tf.keras.Model,
                   val_ds: tf.data.Dataset,
                   class_names: list,
                   model_name: str = 'model',
                   save: bool = True) -> dict:
    """
    Run full evaluation on *val_ds* using *model*.

    Steps
    -----
    1. Collect all predictions and true labels.
    2. Compute per-class precision, recall, F1.
    3. Plot confusion matrix.
    4. Return metrics dict.

    Parameters
    ----------
    model      : Trained tf.keras.Model.
    val_ds     : Normalised, un-shuffled validation dataset.
    class_names: Ordered list of class name strings.
    model_name : Tag used for saved file names.
    save       : Whether to save plots to VISUAL_DIR.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, report
    """
    print(f"\n  Evaluating model: {model_name}")

    y_true, y_pred = [], []
    for x_batch, y_batch in val_ds:
        preds = model.predict(x_batch, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(y_batch.numpy(), axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    print(f"\n  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n  Per-class Classification Report:\n")
    print(report)

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    plot_confusion_matrix(y_true, y_pred, class_names,
                          model_name=model_name, save=save)

    metrics = {
        'accuracy':  float(acc),
        'precision': float(prec),
        'recall':    float(rec),
        'f1':        float(f1),
        'report':    report,
    }

    # Persist metrics JSON
    if save:
        path = os.path.join(MODEL_DIR, f'{model_name}_metrics.json')
        with open(path, 'w') as f:
            json.dump({k: v for k, v in metrics.items() if k != 'report'}, f, indent=2)
        print(f"\n  Metrics saved to {path}")

    return metrics


# ── Confusion matrix ───────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: list,
                          model_name: str = 'model',
                          normalise: bool = True,
                          save: bool = True) -> None:
    """
    Plot a heatmap confusion matrix.

    Parameters
    ----------
    normalise : If True, show row-normalised fractions (reveals per-class accuracy).
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalise:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt, vmax = '.2f', 1.0
        title_suffix = '(Normalised)'
    else:
        cm_display = cm
        fmt, vmax  = 'd', cm.max()
        title_suffix = '(Counts)'

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_display,
        annot=True, fmt=fmt, cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, vmin=0, vmax=vmax, ax=ax,
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'DeepFER Confusion Matrix – {model_name} {title_suffix}', fontsize=13)
    plt.tight_layout()

    if save:
        path = os.path.join(VISUAL_DIR, f'confusion_matrix_{model_name}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Confusion matrix saved to {path}")
    plt.close()


# ── Training history curves ────────────────────────────────────────────────────

def plot_training_history(history_path: str | dict,
                          model_name: str = 'model',
                          save: bool = True) -> None:
    """
    Plot accuracy and loss curves from a saved JSON history file or dict.

    Parameters
    ----------
    history_path : Path to <model>_history.json, OR a plain history dict.
    """
    if isinstance(history_path, str):
        with open(history_path) as f:
            hist = json.load(f)
    else:
        hist = history_path

    epochs = range(1, len(hist['accuracy']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Training History – {model_name}', fontsize=14)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    ax1.plot(epochs, hist['accuracy'],     label='Train Accuracy',  color='royalblue', lw=2)
    ax1.plot(epochs, hist['val_accuracy'], label='Val Accuracy',    color='tomato',    lw=2, linestyle='--')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ── Loss ──────────────────────────────────────────────────────────────────
    ax2.plot(epochs, hist['loss'],     label='Train Loss',  color='royalblue', lw=2)
    ax2.plot(epochs, hist['val_loss'], label='Val Loss',    color='tomato',    lw=2, linestyle='--')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Categorical Cross-Entropy')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(VISUAL_DIR, f'training_history_{model_name}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Training history plot saved to {path}")
    plt.close()


# ── Prediction examples ────────────────────────────────────────────────────────

def visualise_predictions(model: tf.keras.Model,
                           val_ds: tf.data.Dataset,
                           class_names: list,
                           n: int = 16,
                           model_name: str = 'model',
                           save: bool = True) -> None:
    """
    Sample *n* images from *val_ds*, run inference, and display a grid
    with true vs predicted labels.  Green title = correct, Red = wrong.
    """
    images, y_true, y_pred = [], [], []
    for x_batch, y_batch in val_ds.take(2):
        preds = model.predict(x_batch, verbose=0)
        images.extend(x_batch.numpy())
        y_true.extend(np.argmax(y_batch.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    images  = np.array(images[:n])
    y_true  = np.array(y_true[:n])
    y_pred  = np.array(y_pred[:n])

    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    fig.suptitle(f'Prediction Examples – {model_name}', fontsize=13)

    cmap = 'gray' if images.shape[-1] == 1 else None
    for idx, ax in enumerate(axes.flat):
        if idx >= n:
            ax.axis('off')
            continue
        ax.imshow(images[idx].squeeze(), cmap=cmap)
        true_lbl = class_names[y_true[idx]]
        pred_lbl = class_names[y_pred[idx]]
        colour   = 'green' if y_true[idx] == y_pred[idx] else 'red'
        ax.set_title(f'T: {true_lbl}\nP: {pred_lbl}',
                     fontsize=8, color=colour, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    if save:
        path = os.path.join(VISUAL_DIR, f'prediction_examples_{model_name}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Prediction examples saved to {path}")
    plt.close()


# ── Class distribution bar chart ──────────────────────────────────────────────

def plot_class_distribution(save: bool = True) -> None:
    """Plot bar chart of training class distribution to show imbalance."""
    train_counts = [3993, 436, 4103, 7164, 4982, 4938, 3205]
    val_counts   = [960,  111, 1018, 1825, 1216, 1139,  797]

    x = np.arange(len(EMOTION_LABELS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, train_counts, width, label='Train', color='steelblue')
    bars2 = ax.bar(x + width/2, val_counts,   width, label='Validation', color='salmon')

    ax.set_xlabel('Emotion Class', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.set_title('DeepFER Dataset – Class Distribution', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(EMOTION_LABELS, rotation=20, ha='right')
    ax.legend()
    ax.bar_label(bars1, padding=3, fontsize=8)
    ax.bar_label(bars2, padding=3, fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(VISUAL_DIR, 'class_distribution.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Class distribution saved to {path}")
    plt.close()


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("  Generating class distribution chart …")
    plot_class_distribution()

    # If a trained model exists, load and evaluate
    import glob
    model_files = glob.glob(os.path.join(MODEL_DIR, '*.keras'))
    if model_files:
        latest = sorted(model_files)[-1]
        print(f"\n  Loading model: {latest}")
        model = tf.keras.models.load_model(latest)
        _, val_ds, class_names = get_tf_datasets(
            os.path.join(ROOT, 'raw_data', 'images', 'images', 'train'),
            VAL_DIR, batch_size=64
        )
        evaluate_model(model, val_ds, class_names, model_name='loaded_model')
        visualise_predictions(model, val_ds, class_names, model_name='loaded_model')

    # If history JSON exists, plot curves
    hist_files = glob.glob(os.path.join(MODEL_DIR, '*_history.json'))
    for hf in hist_files:
        name = os.path.basename(hf).replace('_history.json', '')
        print(f"\n  Plotting training history: {name}")
        plot_training_history(hf, model_name=name)
