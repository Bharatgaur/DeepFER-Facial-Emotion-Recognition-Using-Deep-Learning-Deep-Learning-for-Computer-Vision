"""
DeepFER – predict.py
=====================
Inference utilities for single-image and batch prediction.
Used by the Streamlit/Flask app and the real-time detector.

Author : DeepFER Team
Version: 1.0.0
"""

import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import load_image, EMOTION_LABELS, IMG_SIZE

ROOT       = os.path.join(os.path.dirname(__file__), '..')
MODEL_DIR  = os.path.join(ROOT, 'models')
VISUAL_DIR = os.path.join(ROOT, 'visuals')
os.makedirs(VISUAL_DIR, exist_ok=True)

# Emotion display labels (no emoji; plain text labels only)
EMOTION_EMOJI = {
    'angry':    '',
    'disgust':  '',
    'fear':     '',
    'happy':    '',
    'neutral':  '',
    'sad':      '',
    'surprise': '',
}

# Confidence colour thresholds for UI feedback
CONF_HIGH   = 0.70   # green  – reliable prediction
CONF_MEDIUM = 0.45   # yellow – uncertain
# below CONF_MEDIUM = red    - low confidence


def load_model_for_inference(model_path: str) -> tf.keras.Model:
    """
    Load a saved .keras / .h5 model from disk.

    Parameters
    ----------
    model_path : Full path to the saved model file.

    Returns
    -------
    Compiled tf.keras.Model ready for inference.
    """
    print(f"  Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"  Model loaded successfully.")
    return model


def preprocess_face(face_img: np.ndarray,
                    target_size: tuple = IMG_SIZE,
                    grayscale: bool = True) -> np.ndarray:
    """
    Prepare a raw face crop (from OpenCV or PIL) for model input.

    Parameters
    ----------
    face_img    : np.ndarray (H, W) or (H, W, 3), uint8 [0–255].
    target_size : (H, W) expected by the model.
    grayscale   : Convert to single-channel grayscale.

    Returns
    -------
    np.ndarray shape (1, H, W, C), float32 [0, 1] – ready for model.predict().
    """
    # Convert to PIL for clean resize
    pil_img = Image.fromarray(face_img)
    mode    = 'L' if grayscale else 'RGB'
    pil_img = pil_img.convert(mode).resize((target_size[1], target_size[0]))
    arr     = np.array(pil_img, dtype=np.float32) / 255.0

    if grayscale:
        arr = arr[..., np.newaxis]          # (H, W, 1)
    return np.expand_dims(arr, axis=0)      # (1, H, W, C)


def predict_emotion(model: tf.keras.Model,
                    face_img: np.ndarray,
                    class_names: list = EMOTION_LABELS,
                    target_size: tuple = IMG_SIZE,
                    grayscale: bool = True) -> dict:
    """
    Predict the emotion for a single face crop.

    Parameters
    ----------
    model      : Loaded tf.keras.Model.
    face_img   : Raw face crop (H, W) or (H, W, 3), uint8.
    class_names: Ordered list matching model output indices.
    target_size: (H, W) to resize the crop to.
    grayscale  : Whether model expects single-channel input.

    Returns
    -------
    dict with keys:
        emotion      – predicted class string
        confidence   – float [0, 1]
        all_probs    – dict {class_name: probability}
        emoji        – emotion emoji
        conf_level   – 'high' | 'medium' | 'low'
    """
    x     = preprocess_face(face_img, target_size=target_size, grayscale=grayscale)
    probs = model.predict(x, verbose=0)[0]           # shape (num_classes,)

    idx        = int(np.argmax(probs))
    emotion    = class_names[idx]
    confidence = float(probs[idx])

    all_probs = {cls: float(p) for cls, p in zip(class_names, probs)}

    if confidence >= CONF_HIGH:
        conf_level = 'high'
    elif confidence >= CONF_MEDIUM:
        conf_level = 'medium'
    else:
        conf_level = 'low'

    return {
        'emotion':    emotion,
        'confidence': confidence,
        'all_probs':  all_probs,
        'emoji':      EMOTION_EMOJI.get(emotion, ''),
        'conf_level': conf_level,
    }


def predict_from_path(model: tf.keras.Model,
                      img_path: str,
                      class_names: list = EMOTION_LABELS,
                      grayscale: bool = True) -> dict:
    """
    Load an image from *img_path* and return emotion prediction.

    Parameters
    ----------
    img_path : Path to a face image file (any PIL-supported format).

    Returns
    -------
    Same dict as predict_emotion().
    """
    pil_img  = Image.open(img_path).convert('RGB')
    face_arr = np.array(pil_img, dtype=np.uint8)
    return predict_emotion(model, face_arr, class_names, grayscale=grayscale)


def predict_batch(model: tf.keras.Model,
                  img_paths: list,
                  class_names: list = EMOTION_LABELS,
                  target_size: tuple = IMG_SIZE,
                  grayscale: bool = True) -> list:
    """
    Efficient batch inference for a list of image paths.

    Returns
    -------
    List of prediction dicts (same format as predict_emotion).
    """
    batch = []
    for path in img_paths:
        try:
            pil  = Image.open(path).convert('L' if grayscale else 'RGB')
            pil  = pil.resize((target_size[1], target_size[0]))
            arr  = np.array(pil, dtype=np.float32) / 255.0
            if grayscale:
                arr = arr[..., np.newaxis]
            batch.append(arr)
        except Exception as e:
            print(f"  Skipping {path}: {e}")
            batch.append(np.zeros((*target_size, 1 if grayscale else 3), dtype=np.float32))

    x        = np.array(batch)                         # (N, H, W, C)
    probs    = model.predict(x, verbose=0)             # (N, num_classes)
    results  = []
    for i, prob_vec in enumerate(probs):
        idx        = int(np.argmax(prob_vec))
        emotion    = class_names[idx]
        confidence = float(prob_vec[idx])
        conf_level = 'high' if confidence >= CONF_HIGH else (
                     'medium' if confidence >= CONF_MEDIUM else 'low')
        results.append({
            'path':       img_paths[i],
            'emotion':    emotion,
            'confidence': confidence,
            'all_probs':  {cls: float(p) for cls, p in zip(class_names, prob_vec)},
            'emoji':      EMOTION_EMOJI.get(emotion, ''),
            'conf_level': conf_level,
        })
    return results


def plot_probability_bar(result: dict,
                         ax: plt.Axes | None = None,
                         title: str = '') -> plt.Figure:
    """
    Plot a horizontal bar chart of emotion probabilities for one prediction.

    Parameters
    ----------
    result : Output dict from predict_emotion().
    ax     : Optional existing Axes to plot on.
    title  : Optional title override.

    Returns
    -------
    matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    else:
        fig = ax.figure

    emotions = list(result['all_probs'].keys())
    probs    = list(result['all_probs'].values())
    colours  = ['green' if e == result['emotion'] else 'steelblue' for e in emotions]

    bars = ax.barh(emotions, probs, color=colours)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
    t = title or f"Prediction: {result['emotion'].capitalize()} " \
                 f"({result['confidence']*100:.1f}%)"
    ax.set_title(t, fontsize=11)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


# ── CLI quick-predict ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    import glob, argparse

    parser = argparse.ArgumentParser(description='DeepFER - Quick Predict')
    parser.add_argument('--image', required=True, help='Path to face image')
    parser.add_argument('--model', default=None,
                        help='Path to .keras model (auto-finds latest if omitted)')
    args = parser.parse_args()

    if args.model:
        model_path = args.model
    else:
        found = sorted(glob.glob(os.path.join(MODEL_DIR, '*.keras')))
        if not found:
            raise FileNotFoundError("No .keras model found in models/. Train first.")
        model_path = found[-1]

    model = load_model_for_inference(model_path)
    result = predict_from_path(model, args.image)

    print(f"\n  Emotion    : {result['emotion']}")
    print(f"  Confidence : {result['confidence']*100:.2f}%  [{result['conf_level']}]")
    print(f"\n  All probabilities:")
    for cls, prob in sorted(result['all_probs'].items(), key=lambda x: -x[1]):
        bar = '█' * int(prob * 30)
        print(f"    {cls:>10} : {bar:<30} {prob*100:5.1f}%")
