"""
DeepFER - realtime/realtime_detector.py
========================================
Real-time facial emotion recognition using OpenCV.

Pipeline
--------
1. Capture frame from webcam (or video file).
2. Detect faces using Haar-cascade frontal face detector.
3. For each face ROI: preprocess, run model inference, overlay result.
4. Display FPS, emotion label, confidence bar on frame.
5. Press 'q' to quit, 's' to save a screenshot.

Usage
-----
    python realtime/realtime_detector.py --model models/custom_cnn_best_*.keras
    python realtime/realtime_detector.py --model models/custom_cnn_best_*.keras --source 0
    python realtime/realtime_detector.py --model models/custom_cnn_best_*.keras --source video.mp4

Author : DeepFER Team
Version: 1.0.0
"""

import os
import sys
import time
import argparse
import glob
import numpy as np
import cv2
import tensorflow as tf

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import (predict_emotion, load_model_for_inference,
                     EMOTION_EMOJI, CONF_HIGH, CONF_MEDIUM)

ROOT     = os.path.join(os.path.dirname(__file__), '..')
MODEL_DIR = os.path.join(ROOT, 'models')
SCREENSHOT_DIR = os.path.join(ROOT, 'visuals', 'screenshots')
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Colour palette (BGR) per emotion for the bounding box
EMOTION_COLOURS = {
    'angry':    (0,   0,   220),   # Red
    'disgust':  (0,   140, 0),     # Dark Green
    'fear':     (128, 0,   128),   # Purple
    'happy':    (0,   200, 50),    # Green
    'neutral':  (200, 200, 200),   # Light Grey
    'sad':      (200, 100, 0),     # Dark Blue-ish
    'surprise': (0,   200, 255),   # Yellow
}
CONF_COLOUR = {
    'high':   (0, 200, 50),    # Green
    'medium': (0, 200, 255),   # Yellow
    'low':    (0, 0, 220),     # Red
}


# ── Haar cascade loader ────────────────────────────────────────────────────────

def load_face_detector() -> cv2.CascadeClassifier:
    """
    Load the Haar-cascade frontal face detector bundled with OpenCV.
    Falls back to a local XML file if the bundled path fails.
    """
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(
            f"Failed to load Haar cascade from {cascade_path}. "
            "Install opencv-contrib-python or place the XML in the current directory."
        )
    print(f"  Face detector loaded: {cascade_path}")
    return detector


# ── Frame annotation helpers ───────────────────────────────────────────────────

def draw_face_box(frame: np.ndarray,
                  x: int, y: int, w: int, h: int,
                  result: dict) -> None:
    """
    Draw a colour-coded bounding box + emotion label on *frame* (in-place).

    Layout:
    A label bar showing the emotion name and confidence percentage is
    drawn directly above the detected face bounding box.
    """
    emotion    = result['emotion']
    confidence = result['confidence']
    conf_level = result['conf_level']

    box_colour  = EMOTION_COLOURS.get(emotion, (255, 255, 255))
    text_colour = CONF_COLOUR.get(conf_level, (255, 255, 255))

    # Bounding box
    thickness = 2
    cv2.rectangle(frame, (x, y), (x + w, y + h), box_colour, thickness)

    # Label background bar
    label      = f"{emotion.upper()}  {confidence*100:.1f}%"
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    font_thick = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thick)
    bar_y1     = max(y - th - 10, 0)
    bar_y2     = y
    cv2.rectangle(frame, (x, bar_y1), (x + tw + 10, bar_y2), box_colour, -1)
    cv2.putText(frame, label,
                (x + 5, y - 4),
                font, font_scale, (0, 0, 0), font_thick, cv2.LINE_AA)

    # Confidence bar (bottom of face box)
    bar_w  = int(w * confidence)
    bar_h  = 6
    cv2.rectangle(frame, (x, y + h + 2), (x + w, y + h + 2 + bar_h),
                  (60, 60, 60), -1)                          # background
    cv2.rectangle(frame, (x, y + h + 2), (x + bar_w, y + h + 2 + bar_h),
                  text_colour, -1)                           # filled portion


def draw_hud(frame: np.ndarray, fps: float, n_faces: int) -> None:
    """Draw HUD overlay: FPS counter, face count, key hints."""
    h, w = frame.shape[:2]

    # Semi-transparent top-left panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 70), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"DeepFER - Real-Time", (8, 20),
                font, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}   Faces: {n_faces}", (8, 45),
                font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "'q' quit   's' screenshot", (8, 65),
                font, 0.42, (150, 150, 150), 1, cv2.LINE_AA)


def draw_all_probs(frame: np.ndarray,
                   result: dict,
                   x_offset: int = 10,
                   y_offset: int = 90) -> None:
    """
    Draw a mini probability bar chart on the right side of the frame for
    the most recently detected face.
    """
    if not result:
        return
    font       = cv2.FONT_HERSHEY_SIMPLEX
    bar_max_w  = 100
    row_h      = 18
    y          = y_offset

    for cls, prob in sorted(result['all_probs'].items(), key=lambda kv: EMOTION_LABELS.index(kv[0])):
        bar_w  = int(prob * bar_max_w)
        colour = EMOTION_COLOURS.get(cls, (180, 180, 180))
        highlighted = cls == result['emotion']
        thick = 2 if highlighted else 1

        # Mini bar
        cv2.rectangle(frame, (x_offset, y - 12), (x_offset + bar_max_w, y + 2),
                      (40, 40, 40), -1)
        cv2.rectangle(frame, (x_offset, y - 12), (x_offset + bar_w, y + 2),
                      colour, -1)
        # Label
        lbl = f"{cls[:4]:>4} {prob*100:.0f}%"
        cv2.putText(frame, lbl, (x_offset + bar_max_w + 5, y),
                    font, 0.38,
                    (255, 255, 0) if highlighted else (180, 180, 180),
                    thick, cv2.LINE_AA)
        y += row_h


# ── Main detection loop ────────────────────────────────────────────────────────

def run_realtime_detection(model_path: str,
                           source: int | str = 0,
                           grayscale: bool = True,
                           scale_factor: float = 1.3,
                           min_neighbours: int = 5,
                           min_face_size: int = 48) -> None:
    """
    Main real-time emotion detection loop.

    Parameters
    ----------
    model_path    : Path to a trained .keras model.
    source        : Camera index (0 = default) or path to a video file.
    grayscale     : Whether the model expects grayscale (True for custom CNN).
    scale_factor  : Haar cascade scale factor (1.1 to 1.5).
    min_neighbours: Haar cascade minimum neighbours (robustness vs sensitivity).
    min_face_size : Minimum face pixel size to detect.
    """
    # Load model and detector
    model    = load_model_for_inference(model_path)
    detector = load_face_detector()

    # Video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    print(f"\n  Capture opened. Press 'q' to quit, 's' for screenshot.\n")

    fps_history  = []
    last_result  = {}            # Most recent prediction (shown in prob bar)
    screenshot_n = 0

    while True:
        t_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            print("  End of stream or cannot read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbours,
            minSize=(min_face_size, min_face_size),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        n_faces = len(faces) if isinstance(faces, np.ndarray) else 0

        for (x, y, w, h) in (faces if n_faces > 0 else []):
            # Extract face ROI
            face_roi = gray[y:y + h, x:x + w] if grayscale else frame[y:y + h, x:x + w, ::-1]

            result = predict_emotion(
                model, face_roi,
                class_names=EMOTION_LABELS,
                target_size=(48, 48),
                grayscale=grayscale,
            )
            last_result = result
            draw_face_box(frame, x, y, w, h, result)

        # HUD & probability sidebar
        t_end = time.perf_counter()
        fps_history.append(1.0 / max(t_end - t_start, 1e-9))
        if len(fps_history) > 30:
            fps_history.pop(0)
        fps = np.mean(fps_history)

        draw_hud(frame, fps, n_faces)
        if last_result:
            draw_all_probs(frame, last_result, x_offset=frame.shape[1] - 165, y_offset=100)

        cv2.imshow('DeepFER - Real-Time Emotion Recognition', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("  Quit signal received.")
            break
        elif key == ord('s'):
            screenshot_n += 1
            fname = os.path.join(SCREENSHOT_DIR, f'screenshot_{screenshot_n:03d}.png')
            cv2.imwrite(fname, frame)
            print(f"  Screenshot saved to: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("  Real-time detection stopped.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepFER Real-Time Detector')
    parser.add_argument('--model', default=None,
                        help='Path to .keras model (auto-finds latest if omitted)')
    parser.add_argument('--source', default='0',
                        help='Camera index (0) or video file path')
    parser.add_argument('--rgb', action='store_true',
                        help='Use RGB input (for MobileNet / VGG16 models)')
    args = parser.parse_args()

    # Auto-find model
    if args.model:
        mpath = args.model
    else:
        found = sorted(glob.glob(os.path.join(MODEL_DIR, '*.keras')))
        if not found:
            raise FileNotFoundError("No .keras model in models/.  Train first.")
        mpath = found[-1]
        print(f"  Auto-selected model: {mpath}")

    src = int(args.source) if args.source.isdigit() else args.source

    run_realtime_detection(
        model_path=mpath,
        source=src,
        grayscale=not args.rgb,
    )
