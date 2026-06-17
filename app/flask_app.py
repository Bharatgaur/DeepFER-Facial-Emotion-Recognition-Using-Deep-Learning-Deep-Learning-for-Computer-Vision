"""
DeepFER - app/flask_app.py
============================
Flask REST API + minimal HTML frontend for Facial Emotion Recognition.

Endpoints
---------
GET  /                     -> Serves index.html (upload form)
POST /predict              -> JSON API: upload image, returns prediction
POST /predict_base64       -> JSON API: base64 image, returns prediction
GET  /health               -> Service health check

Run
---
    cd deepfer_project
    python app/flask_app.py

Author : DeepFER Team
Version: 1.0.0
"""

import os
import sys
import io
import base64
import glob
import json
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import predict_emotion, load_model_for_inference, EMOTION_EMOJI

# ── Setup ──────────────────────────────────────────────────────────────────────
ROOT      = os.path.join(os.path.dirname(__file__), '..')
MODEL_DIR = os.path.join(ROOT, 'models')

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024   # 16 MB max upload

# ── Load model on startup ──────────────────────────────────────────────────────
_model = None
_grayscale = True

def get_model():
    global _model
    if _model is None:
        models = sorted(glob.glob(os.path.join(MODEL_DIR, '*.keras')))
        if not models:
            raise FileNotFoundError('No trained model found in models/.  Train first.')
        _model = load_model_for_inference(models[-1])
    return _model


# ── HTML Template (inline for portability) ────────────────────────────────────
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DeepFER - Emotion Recognition</title>
<style>
  body{font-family:'Segoe UI',sans-serif;background:#f0f2f5;margin:0;padding:30px;}
  .container{max-width:700px;margin:auto;background:#fff;border-radius:16px;
             padding:40px;box-shadow:0 4px 24px rgba(0,0,0,.10);}
  h1{color:#2c3e50;font-size:2rem;}
  .subtitle{color:#7f8c8d;margin-top:-12px;}
  label{font-weight:600;color:#34495e;}
  input[type=file]{display:block;margin:12px 0;width:100%;}
  button{background:#3498db;color:#fff;border:none;padding:12px 28px;
         border-radius:8px;font-size:1rem;cursor:pointer;margin-top:8px;}
  button:hover{background:#2980b9;}
  #result{margin-top:28px;display:none;}
  .emotion-box{text-align:center;padding:20px;border-radius:12px;
               background:#f8f9fa;border:2px solid #dee2e6;}
  .emotion-name{font-size:2.5rem;font-weight:700;}
  .conf{font-size:1.2rem;color:#5a6472;}
  .bar-wrap{background:#ecf0f1;border-radius:8px;height:14px;margin:10px 0;}
  .bar-fill{height:100%;border-radius:8px;transition:width .5s;}
  .probs table{width:100%;border-collapse:collapse;font-size:.9rem;}
  .probs td{padding:5px 8px;border-bottom:1px solid #ecf0f1;}
  .probs tr:hover{background:#f0f4ff;}
  #preview{max-width:300px;border-radius:10px;margin-top:12px;display:none;}
</style>
</head>
<body>
<div class="container">
  <h1>DeepFER</h1>
  <p class="subtitle">Facial Emotion Recognition · Deep Learning</p>
  <hr>

  <form id="uploadForm">
    <label for="imgInput">Select a face image (JPG / PNG):</label>
    <input type="file" id="imgInput" name="image" accept="image/*" required>
    <img id="preview" src="" alt="Preview">
    <button type="submit">Detect Emotion</button>
  </form>

  <div id="result">
    <h2>Prediction</h2>
    <div class="emotion-box">
      <div class="emotion-name" id="emotionName"></div>
      <div class="conf" id="confText"></div>
      <div class="bar-wrap"><div class="bar-fill" id="confBar"></div></div>
    </div>
    <h3>All Probabilities</h3>
    <div class="probs"><table id="probTable"></table></div>
  </div>
</div>

<script>
// Preview selected image
document.getElementById('imgInput').addEventListener('change', function(e){
  const prev = document.getElementById('preview');
  prev.src   = URL.createObjectURL(e.target.files[0]);
  prev.style.display = 'block';
});

// Submit form via AJAX
document.getElementById('uploadForm').addEventListener('submit', async function(e){
  e.preventDefault();
  const formData = new FormData();
  formData.append('image', document.getElementById('imgInput').files[0]);

  const res    = await fetch('/predict', {method:'POST', body:formData});
  const data   = await res.json();
  if(data.error){ alert('Error: '+data.error); return; }

  // Fill result card
  document.getElementById('emotionName').textContent = data.emotion.toUpperCase();
  document.getElementById('confText').textContent    = (data.confidence*100).toFixed(1)+'% confidence ('+data.conf_level+')';

  const bar   = document.getElementById('confBar');
  bar.style.width  = (data.confidence*100)+'%';
  const colour     = data.conf_level === 'high'  ? '#27ae60' :
                     data.conf_level === 'medium' ? '#f39c12' : '#e74c3c';
  bar.style.background = colour;

  // Probability table
  const probs  = data.all_probs;
  const sorted = Object.entries(probs).sort((a,b) => b[1]-a[1]);
  let html = '<tr><th>Emotion</th><th>Probability</th><th>Bar</th></tr>';
  sorted.forEach(([cls,p]) => {
    const bold = cls === data.emotion ? 'font-weight:bold;color:#2c3e50' : '';
    html += `<tr style="${bold}">
               <td>${cls}</td>
               <td>${(p*100).toFixed(2)}%</td>
               <td><div style="width:${p*200}px;height:10px;background:${colour};border-radius:5px;"></div></td>
             </tr>`;
  });
  document.getElementById('probTable').innerHTML = html;
  document.getElementById('result').style.display = 'block';
});
</script>
</body>
</html>
"""


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the web UI."""
    return render_template_string(INDEX_HTML)


@app.route('/health')
def health():
    """Health check endpoint for load balancers / monitoring."""
    return jsonify({'status': 'ok', 'model_loaded': _model is not None})


@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    Body : multipart/form-data with 'image' field.
    Returns: JSON prediction dict.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image field in request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        pil_img  = Image.open(io.BytesIO(file.read())).convert('RGB')
        face_arr = np.array(pil_img, dtype=np.uint8)
        model    = get_model()
        result   = predict_emotion(model, face_arr,
                                   class_names=EMOTION_LABELS,
                                   grayscale=_grayscale)
        return jsonify(result), 200

    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """
    POST /predict_base64
    Body : JSON {"image": "<base64-encoded-bytes>"}
    Returns: JSON prediction dict.
    """
    payload = request.get_json(force=True, silent=True)
    if not payload or 'image' not in payload:
        return jsonify({'error': 'JSON body must contain "image" key with base64 data'}), 400

    try:
        img_bytes = base64.b64decode(payload['image'])
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        face_arr  = np.array(pil_img, dtype=np.uint8)
        model     = get_model()
        result    = predict_emotion(model, face_arr,
                                    class_names=EMOTION_LABELS,
                                    grayscale=_grayscale)
        return jsonify(result), 200

    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


# ── Dev server entry point ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n  DeepFER Flask App starting...")
    print("  Open: http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
