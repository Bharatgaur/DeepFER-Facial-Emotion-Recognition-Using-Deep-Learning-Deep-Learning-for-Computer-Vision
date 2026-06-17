"""
DeepFER - app/streamlit_app.py
================================
Streamlit web application for Facial Emotion Recognition.

Features
--------
- Upload an image -> detect and display predicted emotion with confidence bar.
- Webcam snapshot (via st.camera_input) -> live single-frame prediction.
- Batch gallery: upload multiple images and see a summary table.
- Model comparison: load two models and compare their predictions side-by-side.

Run
---
    cd deepfer_project
    streamlit run app/streamlit_app.py

Author : DeepFER Team
Version: 1.0.0
"""

import os
import sys
import glob
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import (load_model_for_inference, predict_emotion,
                     EMOTION_EMOJI, CONF_HIGH, CONF_MEDIUM)

ROOT      = os.path.join(os.path.dirname(__file__), '..')
MODEL_DIR = os.path.join(ROOT, 'models')

EMOTION_LABELS  = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_COLOURS = {
    'angry':    '#e74c3c',
    'disgust':  '#27ae60',
    'fear':     '#8e44ad',
    'happy':    '#f1c40f',
    'neutral':  '#95a5a6',
    'sad':      '#2980b9',
    'surprise': '#e67e22',
}

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='DeepFER - Facial Emotion Recognition',
    layout='wide',
    initial_sidebar_state='expanded',
)


# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title   { font-size: 2.4rem; font-weight: 700; color: #2c3e50; }
    .subtitle     { font-size: 1.1rem; color: #7f8c8d; margin-top: -10px; }
    .metric-card  { background: #f8f9fa; border-radius: 10px; padding: 15px; margin: 5px; }
    .emotion-tag  { font-size: 2rem; font-weight: bold; }
    .conf-bar     { height: 14px; border-radius: 7px; background: #ecf0f1; overflow: hidden; }
    .conf-fill    { height: 100%; border-radius: 7px; }
    .low-conf     { color: #e74c3c; }
    .med-conf     { color: #f39c12; }
    .high-conf    { color: #27ae60; }
    footer        { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image('https://via.placeholder.com/260x80.png?text=DeepFER', use_column_width=True)
    st.title('Settings')

    # Model selector
    model_files = sorted(glob.glob(os.path.join(MODEL_DIR, '*.keras')))
    if not model_files:
        st.warning('No trained model found in models/.  Train first.')
        st.stop()

    model_labels  = [os.path.basename(m) for m in model_files]
    selected_idx  = st.selectbox('Select Model', range(len(model_labels)),
                                 format_func=lambda i: model_labels[i])
    model_path    = model_files[selected_idx]

    grayscale     = st.checkbox('Grayscale input (custom CNN)', value=True)

    st.markdown('---')
    st.info('**Tip:** Custom CNN expects grayscale.  MobileNet / VGG16 expect RGB (uncheck above).')

    st.markdown('### About')
    st.markdown(
        'DeepFER recognises **7 emotions**:\n\n'
        'Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise'
    )


# ─── Model cache ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_model(path: str):
    return load_model_for_inference(path)

model = get_model(model_path)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def pil_to_array(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert('RGB'), dtype=np.uint8)


def make_prob_figure(result: dict) -> plt.Figure:
    """Horizontal bar chart of emotion probabilities."""
    emotions = EMOTION_LABELS
    probs    = [result['all_probs'].get(e, 0.0) for e in emotions]
    colours  = [EMOTION_COLOURS[e] for e in emotions]
    highlight = ['white' if e == result['emotion'] else EMOTION_COLOURS[e] for e in emotions]

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.barh(emotions, probs, color=highlight, edgecolor='black', linewidth=0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    ax.set_title('Emotion Probabilities', fontsize=11)
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def display_result_card(result: dict, col) -> None:
    """Render prediction card in a Streamlit column."""
    with col:
        emotion    = result['emotion']
        confidence = result['confidence']
        conf_level = result['conf_level']

        conf_class = {'high': 'high-conf', 'medium': 'med-conf', 'low': 'low-conf'}[conf_level]
        bar_colour = {'high': '#27ae60', 'medium': '#f39c12', 'low': '#e74c3c'}[conf_level]

        st.markdown(f"""
        <div class="metric-card" style="text-align:center;">
            <div class="emotion-tag">{emotion.upper()}</div>
            <h2 style="color:{EMOTION_COLOURS[emotion]};margin:4px 0;">{emotion.capitalize()}</h2>
            <p class="{conf_class}" style="font-size:1.3rem;font-weight:bold;">
                {confidence*100:.1f}% confidence
            </p>
            <div class="conf-bar">
                <div class="conf-fill"
                     style="width:{confidence*100:.1f}%;background:{bar_colour};"></div>
            </div>
            <small style="color:#95a5a6">Confidence level: <b>{conf_level.upper()}</b></small>
        </div>
        """, unsafe_allow_html=True)


# ─── Main UI ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">DeepFER - Facial Emotion Recognition</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deep Learning · CNN · Transfer Learning · Real-Time</div>',
            unsafe_allow_html=True)
st.markdown('---')

tab1, tab2, tab3 = st.tabs(['Upload Image', 'Webcam', 'Batch Gallery'])


# ─── Tab 1: Single image upload ────────────────────────────────────────────────
with tab1:
    st.subheader('Upload a face image')
    uploaded = st.file_uploader(
        'Choose an image file (JPG / PNG)',
        type=['jpg', 'jpeg', 'png'],
        label_visibility='collapsed',
    )

    if uploaded is not None:
        pil_img = Image.open(uploaded)

        c1, c2, c3 = st.columns([1, 1, 1])

        with c1:
            st.image(pil_img, caption='Input Image', use_column_width=True)

        with st.spinner('Analysing emotion …'):
            face_arr = pil_to_array(pil_img)
            result   = predict_emotion(model, face_arr,
                                       class_names=EMOTION_LABELS,
                                       grayscale=grayscale)

        display_result_card(result, c2)

        with c3:
            fig = make_prob_figure(result)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # Expandable detail
        with st.expander('Raw probabilities'):
            for cls in EMOTION_LABELS:
                p = result['all_probs'][cls]
                st.progress(p, text=f"{cls}: {p*100:.2f}%")
    else:
        st.info('Upload a face image above to get started.')


# ─── Tab 2: Webcam snapshot ────────────────────────────────────────────────────
with tab2:
    st.subheader('Webcam - Capture and Predict')
    st.info('Click the camera button below to take a photo, then see the prediction instantly.')

    img_data = st.camera_input('Take a photo')

    if img_data is not None:
        pil_img  = Image.open(img_data)
        c1, c2   = st.columns(2)
        c1.image(pil_img, caption='Captured Frame', use_column_width=True)

        with st.spinner('Analysing …'):
            face_arr = pil_to_array(pil_img)
            result   = predict_emotion(model, face_arr,
                                       class_names=EMOTION_LABELS,
                                       grayscale=grayscale)

        display_result_card(result, c2)

        fig = make_prob_figure(result)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ─── Tab 3: Batch gallery ──────────────────────────────────────────────────────
with tab3:
    st.subheader('Batch Emotion Analysis')
    uploads = st.file_uploader(
        'Upload multiple face images',
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        label_visibility='collapsed',
    )

    if uploads:
        results_data = []
        cols_per_row = 4
        rows         = [uploads[i:i+cols_per_row]
                        for i in range(0, len(uploads), cols_per_row)]

        with st.spinner(f'Analysing {len(uploads)} images …'):
            for row_files in rows:
                cols = st.columns(len(row_files))
                for col, uf in zip(cols, row_files):
                    pil_img  = Image.open(uf)
                    face_arr = pil_to_array(pil_img)
                    result   = predict_emotion(model, face_arr,
                                              class_names=EMOTION_LABELS,
                                              grayscale=grayscale)
                    with col:
                        st.image(pil_img, use_column_width=True)
                        st.markdown(
                            f"**{result['emotion'].capitalize()}**  \n"
                            f"{result['confidence']*100:.1f}%"
                        )
                    results_data.append({
                        'File':       uf.name,
                        'Emotion':    result['emotion'],
                        'Confidence': f"{result['confidence']*100:.1f}%",
                        'Level':      result['conf_level'],
                    })

        st.markdown('---')
        st.subheader('Summary Table')
        import pandas as pd
        st.dataframe(pd.DataFrame(results_data), use_container_width=True)

        # Emotion distribution pie chart
        from collections import Counter
        emo_counts = Counter(r['Emotion'] for r in results_data)
        fig2, ax2  = plt.subplots(figsize=(5, 4))
        ax2.pie(emo_counts.values(), labels=emo_counts.keys(),
                autopct='%1.1f%%', startangle=90,
                colors=[EMOTION_COLOURS[e] for e in emo_counts])
        ax2.set_title('Emotion Distribution in Batch')
        st.pyplot(fig2, use_container_width=False)
        plt.close(fig2)

    else:
        st.info('Upload multiple images to run batch analysis.')
