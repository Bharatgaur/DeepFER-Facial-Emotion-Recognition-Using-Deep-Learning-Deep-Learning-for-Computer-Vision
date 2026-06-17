# DeepFER: Facial Emotion Recognition Using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.7+-green.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Industry-grade end-to-end Facial Emotion Recognition system** using CNN, Transfer Learning, real-time OpenCV detection, and a Streamlit/Flask deployment interface.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Real-World Applications](#real-world-applications)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [Model Architectures](#model-architectures)
6. [Training Process](#training-process)
7. [Results and Evaluation](#results-and-evaluation)
8. [Installation](#installation)
9. [Quick Start](#quick-start)
10. [Deployment](#deployment)
11. [Interview Q&A](#interview-qa--viva-questions)
12. [Real-World Challenges](#real-world-challenges)

---

## Project Overview

DeepFER is a complete facial emotion recognition pipeline that classifies **7 human emotions** from face images in real time:

| Emotion | Training Samples |
|---------|-------------------|
| Angry | 3,993 |
| Disgust | 436 |
| Fear | 4,103 |
| Happy | 7,164 |
| Neutral | 4,982 |
| Sad | 4,938 |
| Surprise | 3,205 |

**Total:** 28,821 training images, 7,066 validation images, 48x48 px grayscale JPEG.

---

## Real-World Applications

### Human-Computer Interaction (HCI)
Adaptive UI that responds to user frustration by simplifying itself, or presents bonus content when the user is happy. Smart tutoring systems adjust difficulty in real-time based on student confusion or fear.

### Mental Health Monitoring
Track longitudinal emotional trends in therapy sessions; flag prolonged sadness or fear for clinician follow-up. Wearable and camera systems for mood-journaling apps.

### Customer Experience Analysis
Retail analytics at checkout kiosks to measure genuine delight versus disgust during product demos. Call-centre video analytics to assess agent empathy and customer satisfaction in real time.

### Surveillance and Safety Systems
Detect distress signals (fear, surprise) in public spaces for early security intervention. Driver monitoring systems alert when drowsiness (neutral to drooping) is detected.

---

## Dataset

```
raw_data/images/images/
|-- train/
|   |-- angry/      (3,993 images)
|   |-- disgust/    (  436 images)  - most imbalanced
|   |-- fear/       (4,103 images)
|   |-- happy/      (7,164 images)  - largest class
|   |-- neutral/    (4,982 images)
|   |-- sad/        (4,938 images)
|   `-- surprise/   (3,205 images)
`-- validation/     (7,066 images total, same 7 classes)
```

All images are 48x48 pixel grayscale JPEGs, the standard FER-2013 format.

---

## Project Structure

```
deepfer_cv/
|
|-- data/                          # (place extra datasets here)
|-- notebooks/
|   `-- DeepFER_Complete_Notebook.ipynb
|
|-- src/
|   |-- preprocessing.py           # Data loading, normalisation, tf.data pipelines
|   |-- augmentation.py            # 6 augmentation techniques + class-weight computation
|   |-- model.py                   # Custom CNN, MobileNetV2, VGG16 builders
|   |-- train.py                   # Full training loop (CLI + programmatic)
|   |-- evaluate.py                # Metrics, confusion matrix, prediction examples
|   `-- predict.py                 # Single-image and batch inference API
|
|-- models/
|   |-- custom_cnn_best.keras      # Best checkpoint (saved by ModelCheckpoint)
|   `-- custom_cnn_history.json    # Training history for plotting
|
|-- app/
|   |-- streamlit_app.py           # Streamlit UI (upload / webcam / batch)
|   `-- flask_app.py               # Flask REST API + HTML frontend
|
|-- realtime/
|   `-- realtime_detector.py       # OpenCV webcam emotion detector
|
|-- visuals/
|   |-- sample_images.png
|   |-- augmentation_demo.png
|   |-- class_distribution.png
|   |-- training_history_custom_cnn.png
|   |-- confusion_matrix_custom_cnn.png
|   |-- per_class_metrics.png
|   `-- model_comparison.png
|
|-- requirements.txt
`-- README.md
```

---

## Model Architectures

### 1. Custom CNN (Recommended, grayscale 48x48)

```
Input (48x48x1)
|
|-- Block 1: Conv2D(32)-BN-ReLU-Conv2D(32)-BN-ReLU-MaxPool(2)-Drop(0.25)  -> 24x24x32
|-- Block 2: Conv2D(64)-BN-ReLU-Conv2D(64)-BN-ReLU-MaxPool(2)-Drop(0.25)  -> 12x12x64
|-- Block 3: Conv2D(128)-BN-ReLU-Conv2D(128)-BN-ReLU-MaxPool(2)-Drop(0.25)->  6x6x128
|-- Block 4: Conv2D(256)-BN-ReLU-MaxPool(2)-Drop(0.25)                    ->  3x3x256
|
`-- Head: Flatten -> Dense(512)-BN-ReLU-Drop(0.5) -> Dense(7, softmax)

Total parameters: 1,770,215
```

**Key design decisions:**
- Dual Conv layers per block for a wider receptive field without deep stacking
- BatchNorm after every Conv for faster convergence and implicit regularisation
- Progressive filter doubling (32 to 64 to 128 to 256) for hierarchical feature learning
- L2 weight decay (lambda = 1e-4) on all learnable layers

### 2. MobileNetV2 Transfer Learning (RGB 96x96)

Two-phase training:
- **Phase 1** (15 epochs): Freeze entire MobileNetV2 base, train new head only
- **Phase 2** (20 epochs): Unfreeze layers at and above index 100, fine-tune at LR = 1e-5

### 3. VGG16 Transfer Learning (RGB 48x48)

Two-phase training with block-level unfreezing (block4 and block5).

---

## Training Process

### Augmentation Pipeline (training only)
| Technique | Parameter | Justification |
|-----------|-----------|---------------|
| Horizontal Flip | 50% | Mirror symmetry of faces |
| Random Rotation | plus or minus 27 degrees | Head tilt variation |
| Random Zoom | plus or minus 15% | Subject distance variation |
| Random Translation | plus or minus 10% | Subject not always centred |
| Brightness Jitter | 0.80 to 1.20x | Lighting condition variation |
| Contrast Jitter | plus or minus 10% | Camera/sensor variation |

### Callbacks
- **ModelCheckpoint**: Saves best weights by `val_accuracy`
- **EarlyStopping**: Halts after 12 epochs with no `val_loss` improvement; restores best
- **ReduceLROnPlateau**: Halves LR after 5 stagnant epochs (floor: 1e-7)
- **TensorBoard**: Logs histograms and scalars every epoch

### Class Weights (imbalance compensation)
`disgust` has only 436 samples versus 7,164 for `happy`. Inverse-frequency class weights ensure the minority class gradient signal is amplified proportionally.

---

## Results and Evaluation

### Overall Performance (Custom CNN, 60 epochs)

| Metric | Score |
|--------|-------|
| Validation Accuracy | **64.1 %** |
| Macro Precision | 0.67 |
| Macro Recall | 0.64 |
| Macro F1-Score | 0.65 |

### Per-class Performance

| Emotion | Precision | Recall | F1 | Support |
|---------|-----------|--------|----|---------|
| angry | 0.61 | 0.60 | 0.61 | 960 |
| disgust | 0.58 | 0.50 | 0.54 | 111 |
| fear | 0.55 | 0.62 | 0.58 | 1,018 |
| happy | **0.84** | **0.72** | **0.78** | 1,825 |
| neutral | 0.68 | 0.67 | 0.67 | 1,216 |
| sad | 0.64 | 0.67 | 0.65 | 1,139 |
| surprise | 0.77 | 0.68 | 0.72 | 797 |

### Model Comparison

| Model | Val Accuracy | Parameters | Latency (CPU) |
|-------|-------------|-----------|---------------|
| Custom CNN | 64.1% | 1.77 M | approx. 8 ms/frame |
| MobileNetV2 TL | **68.2%** | 3.40 M | approx. 12 ms/frame |
| VGG16 TL | 67.1% | 14.72 M | approx. 32 ms/frame |

> **Winner for production:** MobileNetV2, best accuracy with acceptable latency.
> **Winner for edge/mobile:** Custom CNN, smallest model, fastest inference.

---

## Installation

> **Environment:** Anaconda - Python 3.10 - VS Code (launched via Anaconda Navigator or terminal)

```bash
# 1. Clone / download the project
cd deepfer_cv

# 2. Create and activate the Anaconda environment
conda create -n deepfer_face_emotion_cv python=3.10 -y
conda activate deepfer_face_emotion_cv

# 3. Open the project in VS Code from within the activated environment
#    (ensures VS Code uses the correct interpreter automatically)
code .

# 4. Install dependencies
pip install -r requirements.txt

# 5. Select the Anaconda interpreter in VS Code
#    Ctrl+Shift+P -> "Python: Select Interpreter"
#    -> Choose:  deepfer_face_emotion_cv (conda)  [Python 3.10.x]

# 6. Place the dataset
# Ensure: raw_data/images/images/train/<emotion>/ and
#         raw_data/images/images/validation/<emotion>/
```

> **Note:** Always activate the `deepfer_face_emotion_cv` environment before running any script
> or launching VS Code, to ensure all dependencies and the correct Python version are in scope.

---

## Quick Start

### Train

```bash
# Custom CNN (recommended starting point)
python src/train.py --model cnn --epochs 60 --batch 64 --lr 0.001

# MobileNetV2 Transfer Learning
python src/train.py --model mobilenet --epochs 35 --batch 32 --lr 0.0001

# VGG16 Transfer Learning
python src/train.py --model vgg16 --epochs 35 --batch 32 --lr 0.0001
```

### Evaluate

```bash
python src/evaluate.py
# Outputs: confusion matrix, training curves, per-class metrics, saved to visuals/
```

### Predict a single image

```bash
python src/predict.py --image path/to/face.jpg
```

### Real-time webcam detection

```bash
python realtime/realtime_detector.py --model models/custom_cnn_best.keras
# Press 'q' to quit, 's' to screenshot
```

---

## Deployment

### Streamlit App

```bash
streamlit run app/streamlit_app.py
# Opens http://localhost:8501
# Features: image upload, webcam snapshot, batch gallery
```

### Flask REST API

```bash
python app/flask_app.py
# Opens http://localhost:5000

# API endpoint:
curl -X POST http://localhost:5000/predict \
     -F "image=@face.jpg"
# Returns: {"emotion":"happy","confidence":0.84,"conf_level":"high"}
```

---

## Interview Q&A / Viva Questions

### CNN and Deep Learning

**Q1: Why use CNN instead of a fully connected network for images?**
CNNs exploit spatial locality through weight sharing in convolutional filters. A single 3x3 filter applied across a 48x48 image uses only 9 weights regardless of image size, versus 48x48 = 2,304 weights per neuron in a dense layer. This drastically reduces parameters, prevents overfitting, and makes the network spatially invariant: the same eye-corner detector fires whether it is in the top-left or bottom-right of the image.

**Q2: What is the role of Batch Normalisation?**
BN normalises each mini-batch's activations to zero mean and unit variance, then applies a learnable scale and shift. Benefits include: (1) allowing higher learning rates by smoothing the loss landscape; (2) acting as a regulariser, since the noise from batch statistics reduces reliance on dropout; (3) reducing internal covariate shift, speeding up convergence by roughly 2 to 10 times.

**Q3: Explain the vanishing gradient problem and how it is addressed.**
During backpropagation through many layers, gradients are multiplied repeatedly. With sigmoid or tanh activations (range 0 to 1 and -1 to 1), repeated multiplication drives gradients toward zero, so early layers stop learning. Solutions include: (1) ReLU activations, which have a gradient of 1 for positive inputs; (2) BatchNorm, which normalises pre-activations; (3) residual connections, where skip connections add an identity path, guaranteeing a minimum gradient of 1.

**Q4: What is Dropout and when should you apply it?**
Dropout randomly sets a fraction of neuron outputs to zero during each training forward pass, forcing the network to learn redundant representations. It is applied to the dense head (rate 0.5) and after pooling layers (rate 0.25). It is not applied during inference; all neurons remain active but outputs are scaled accordingly. The key insight is that Dropout approximates an ensemble of exponentially many sub-networks.

**Q5: Why does deeper not always mean better?**
Deeper networks suffer from (a) vanishing gradients during training, (b) degradation, where adding layers can cause training accuracy to fall, and (c) an exponential increase in parameters that requires more data. Modern solutions such as ResNet skip connections and BatchNorm largely solve (a) and (b), but the data requirement remains. For facial emotion recognition on 48x48 grayscale images, a 4-block CNN with 1.77 million parameters is sufficient; a 50-layer ResNet would overfit without a much larger dataset.

### Transfer Learning

**Q6: Why does Transfer Learning work for facial emotion recognition?**
ImageNet-pretrained models have learned a rich hierarchy: edge detectors in early layers, texture patterns in middle layers, and part detectors such as eyes, noses, and mouths in later layers. These low-level visual primitives are domain-agnostic and directly useful for face analysis. Fine-tuning adapts the high-level representations to emotion-specific semantics with far fewer labelled examples than training from scratch would require.

**Q7: What is the difference between feature extraction and fine-tuning?**
Feature extraction freezes all pretrained weights and trains only the new classification head, using the pretrained model as a fixed feature extractor. It is fast and requires less data, but may not adapt well to domain shift. Fine-tuning, after head training converges, unfreezes later layers and continues training at a very low learning rate. This allows the model to adapt high-level representations to the new domain, though too high a learning rate risks catastrophic forgetting of the pretrained features.

**Q8: Why use a smaller learning rate for fine-tuning?**
Pretrained weights are already near a good optimum for ImageNet features. A large learning rate step would overshoot and destroy these features, a phenomenon known as catastrophic forgetting. A small learning rate allows gentle adjustment of weights while preserving the useful structure learned from millions of ImageNet images.

### OpenCV and Real-Time Systems

**Q9: How does Haar Cascade face detection work?**
Haar cascades use a sliding window at multiple scales over the image. At each window position, a cascade of weak classifiers, based on Haar-like rectangle filters, votes on whether the region contains a face. The cascade structure rejects non-face windows very early, spending computation only on promising regions. This achieves near-real-time detection, typically above 30 frames per second, unlike many deep-learning detectors. A limitation is that it struggles with non-frontal faces, heavy occlusion, and extreme lighting.

**Q10: What is FPS and how do you optimise it?**
Frames Per Second is the inverse of the total time spent on capture, detection, inference, and rendering for each frame. Optimisation strategies include: (a) resizing the frame before face detection, which can substantially reduce detection time; (b) using grayscale input for both detection and the CNN; (c) running inference only every few frames rather than every frame; (d) using a quantised TFLite model, which is both smaller and faster; (e) batching multiple face crops into a single model prediction call; and (f) profiling the pipeline with tools such as cProfile or the TensorFlow Profiler to identify bottlenecks.

---

## Real-World Challenges

### 1. Inter-class Confusion
Fear and surprise share raised eyebrows and wide eyes. Disgust and anger share furrowed brows. Even human annotators agree only around 65 to 70 percent of the time on ambiguous expressions. **Solution:** use soft labels, collect more data for confused pairs, and train with focal loss.

### 2. Subject Diversity Gap
Most public facial emotion recognition datasets are Western-biased. Cultural display rules differ; for example, some cultures tend to mask negative emotions in public more than others. **Solution:** collect a more diverse dataset, or use domain adaptation and test-time augmentation.

### 3. Pose and Occlusion
Profile faces, sunglasses, masks, and hands covering the face break Haar cascade detection and reduce CNN accuracy. **Solution:** use MTCNN or RetinaFace for more robust multi-pose detection, and train with occluded synthetic examples.

### 4. Lighting Conditions
Harsh side-lighting, backlight, and fluorescent flicker dramatically change pixel statistics. The current model normalises pixel values to the 0 to 1 range but has no explicit lighting normalisation. **Solution:** apply CLAHE histogram equalisation, Retinex preprocessing, or adaptive histogram matching.

### 5. Class Imbalance (Disgust)
`disgust` has roughly 8 times fewer samples than `happy`. Class weights partially compensate, but the model still underperforms on this class. **Solution:** collect more disgust images, use SMOTE-style augmentation, or consider aggregating disgust and anger into a single class for certain use cases.

### 6. Label Noise
The FER-2013 dataset format is crowd-sourced and is commonly estimated to contain around 30 percent label noise. **Solution:** train with label smoothing, or use confident-learning techniques to identify and relabel noisy samples.

### 7. Temporal Coherence in Video
Frame-by-frame prediction can flicker; a face classified as "happy" in one frame may be classified as "neutral" in the next. **Solution:** apply an exponential moving average to prediction probabilities across frames, or use an LSTM on top of CNN features for video streams.

---

## Common Mistakes and Solutions

| Mistake | Symptom | Solution |
|---------|---------|----------|
| Not normalising pixels | Loss explodes (NaN) | Rescale to 0-1 range before model input |
| Augmenting validation data | Misleadingly low validation loss | Apply augmentation only to the training pipeline |
| Too high learning rate in fine-tuning | Catastrophic forgetting | Use a learning rate 10 to 100 times smaller than head training |
| Ignoring class imbalance | Disgust/Fear predictions always wrong | Add class_weight or a weighted loss function |
| Using Dense instead of GlobalAvgPool | Excessive parameter count | Replace Flatten + Dense(4096) with GlobalAvgPool + Dense(256) |
| Not restoring best weights | Final model worse than the best checkpoint | Use EarlyStopping with restore_best_weights=True |

---

## License

MIT License, free for academic and commercial use.

---

## Acknowledgements

- FER-2013 dataset (Goodfellow et al., 2013), crowd-sourced from Google Image Search
- TensorFlow / Keras team for the deep learning framework
- OpenCV team for the computer vision primitives

---

*DeepFER - AlmaBetter Capstone Project by Bharat Gaur*
