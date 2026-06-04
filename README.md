# SignBridge — Vietnamese Sign Language Recognition

Real-time Vietnamese sign language recognition using a **Transformer + Causal DWConv1D** architecture (Top 1 Kaggle ISLR 2023), adapted for local training and low-latency webcam inference.

---

## Architecture Highlights
- **Hybrid Landmark Extraction:** Combines `HolisticLandmarker` (for face/pose stability) and `HandLandmarker` (for high-fidelity hand joints) into a unified 543-point feature vector.
- **EMA Temporal Tracking:** Prevents jitter and hand-swapping during occlusions using an Exponential Moving Average tracker and plausibility boundary checks.
- **TFLite Deployment:** Employs multi-threaded CPU inference (`concurrent.futures.ThreadPoolExecutor`), running MediaPipe extraction and TFLite model inference in parallel to bypass GIL bottlenecks and achieve maximum FPS.

---

## Hardware Support

| Environment | Compute |
|---|---|
| Local machine (Windows/Linux/macOS) | **CPU only** (Highly Optimized) |
| Google Colab | **TPU** (For Training) |

> GPU is **not required**. The TFLite inference pipeline runs exceptionally well on standard laptop CPUs.

---

## Project Structure

```
├── extract_frames.py       # Step 1: Extract frames from videos
├── extract_mediapipe.py    # Step 2: Extract 543 body landmarks via Hybrid MediaPipe
├── prepare_tfrecords.py    # Step 3: Pack .npy matrices into TFRecord files
├── train.py                # Step 4: Train the model
├── export_tflite.py        # Step 5: Convert trained weights to TFLite
├── camera_demo.py          # Step 6: Real-time inference (Standard Keras - Debug)
├── camera_demo_tflite.py   # Step 6: Real-time inference (TFLite Engine - Production)
│
├── holistic_landmarker.task   # Auto-downloaded MediaPipe model
├── hand_landmarker.task       # Auto-downloaded MediaPipe model
├── requirements.txt
│
├── dataset/Vietnamese/        # Raw videos + label CSV
├── data/Vietnamese/
│   ├── frames/                # Output of Step 1
│   ├── mediapipe/             # Output of Step 2 (.npy matrices)
│   └── TFRecord/              # Output of Step 3 (.tfrecords + label_map.json)
└── output/                    # Training output (weights + logs + tflite)
```

---

## Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\activate

# Activate (Linux / macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Full Pipeline Guide

### Step 1 — Extract frames
```bash
python extract_frames.py
```
Reads videos from `dataset/Vietnamese/` → saves frames to `data/Vietnamese/frames/`.

---

### Step 2 — Extract MediaPipe landmarks
```bash
python extract_mediapipe.py
```
Reads videos → extracts 543 body keypoints (face, hands, pose) per frame using the **Hybrid Tracker** → saves as `.npy` to `data/Vietnamese/mediapipe/`.
> ⚠️ **Note:** Downloads `holistic_landmarker.task` and `hand_landmarker.task` automatically on first run.

---

### Step 3 — Build TFRecord dataset
```bash
python prepare_tfrecords.py
```
Reads `.npy` files + label CSV → packages into 5-fold `.tfrecords` files in `data/Vietnamese/TFRecord/`. Generates `label_map.json` mapping string labels to class IDs.

---

### Step 4 — Train the model
```bash
python train.py
```
Trains a Transformer model using 5-Fold Cross-Validation.
Results are saved to `output/[timestamp]/`:
- `*.weights.h5` — best model weights per fold
- `*-logs.csv` — loss/accuracy logs

---

### Step 5 — Export to TFLite (Optional/Manual)
*Note: `train.py` might auto-export at the end. If you need to re-export manually:*
```bash
python export_tflite.py
```
Compresses the standard Keras weights into an optimized `float16` `.tflite` model, ready for real-time deployment.

---

### Step 6 — Run webcam demo (Production)

**We highly recommend using the TFLite version** for significantly better FPS and lower resource usage. Both versions feature identical EMA Temporal Tracking.

```bash
# Fast inference using the compressed TFLite Engine (Recommended)
python camera_demo_tflite.py
```

On startup, an interactive menu will appear asking you to select the trained model directory from `output/`.

**Controls during demo:**
| Key | Action |
|---|---|
| `R` | Reset temporal buffer & EMA trackers |
| `S` | Save screenshot to `captures/` |
| `Q` / `ESC` | Quit |

**Optional flags:**
```bash
python camera_demo_tflite.py --camera 1        # Use a different camera index
python camera_demo_tflite.py --topk 5          # Show top 5 predictions
```

---

## Model Architecture

```
Input: (N, 384, 708)
  ↓ Preprocess (normalize by nose point #17, compute velocity dx and dx2)
  ↓ Dense(192) stem
  ↓ [Conv1DBlock × 3 → TransformerBlock] × 2
       Conv1DBlock: Expand → CausalDWConv1D(kernel=17) → ECA → Project
       TransformerBlock: Multi-Head Self-Attention (4 heads) + FFN
  ↓ Dense(384) → GlobalAveragePooling
  ↓ LateDropout(0.8)
  ↓ Dense(4) classifier
Output: class probabilities
```

| Property | Value |
|---|---|
| Total parameters | ~1.74 million |
| Compressed Size | ~6.6 MB (.tflite float16) |
| Input features | 118 landmarks × 6 channels (x, y, dx, dy, dx2, dy2) = 708 |

---

## References

- [ISLR Top 1 Solution — Hoyeol Sohn (Kaggle)](https://www.kaggle.com/competitions/asl-signs/discussion/406684)
- [MediaPipe Tasks API](https://developers.google.com/mediapipe/solutions/vision/holistic_landmarker)
- [TensorFlow TFLite](https://www.tensorflow.org/lite)
