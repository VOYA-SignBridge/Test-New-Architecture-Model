# SignBridge — Vietnamese Sign Language Recognition

Real-time Vietnamese sign language recognition using a **Transformer + Causal DWConv1D** architecture (Top 1 Kaggle ISLR 2023), adapted for local training and webcam inference.

---

## Hardware Support

| Environment | Compute |
|---|---|
| Local machine (Windows/Linux/macOS) | **CPU only** |
| Google Colab | **TPU** |

> GPU is **not supported**.

---

## Project Files

```
├── extract_frames.py       # Step 1: Extract frames from videos
├── extract_mediapipe.py    # Step 2: Extract 543 body landmarks via MediaPipe
├── prepare_tfrecords.py    # Step 3: Pack .npy matrices into TFRecord files
├── train.py                # Step 4: Train the model & Auto-export to .tflite
├── camera_demo.py          # Step 5: Real-time inference (Standard Keras)
├── camera_demo_tflite.py   # Step 5: Real-time inference (TFLite Engine)
│
├── holistic_landmarker.task   ⚠️ Required — see note below
├── requirements.txt
│
├── dataset/Vietnamese/        # Raw videos + label CSV
├── data/Vietnamese/
│   ├── frames/                # Output of Step 1
│   ├── mediapipe/             # Output of Step 2 (.npy matrices)
│   └── TFRecord/              # Output of Step 3 (.tfrecords + label_map.json)
└── output/                    # Training output (weights + logs), created automatically
    └── [H-M-D-M-YYYY]/
```

> ⚠️ **`holistic_landmarker.task`** (~13 MB) is required by `extract_mediapipe.py`.
> It downloads automatically on first run.
> If download fails, get it manually from:
> https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task
> and place it in the project root.

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

## Training Pipeline

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
Reads videos → extracts 543 body keypoints (face, hands, pose) per frame → saves as `.npy` to `data/Vietnamese/mediapipe/`.

> Downloads `holistic_landmarker.task` automatically on first run if missing.

---

### Step 3 — Build TFRecord dataset

```bash
python prepare_tfrecords.py
```
Reads `.npy` files + label CSV → packages into 5-fold `.tfrecords` files in `data/Vietnamese/TFRecord/`.
Also generates `label_map.json` (word name → class index mapping).

---

### Step 4 — Train the model

```bash
python train.py
```
Trains a Transformer model using 5-Fold Cross-Validation.
Results are saved to `output/[timestamp]/`:
- `*.weights.h5` — best model weights per fold
- `*-logs.csv` — loss/accuracy log per epoch
- `*-best-float16.tflite` — compressed, deployment-ready TFLite model (auto-generated)

Key settings inside `train.py` (class `CFG`):

| Parameter | Default | Description |
|---|---|---|
| `epoch` | `300` | Number of training epochs |
| `dim` | `192` | Model hidden dimension |
| `MAX_LEN` | `384` | Max sequence length (frames) |
| `batch_size` | `64 × replicas` | Batch size |

---

### Step 5 — Run webcam demo

There are two versions of the webcam demo available. **We highly recommend using the TFLite version** for significantly better performance and lower resource usage.

```bash
# Option A: Fast inference using the compressed TFLite Engine (Recommended)
python camera_demo_tflite.py

# Option B: Standard Keras inference (Builds architecture dynamically)
python camera_demo.py
```

On startup, an interactive menu will appear asking you to select the trained model directory.

**Controls during demo:**

| Key | Action |
|---|---|
| `R` | Reset buffer |
| `S` | Save screenshot to `captures/` |
| `Q` / `ESC` | Quit |

**Reading the display:**
- Progress bar (left panel): frame collection progress (0% → 100% → predict)
- Bottom-right corner: predicted word + confidence percentage
- Color: Green (>70%), Orange-yellow (>50%), Orange (<50%)
- Top-K panel: top 3 candidate predictions

**Optional flags:**
```bash
python camera_demo_tflite.py --camera 1        # Use a different camera index
python camera_demo_tflite.py --topk 5          # Show top 5 predictions
python camera_demo_tflite.py --model path/to/model.tflite   # Load specific model file
```

---

## Model Architecture

```
Input: (N, 384, 708)
  ↓ Preprocess (normalize by nose point #17, compute dx and dx2)
  ↓ Dense(192) stem
  ↓ [Conv1DBlock × 3 → TransformerBlock] × 2
       Conv1DBlock: Expand → CausalDWConv1D(kernel=17) → ECA → Project
       TransformerBlock: Multi-Head Self-Attention (4 heads) + FFN
  ↓ Dense(384) → GlobalAveragePooling
  ↓ LateDropout(0.8)
  ↓ Dense(4) classifier
Output: class probabilities over 4 sign labels
```

| Property | Value |
|---|---|
| Total parameters | ~1.74 million |
| Model size | ~6.6 MB |
| Input features | 118 landmarks × 6 channels (x, y, dx, dy, dx2, dy2) = 708 |
| Output classes | 4 (number of sign words) |

---

## References

- [ISLR Top 1 Solution — Hoyeol Sohn (Kaggle)](https://www.kaggle.com/competitions/asl-signs/discussion/406684)
- [MediaPipe Holistic Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker)
- [TensorFlow TFRecord Guide](https://www.tensorflow.org/tutorials/load_data/tfrecord)
