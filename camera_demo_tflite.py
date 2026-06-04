"""
camera_demo_tflite.py
=====================
Nhận diện ký hiệu tay realtime từ camera bằng mô hình TFLite siêu nén.

Điểm khác biệt so với camera_demo.py:
  - Khởi động cực kỳ nhanh.
  - Sử dụng file .tflite thay vì .weights.h5.
  - Nhẹ hơn, tốn ít RAM và CPU hơn.
"""

import os
import sys
import time
import argparse
import glob
import numpy as np

try:
    import cv2
except ImportError:
    print("[ERROR] Thiếu opencv-python. Chạy: pip install opencv-python")
    sys.exit(1)

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    print("[ERROR] Thiếu mediapipe. Chạy: pip install mediapipe")
    sys.exit(1)

import urllib.request

try:
    import tensorflow as tf
except ImportError:
    print("[ERROR] Thiếu tensorflow. Chạy: pip install tensorflow")
    sys.exit(1)


# ─── CONSTANTS ───────────────────────────────────────────────────────────────
MAX_LEN         = 384
BUFFER_LEN      = 60
NUM_CLASSES     = 4
PAD             = -100.

# 118 Landmark indices (khớp với quá trình train)
NOSE   = [1, 2, 98, 327]
LIP = [
    0, 61, 185, 40, 39, 37, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
LHAND = list(range(468, 489))
RHAND = list(range(522, 543))
REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133,
        246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362,
        466, 388, 387, 386, 385, 384, 398]

POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE
NUM_NODES       = len(POINT_LANDMARKS)
CHANNELS        = 6 * NUM_NODES

DEFAULT_MODEL_DIR = "./output"

HOLISTIC_MODEL_PATH = "holistic_landmarker.task"
HOLISTIC_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"

HANDS_MODEL_PATH = "hand_landmarker.task"
HANDS_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

FACE_START,  FACE_END   = 0,   468
LHAND_START, LHAND_END  = 468, 489
POSE_START,  POSE_END   = 489, 522
RHAND_START, RHAND_END  = 522, 543

def _ensure_mp_model():
    """Tải model MediaPipe nếu chưa có."""
    downloaded = False
    if not os.path.exists(HOLISTIC_MODEL_PATH):
        print(f"[INFO] Đang tải model Holistic ({HOLISTIC_MODEL_PATH})...")
        urllib.request.urlretrieve(HOLISTIC_URL, HOLISTIC_MODEL_PATH)
        downloaded = True
    if not os.path.exists(HANDS_MODEL_PATH):
        print(f"[INFO] Đang tải model Hand ({HANDS_MODEL_PATH})...")
        urllib.request.urlretrieve(HANDS_URL, HANDS_MODEL_PATH)
        downloaded = True
    if downloaded:
        print("[INFO] Đã tải đủ các mô hình MediaPipe!")


# ─── LOAD LABEL MAP ───────────────────────────────────────────────────────────
def load_label_map():
    import json
    json_path = os.path.join("data", "Vietnamese", "TFRecord", "label_map.json")
    label_map      = {}
    label_original = {}

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for k, v in data.items():
                idx = int(v)
                label_map[idx] = k
                label_original[idx] = k
    else:
        print(f"[WARN] label_map.json không tìm thấy: {json_path}")
        label_map      = {i: f"class_{i}" for i in range(NUM_CLASSES)}
        label_original = label_map.copy()

    return label_map, label_original


# ─── MEDIAPIPE SETUP (Tasks API) ──────────────────────────────────────────────
def init_mediapipe():
    """Khởi tạo HolisticLandmarker và HandLandmarker dùng MediaPipe Tasks API."""
    _ensure_mp_model()
    
    # 1. Holistic
    holistic_base = mp_python.BaseOptions(model_asset_path=HOLISTIC_MODEL_PATH)
    holistic_options = mp_vision.HolisticLandmarkerOptions(
        base_options=holistic_base,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
        output_face_blendshapes=False,
    )
    
    # 2. Hands
    hands_base = mp_python.BaseOptions(model_asset_path=HANDS_MODEL_PATH)
    hands_options = mp_vision.HandLandmarkerOptions(
        base_options=hands_base,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    return (mp_vision.HolisticLandmarker.create_from_options(holistic_options),
            mp_vision.HandLandmarker.create_from_options(hands_options))

class HandTracker:
    class _P:
        __slots__ = ('x', 'y')
        def __init__(self, x: float, y: float): self.x = x; self.y = y
    def __init__(self, ema_alpha: float = 0.45, miss_ttl: int = 5, max_jump: float = 0.28):
        self.alpha = ema_alpha
        self.ttl = miss_ttl
        self.max_jump = max_jump
        self._ema_l = None
        self._ema_r = None
        self._miss_l = 0
        self._miss_r = 0
    def anchors(self):
        P = self._P
        return (P(*self._ema_l) if self._ema_l else None, P(*self._ema_r) if self._ema_r else None)
    def is_plausible(self, x: float, y: float, slot: int) -> bool:
        ema = self._ema_l if slot == LHAND_START else self._ema_r
        if ema is None: return True
        return ((x - ema[0])**2 + (y - ema[1])**2)**0.5 < self.max_jump
    def update(self, frame_data: np.ndarray):
        def _upd(ema, miss, slot):
            w = frame_data[slot]
            if not np.isnan(w[0]):
                wx, wy = float(w[0]), float(w[1])
                if ema is None: return (wx, wy), 0
                return (self.alpha*wx + (1-self.alpha)*ema[0], self.alpha*wy + (1-self.alpha)*ema[1]), 0
            miss += 1
            return (None if miss >= self.ttl else ema), miss
        self._ema_l, self._miss_l = _upd(self._ema_l, self._miss_l, LHAND_START)
        self._ema_r, self._miss_r = _upd(self._ema_r, self._miss_r, RHAND_START)
    def reset(self):
        self._ema_l = self._ema_r = None
        self._miss_l = self._miss_r = 0

def extract_hybrid(res_holistic, res_hands, tracker=None) -> np.ndarray:
    frame_data = np.full((543, 3), np.nan, dtype=np.float32)
    if res_holistic.face_landmarks:
        for i, lm in enumerate(res_holistic.face_landmarks):
            if i < 468: frame_data[FACE_START + i] = [lm.x, lm.y, lm.z]
    if res_holistic.pose_landmarks:
        for i, lm in enumerate(res_holistic.pose_landmarks):
            if i < 33: frame_data[POSE_START + i] = [lm.x, lm.y, lm.z]
    anchor_l = res_holistic.left_hand_landmarks[0] if res_holistic.left_hand_landmarks else None
    anchor_r = res_holistic.right_hand_landmarks[0] if res_holistic.right_hand_landmarks else None
    if tracker is not None:
        t_anc_l, t_anc_r = tracker.anchors()
        if anchor_l is None: anchor_l = t_anc_l
        if anchor_r is None: anchor_r = t_anc_r
    if anchor_l is None and res_holistic.pose_landmarks:
        anchor_l = res_holistic.pose_landmarks[15]
    if anchor_r is None and res_holistic.pose_landmarks:
        anchor_r = res_holistic.pose_landmarks[16]
    def _dist(a, b): return ((a.x - b.x)**2 + (a.y - b.y)**2) ** 0.5
    def _slot_for_hand(wrist) -> int:
        if anchor_l is not None and anchor_r is not None:
            slot = LHAND_START if _dist(wrist, anchor_l) < _dist(wrist, anchor_r) else RHAND_START
        elif anchor_l is not None:
            slot = LHAND_START if _dist(wrist, anchor_l) < 0.15 else RHAND_START
        elif anchor_r is not None:
            slot = RHAND_START if _dist(wrist, anchor_r) < 0.15 else LHAND_START
        else:
            slot = LHAND_START if wrist.x >= 0.5 else RHAND_START
        if tracker is not None and not tracker.is_plausible(wrist.x, wrist.y, slot):
            other = RHAND_START if slot == LHAND_START else LHAND_START
            if tracker.is_plausible(wrist.x, wrist.y, other): slot = other
        return slot
    def _is_duplicate_hand(wrist, threshold=0.1):
        if not np.isnan(frame_data[LHAND_START, 0]):
            l_wrist = frame_data[LHAND_START]
            if ((wrist.x - l_wrist[0])**2 + (wrist.y - l_wrist[1])**2)**0.5 < threshold: return True
        if not np.isnan(frame_data[RHAND_START, 0]):
            r_wrist = frame_data[RHAND_START]
            if ((wrist.x - r_wrist[0])**2 + (wrist.y - r_wrist[1])**2)**0.5 < threshold: return True
        return False
    if res_hands.hand_landmarks:
        for hand_lms in res_hands.hand_landmarks:
            wrist = hand_lms[0]
            slot  = _slot_for_hand(wrist)
            if np.isnan(frame_data[slot, 0]) and not _is_duplicate_hand(wrist):
                for i, lm in enumerate(hand_lms):
                    if i < 21: frame_data[slot + i] = [lm.x, lm.y, lm.z]
    if res_holistic.left_hand_landmarks and np.isnan(frame_data[LHAND_START, 0]):
        if not _is_duplicate_hand(res_holistic.left_hand_landmarks[0]):
            for i, lm in enumerate(res_holistic.left_hand_landmarks):
                if i < 21: frame_data[LHAND_START + i] = [lm.x, lm.y, lm.z]
    if res_holistic.right_hand_landmarks and np.isnan(frame_data[RHAND_START, 0]):
        if not _is_duplicate_hand(res_holistic.right_hand_landmarks[0]):
            for i, lm in enumerate(res_holistic.right_hand_landmarks):
                if i < 21: frame_data[RHAND_START + i] = [lm.x, lm.y, lm.z]
    return frame_data

def draw_landmarks_cv2(frame: np.ndarray, vec: np.ndarray) -> np.ndarray:
    h, w, _ = frame.shape
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20),
        (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
    ]
    def _draw_hand(start_idx, dot_color, line_color, radius=4, thickness=2):
        pts = []
        for i in range(21):
            lm = vec[start_idx + i]
            if np.isnan(lm[0]): pts.append(None)
            else: pts.append((int(lm[0] * w), int(lm[1] * h)))
        if all(p is None for p in pts): return
        for p1, p2 in HAND_CONNECTIONS:
            if pts[p1] is not None and pts[p2] is not None:
                cv2.line(frame, pts[p1], pts[p2], line_color, thickness)
        for p in pts:
            if p is not None:
                cv2.circle(frame, p, radius, dot_color, -1)
    _draw_hand(LHAND_START, (0, 255, 0), (144, 238, 144))
    _draw_hand(RHAND_START, (0, 0, 255), (128, 128, 255))
    return frame


# ─── PREPROCESS LAYER ─────────────────────────────────────────────────────────
def tf_nan_mean(x, axis=0, keepdims=False):
    return (
        tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x),
                      axis=axis, keepdims=keepdims)
        / tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)),
                        axis=axis, keepdims=keepdims)
    )

def tf_nan_std(x, center=None, axis=0, keepdims=False):
    if center is None:
        center = tf_nan_mean(x, axis=axis, keepdims=True)
    d = x - center
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis, keepdims=keepdims))

class Preprocess(tf.keras.layers.Layer):
    def __init__(self, max_len=MAX_LEN, point_landmarks=POINT_LANDMARKS, **kwargs):
        super().__init__(**kwargs)
        self.max_len         = max_len
        self.point_landmarks = point_landmarks

    def call(self, inputs):
        if len(inputs.shape) == 3:
            x = inputs[None, ...]
        else:
            x = inputs

        mean = tf_nan_mean(tf.gather(x, [17], axis=2), axis=[1, 2], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5, x.dtype), mean)
        x    = tf.gather(x, self.point_landmarks, axis=2)
        std  = tf_nan_std(x, center=mean, axis=[1, 2], keepdims=True)
        x    = (x - mean) / std

        if self.max_len is not None:
            x = x[:, :self.max_len]
        length = tf.shape(x)[1]
        x = x[..., :2]

        dx = tf.cond(
            tf.shape(x)[1] > 1,
            lambda: tf.pad(x[:, 1:] - x[:, :-1], [[0,0],[0,1],[0,0],[0,0]]),
            lambda: tf.zeros_like(x)
        )
        dx2 = tf.cond(
            tf.shape(x)[1] > 2,
            lambda: tf.pad(x[:, 2:] - x[:, :-2], [[0,0],[0,2],[0,0],[0,0]]),
            lambda: tf.zeros_like(x)
        )

        x = tf.concat([
            tf.reshape(x,   (-1, length, 2 * len(self.point_landmarks))),
            tf.reshape(dx,  (-1, length, 2 * len(self.point_landmarks))),
            tf.reshape(dx2, (-1, length, 2 * len(self.point_landmarks))),
        ], axis=-1)
        x = tf.where(tf.math.is_nan(x), tf.constant(0., x.dtype), x)
        return x


# ─── LOAD TFLITE MODEL ────────────────────────────────────────────────────────
def load_tflite_model(model_path: str = ""):
    if not model_path:
        if not os.path.exists(DEFAULT_MODEL_DIR):
            print(f"[ERROR] Thư mục {DEFAULT_MODEL_DIR} không tồn tại!")
            sys.exit(1)

        folders = [f for f in os.listdir(DEFAULT_MODEL_DIR) if os.path.isdir(os.path.join(DEFAULT_MODEL_DIR, f))]
        
        if not folders:
            print(f"[ERROR] Không có thư mục con nào trong {DEFAULT_MODEL_DIR}")
            sys.exit(1)
            
        print("\n" + "="*50)
        print(" DANH SÁCH CÁC LẦN HUẤN LUYỆN (MODEL) HIỆN CÓ")
        print("="*50)
        for i, f in enumerate(folders):
            print(f"  [{i+1}] {f}")
            
        print("-" * 50)
        print("[?] Nhập tên thư mục (hoặc số thứ tự) để tìm file .tflite:")
        choice = input(">> ").strip()
        
        selected_folder = None
        if choice.isdigit() and 1 <= int(choice) <= len(folders):
            selected_folder = folders[int(choice)-1]
        elif choice in folders:
            selected_folder = choice
        else:
            print("[ERROR] Lựa chọn không hợp lệ.")
            sys.exit(1)
            
        candidates = glob.glob(os.path.join(DEFAULT_MODEL_DIR, selected_folder, "*.tflite"))
        if not candidates:
            print(f"[ERROR] Không tìm thấy file .tflite nào trong {selected_folder}/")
            print("  → Hãy chạy export_tflite.py trước!")
            sys.exit(1)
            
        # Ưu tiên file 'best' hơn các file khác
        best_files = [c for c in candidates if "best" in os.path.basename(c)]
        model_path = best_files[0] if best_files else candidates[0]
        print(f"\n[INFO] Đã chọn model: {model_path}")

    if not os.path.exists(model_path):
        print(f"[ERROR] File không tồn tại: {model_path}")
        sys.exit(1)

    print(f"[INFO] Đang nạp mô hình TFLite...")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # In ra thông tin input/output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"[INFO] Input shape : {input_details[0]['shape']}")
    print(f"[INFO] Output shape: {output_details[0]['shape']}")
    
    return interpreter, input_details[0], output_details[0]


# ─── FRAME BUFFER & PREDICTOR ─────────────────────────────────────────────────
class SignPredictorTFLite:
    def __init__(self, interpreter, input_details, output_details, 
                 label_map: dict, label_original: dict, topk: int = 3, slide: int = 30):
        self.interpreter    = interpreter
        self.input_details  = input_details
        self.output_details = output_details
        self.label_map      = label_map
        self.label_original = label_original
        self.topk           = topk
        self.slide          = slide

        self.preprocess = Preprocess(max_len=MAX_LEN)
        self.buffer     = []

        self.last_label          = ""
        self.last_original       = ""
        self.last_confidence     = 0.0
        self.last_topk           = []
        self.is_detecting        = False
        self.hand_detected       = False
        self._grace_count        = 0
        self._recent_labels      = []
        self._last_known_vec     = None

    GRACE_MAX   = 8
    CLEAR_AFTER = 20

    def push_frame(self, vec_543: np.ndarray) -> bool:
        lhand = vec_543[LHAND_START:LHAND_END]
        rhand = vec_543[RHAND_START:RHAND_END]
        has_hand = not (np.isnan(lhand).all() and np.isnan(rhand).all())

        self.hand_detected = has_hand

        if has_hand:
            self.is_detecting    = True
            self._grace_count    = 0
            self._last_known_vec = vec_543.copy()
            self.buffer.append(vec_543)
        elif self._grace_count < self.GRACE_MAX and self._last_known_vec is not None:
            self._grace_count += 1
            self.buffer.append(self._last_known_vec)
        else:
            self._grace_count += 1
            if self._grace_count >= self.CLEAR_AFTER:
                self.buffer          = []
                self.is_detecting    = False
                self._last_known_vec = None

        if len(self.buffer) >= BUFFER_LEN:
            self._predict()
            self.buffer = self.buffer[self.slide:]
            return True

        return False

    def _predict(self):
        seq = np.stack(self.buffer[:BUFFER_LEN], axis=0)   # (60, 543, 3)  NumPy
        seq_tf = tf.constant(seq, dtype=tf.float32)

        # Preprocess → (1, 60, 708)
        features = self.preprocess(seq_tf)                   # tf.Tensor (1, 60, 708)
        features = tf.cast(features, tf.float32).numpy()     # → NumPy (1, 60, 708)

        # Padding lên đủ MAX_LEN (384) để khớp với đầu vào model
        T = features.shape[1]    # <-- Dùng Python-level shape (không phải tf.shape)
        if T < MAX_LEN:
            pad = np.full((1, MAX_LEN - T, CHANNELS), PAD, dtype=np.float32)
            features = np.concatenate([features, pad], axis=1)  # NumPy concat

        # TFLite yêu cầu đúng dtype (float32 hoặc float16 tùy model đã nén)
        input_dtype = self.input_details["dtype"]
        input_data = features.astype(input_dtype)            # NumPy array, đúng dtype

        # INFERENCE VỚI TFLITE (set_tensor chỉ chấp nhận NumPy, không chấp nhận Tensor)
        self.interpreter.set_tensor(self.input_details["index"], input_data)
        self.interpreter.invoke()
        logits = self.interpreter.get_tensor(self.output_details["index"])  # NumPy (1, 4)

        probs = tf.nn.softmax(logits[0]).numpy()

        # Top-K
        top_idx = np.argsort(probs)[::-1][:self.topk]
        self.last_topk = [
            (self.label_map.get(i, f"class_{i}"),
             self.label_original.get(i, f"class_{i}"),
             float(probs[i]))
            for i in top_idx
        ]

        best_idx  = top_idx[0]
        raw_label = self.label_map.get(best_idx, f"class_{best_idx}")
        raw_conf  = float(probs[best_idx])

        # Smoothing: majority vote
        self._recent_labels.append(raw_label)
        if len(self._recent_labels) > 3:
            self._recent_labels.pop(0)

        from collections import Counter
        vote = Counter(self._recent_labels).most_common(1)[0]
        if vote[1] >= 2:
            self.last_label      = raw_label
            self.last_original   = self.label_original.get(best_idx, raw_label)
            self.last_confidence = raw_conf

    def reset(self):
        self.buffer          = []
        self.last_label      = ""
        self.last_original   = ""
        self.last_confidence = 0.0
        self.last_topk       = []
        self.is_detecting    = False
        self._grace_count    = 0
        self._recent_labels  = []
        self._last_known_vec = None


# ─── DRAW OVERLAY ─────────────────────────────────────────────────────────────
def draw_overlay(frame: np.ndarray, predictor: SignPredictorTFLite,
                 buffer_size: int, topk: int):
    h, w, _ = frame.shape
    
    if predictor.last_label:
        conf_pct = predictor.last_confidence * 100
        
        text = f"{predictor.last_label} : {conf_pct:.0f}%"
        
        if conf_pct > 70:
            color = (0, 255, 0) # Xanh lá
        elif conf_pct > 40:
            color = (0, 200, 255) # Vàng/Cam
        else:
            color = (0, 0, 255) # Đỏ
            
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.5
        thickness = 2
        
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        x = max(50, (w - text_width) // 2)
        y = h - 50
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 15, y - text_height - 15), (x + text_width + 15, y + 15), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
        
    return frame


def run_camera(model_path: str = "", camera_idx: int = 0, topk: int = 3):
    from concurrent.futures import ThreadPoolExecutor
    
    label_map, label_original = load_label_map()
    interpreter, input_details, output_details = load_tflite_model(model_path)
    holistic_model, hand_model = init_mediapipe()
    
    predictor = SignPredictorTFLite(
        interpreter, input_details, output_details,
        label_map, label_original, topk=topk
    )

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Không mở được camera {camera_idx}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    cv2.namedWindow("SignBridge AI (TFLite)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SignBridge AI (TFLite)", 640, 480)

    print("\n[INFO] Camera đang chạy (Phiên bản TFLite + Hybrid Tracker).")
    os.makedirs("captures", exist_ok=True)
    frame_count = 0
    fps_time    = time.time()
    fps         = 0.0
    t_start_ns  = time.perf_counter_ns()

    tracker = HandTracker()
    with ThreadPoolExecutor(max_workers=2) as mp_executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_time)
                fps_time = time.time()

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            current_ms = (time.perf_counter_ns() - t_start_ns) // 1_000_000
            
            future_holistic = mp_executor.submit(holistic_model.detect_for_video, mp_image, current_ms)
            future_hands    = mp_executor.submit(hand_model.detect_for_video,    mp_image, current_ms)

            res_holistic = future_holistic.result()
            res_hands    = future_hands.result()

            vec = extract_hybrid(res_holistic, res_hands, tracker)
            tracker.update(vec)
            predictor.push_frame(vec)

            draw_landmarks_cv2(frame, vec)

            frame = cv2.flip(frame, 1)

            frame = draw_overlay(frame, predictor, len(predictor.buffer), topk)
            cv2.putText(frame, f"FPS: {fps:.0f} (TFLITE HYBRID)", (frame.shape[1] - 180, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            l_ok = not np.isnan(vec[LHAND_START, 0])
            r_ok = not np.isnan(vec[RHAND_START, 0])
            hand_status = f"L:{'OK' if l_ok else '--'}  R:{'OK' if r_ok else '--'}  buf:{len(predictor.buffer):02d}/{BUFFER_LEN}"
            cv2.putText(frame, hand_status, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

            cv2.imshow("SignBridge AI (TFLite)", frame)

            if cv2.getWindowProperty("SignBridge AI (TFLite)", cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                predictor.reset()
                tracker.reset()
            elif key == ord('s'):
                fname = f"captures/capture_{int(time.time())}.png"
                cv2.imwrite(fname, frame)

    cap.release()
    holistic_model.close()
    hand_model.close()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="SignBridge TFLite - Nhận diện realtime")
    parser.add_argument("--model", type=str, default="", help="Đường dẫn file .tflite")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    run_camera(
        model_path = args.model,
        camera_idx = args.camera,
        topk       = args.topk,
    )

if __name__ == "__main__":
    main()
