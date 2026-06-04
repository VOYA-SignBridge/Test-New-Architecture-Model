"""
camera_demo.py
==============
Nhận diện ký hiệu tay realtime từ camera.

Pipeline:
  Camera → MediaPipe Holistic → buffer 60 frames → Model → Hiện từ lên màn hình

CÀI ĐẶT THÊM:
  pip install opencv-python mediapipe

CÁCH CHẠY:
  python camera_demo.py
  python camera_demo.py --camera 1   # nếu có nhiều camera
  python camera_demo.py --topk 3     # hiện top 3 gợi ý
"""

import os
import sys
import time
import argparse
import glob
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# ─── KIỂM TRA THƯ VIỆN ────────────────────────────────────────────────────────
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


# ─── CONSTANTS (Khớp 100% với train.py) ──────────────────────────────────────
MAX_LEN         = 384
BUFFER_LEN      = 60       # số frame thu thập thực tế trước khi dự đoán
NUM_CLASSES     = 4
PAD             = -100.

# Lấy chính xác các cụm điểm giống train.py
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
CHANNELS        = 6 * NUM_NODES   # 708

DEFAULT_MODEL_DIR   = "./output"
DEFAULT_DATASET_DIR = "./dataset/Vietnamese"

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
    if not os.path.exists(HOLISTIC_MODEL_PATH):
        print(f"[INFO] Đang tải model Holistic ({HOLISTIC_MODEL_PATH})...")
        urllib.request.urlretrieve(HOLISTIC_URL, HOLISTIC_MODEL_PATH)
    if not os.path.exists(HANDS_MODEL_PATH):
        print(f"[INFO] Đang tải model Hand ({HANDS_MODEL_PATH})...")
        urllib.request.urlretrieve(HANDS_URL, HANDS_MODEL_PATH)
    print("[INFO] Đã tải đủ các mô hình MediaPipe!")


# ─── LOAD LABEL MAP ───────────────────────────────────────────────────────────
def load_label_map(dataset_dir: str = DEFAULT_DATASET_DIR):
    """Đọc mapping class_idx → tên tiếng Việt từ label_map.json."""
    import json
    # label_map.json nằm trong data/Vietnamese/TFRecord/ (do prepare_tfrecords.py tạo ra)
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

    print("[INFO] Label map:")
    for i in sorted(label_map):
        print(f"  [{i}] {label_original[i]}")

    return label_map, label_original


# ─── MEDIAPIPE SETUP (Tasks API 0.10+) ────────────────────────────────────────
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


# ─── EMA TEMPORAL HAND TRACKER ───────────────────────────────────────────────
class HandTracker:
    """
    Exponential Moving Average tracker cho tay Trái và Phải.
    Thay vì chỉ nhớ 1 frame trước, HandTracker duy trì vị trí EMA
    (mượt hơn, kháng nhiễu hơn) và tự xoá khi tay biến mất > miss_ttl frame.

    Ngoài ra cung cấp is_plausible() để phát hiện detection nhảy bên đột ngột.
    """
    class _P:
        """Lightweight point object tương thích với MediaPipe landmark."""
        __slots__ = ('x', 'y')
        def __init__(self, x: float, y: float): self.x = x; self.y = y

    def __init__(self, ema_alpha: float = 0.45, miss_ttl: int = 5,
                 max_jump: float = 0.28):
        """
        ema_alpha : hệ số EMA ∈ (0,1). Lớn=cập nhật nhanh, Nhỏ=mượt hơn.
        miss_ttl  : frame không thấy tay liên tiếp trước khi xoá EMA.
        max_jump  : khoảng cách (normalized) tối đa 1 tay di chuyển/frame.
                    Vượt quá → detection bị nghi sai bên, thử gán ngược.
        """
        self.alpha    = ema_alpha
        self.ttl      = miss_ttl
        self.max_jump = max_jump
        self._ema_l   = None   # (x, y) EMA tay Trái
        self._ema_r   = None   # (x, y) EMA tay Phải
        self._miss_l  = 0
        self._miss_r  = 0

    def anchors(self):
        """Trả về (anchor_l, anchor_r) — dùng làm điểm neo trong extract_hybrid."""
        P = self._P
        return (
            P(*self._ema_l) if self._ema_l else None,
            P(*self._ema_r) if self._ema_r else None,
        )

    def is_plausible(self, x: float, y: float, slot: int) -> bool:
        """Kiểm tra detection có hợp lý không so với EMA (không nhảy xa quá)."""
        ema = self._ema_l if slot == LHAND_START else self._ema_r
        if ema is None:
            return True  # chưa có lịch sử → luôn chấp nhận
        return ((x - ema[0])**2 + (y - ema[1])**2)**0.5 < self.max_jump

    def update(self, frame_data: np.ndarray):
        """Cập nhật EMA từ kết quả extract_hybrid(). Gọi sau mỗi frame."""
        def _upd(ema, miss, slot):
            w = frame_data[slot]
            if not np.isnan(w[0]):
                wx, wy = float(w[0]), float(w[1])
                if ema is None:
                    return (wx, wy), 0
                return (self.alpha*wx + (1-self.alpha)*ema[0],
                        self.alpha*wy + (1-self.alpha)*ema[1]), 0
            miss += 1
            return (None if miss >= self.ttl else ema), miss
        self._ema_l, self._miss_l = _upd(self._ema_l, self._miss_l, LHAND_START)
        self._ema_r, self._miss_r = _upd(self._ema_r, self._miss_r, RHAND_START)

    def reset(self):
        self._ema_l = self._ema_r = None
        self._miss_l = self._miss_r = 0


def extract_hybrid(res_holistic, res_hands, tracker=None) -> np.ndarray:
    """
    Trích xuất Hybrid vector (543, 3) từ Holistic và HandLandmarker.
    Cấu trúc: [0:468] Face | [468:489] LHand (user) | [489:522] Pose | [522:543] RHand (user)

    CHIẾN LƯỢC PHÂN LOẠI TAY — Holistic-Anchor Assignment:
    ─────────────────────────────────────────────────────────────────
    Holistic đã tự xử lý mirror nội bộ và cung cấp 2 anchor TIN CẬY:
      • left_hand_landmarks  → Cổ tay TRÁI người dùng  (chuẩn, đã flip)
      • right_hand_landmarks → Cổ tay PHẢI người dùng (chuẩn, đã flip)

    Quy trình:
      1. Holistic cung cấp 1–2 cổ tay anchor (trái/phải).
      2. Với MỖI bàn tay từ HandLandmarker:
           - Tính khoảng cách cổ tay (landmark 0) tới anchor Trái và anchor Phải.
           - Gán vào slot (LHAND / RHAND) tương ứng với anchor GẦN hơn.
           - Nếu Holistic không thấy tay nào làm anchor (mất toàn bộ): fallback sang Pose Wrist.
           - Nếu mất luôn Pose: fallback sang x-position (x>=0.5 ảnh gốc = Trái user).
      3. Slot đã có dữ liệu không bị ghi đè (first-win — HandLandmarker ưu tiên).
      4. Sau khi HandLandmarker chạy xong, dùng Holistic left/right_hand_landmarks để
         lấp đầy slot còn NaN (chỉ khi anchor đó không trùng với slot đã có).
    """
    frame_data = np.full((543, 3), np.nan, dtype=np.float32)

    # ── 1. MẶT & TƯ THẾ (từ Holistic, luôn đáng tin) ─────────────────────────
    if res_holistic.face_landmarks:
        for i, lm in enumerate(res_holistic.face_landmarks):
            if i < 468:
                frame_data[FACE_START + i] = [lm.x, lm.y, lm.z]

    if res_holistic.pose_landmarks:
        for i, lm in enumerate(res_holistic.pose_landmarks):
            if i < 33:
                frame_data[POSE_START + i] = [lm.x, lm.y, lm.z]

    # ── 2. BÀN TAY ─────────────────────────────────────────────────────────────
    # Nguồn anchor ưu tiên cao nhất: Holistic left/right_hand_landmarks
    anchor_l = res_holistic.left_hand_landmarks[0]  if res_holistic.left_hand_landmarks  else None
    anchor_r = res_holistic.right_hand_landmarks[0] if res_holistic.right_hand_landmarks else None

    # Fallback 1: EMA Temporal Tracking (ổn định hơn single-frame lookback)
    if tracker is not None:
        t_anc_l, t_anc_r = tracker.anchors()
        if anchor_l is None: anchor_l = t_anc_l
        if anchor_r is None: anchor_r = t_anc_r

    # Fallback 2: Pose Wrist
    if anchor_l is None and res_holistic.pose_landmarks:
        anchor_l = res_holistic.pose_landmarks[15]
    if anchor_r is None and res_holistic.pose_landmarks:
        anchor_r = res_holistic.pose_landmarks[16]

    def _dist(a, b):
        return ((a.x - b.x)**2 + (a.y - b.y)**2) ** 0.5

    def _slot_for_hand(wrist) -> int:
        """Gán slot dựa trên anchor gần nhất, có kiểm tra plausibility từ EMA."""
        if anchor_l is not None and anchor_r is not None:
            slot = LHAND_START if _dist(wrist, anchor_l) < _dist(wrist, anchor_r) else RHAND_START
        elif anchor_l is not None:
            slot = LHAND_START if _dist(wrist, anchor_l) < 0.15 else RHAND_START
        elif anchor_r is not None:
            slot = RHAND_START if _dist(wrist, anchor_r) < 0.15 else LHAND_START
        else:
            slot = LHAND_START if wrist.x >= 0.5 else RHAND_START

        # Plausibility check: nếu detection nhảy xa hơn max_jump so với EMA,
        # thử gán sang slot ngược lại — có thể EMA bên kia hợp lý hơn.
        if tracker is not None and not tracker.is_plausible(wrist.x, wrist.y, slot):
            other = RHAND_START if slot == LHAND_START else LHAND_START
            if tracker.is_plausible(wrist.x, wrist.y, other):
                slot = other
        return slot

    def _is_duplicate_hand(wrist, threshold=0.1):
        """Kiểm tra xem tay này đã được ghi vào slot nào chưa (chống 1 tay thành 2 tay)"""
        # Kiểm tra slot Trái
        if not np.isnan(frame_data[LHAND_START, 0]):
            l_wrist = frame_data[LHAND_START]
            if ((wrist.x - l_wrist[0])**2 + (wrist.y - l_wrist[1])**2)**0.5 < threshold:
                return True
        # Kiểm tra slot Phải
        if not np.isnan(frame_data[RHAND_START, 0]):
            r_wrist = frame_data[RHAND_START]
            if ((wrist.x - r_wrist[0])**2 + (wrist.y - r_wrist[1])**2)**0.5 < threshold:
                return True
        return False

    # ── 2a. Nguồn CHÍNH: HandLandmarker ───────────────────────────────────────
    if res_hands.hand_landmarks:
        for hand_lms in res_hands.hand_landmarks:
            wrist = hand_lms[0]
            slot  = _slot_for_hand(wrist)
            # First-win: không ghi đè slot đã có và kiểm tra chống trùng
            if np.isnan(frame_data[slot, 0]) and not _is_duplicate_hand(wrist):
                for i, lm in enumerate(hand_lms):
                    if i < 21:
                        frame_data[slot + i] = [lm.x, lm.y, lm.z]

    # ── 2b. Nguồn FALLBACK: Holistic left/right_hand_landmarks ────────────────
    # Holistic đã xác nhận đúng trái/phải; chỉ dùng nếu slot còn NaN.
    if res_holistic.left_hand_landmarks and np.isnan(frame_data[LHAND_START, 0]):
        if not _is_duplicate_hand(res_holistic.left_hand_landmarks[0]):
            for i, lm in enumerate(res_holistic.left_hand_landmarks):
                if i < 21:
                    frame_data[LHAND_START + i] = [lm.x, lm.y, lm.z]

    if res_holistic.right_hand_landmarks and np.isnan(frame_data[RHAND_START, 0]):
        if not _is_duplicate_hand(res_holistic.right_hand_landmarks[0]):
            for i, lm in enumerate(res_holistic.right_hand_landmarks):
                if i < 21:
                    frame_data[RHAND_START + i] = [lm.x, lm.y, lm.z]

    return frame_data




def draw_landmarks_cv2(frame: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """
    Vẽ landmarks bằng OpenCV thuần từ array vec_543 (ảnh gốc chưa bị mirror)
    """
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
        # Lấy 21 điểm của tay; None nếu điểm đó là NaN
        pts = []
        for i in range(21):
            lm = vec[start_idx + i]
            if np.isnan(lm[0]):
                pts.append(None)
            else:
                pts.append((int(lm[0] * w), int(lm[1] * h)))

        # Nếu toàn bộ tay là NaN thì bỏ qua
        if all(p is None for p in pts):
            return
            
        # Vẽ các đoạn thẳng (xương) — bỏ qua nếu một trong 2 đầu là NaN
        for p1, p2 in HAND_CONNECTIONS:
            if pts[p1] is not None and pts[p2] is not None:
                cv2.line(frame, pts[p1], pts[p2], line_color, thickness)
                
        # Vẽ các chấm (khớp) — bỏ qua điểm NaN
        for p in pts:
            if p is not None:
                cv2.circle(frame, p, radius, dot_color, -1)

    _draw_hand(LHAND_START, (0, 255, 0), (144, 238, 144))  # tay trái người dùng — xanh lá
    _draw_hand(RHAND_START, (0, 0, 255), (128, 128, 255))  # tay phải người dùng — đỏ
    
    return frame


# ─── PREPROCESS (phải khớp 100% với train.py) ────────────────────────────────
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
    """Giống hệt train.py — normalize, gather landmarks, velocity, fill NaN."""
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


# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
def load_model(model_path: str = ""):
    """
    Load model từ file .weights.h5.
    Hỏi người dùng chọn thư mục chứa model từ output/
    """
    if not model_path:
        if not os.path.exists(DEFAULT_MODEL_DIR):
            print(f"[ERROR] Thư mục {DEFAULT_MODEL_DIR} không tồn tại!")
            print("  → Hãy train trước: python train.py")
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
        print("[?] Nhập tên thư mục (vd: 2-42-3-6-2026) HOẶC nhập số thứ tự [1, 2, ...]:")
        choice = input(">> ").strip()
        
        selected_folder = None
        if choice.isdigit() and 1 <= int(choice) <= len(folders):
            selected_folder = folders[int(choice)-1]
        elif choice in folders:
            selected_folder = choice
        else:
            print("[ERROR] Lựa chọn không hợp lệ. Đang thoát...")
            sys.exit(1)
            
        # Tìm file .weights.h5 trong thư mục đã chọn
        candidates = glob.glob(os.path.join(DEFAULT_MODEL_DIR, selected_folder, "*.weights.h5"))
        if not candidates:
            print(f"[ERROR] Không tìm thấy file .weights.h5 nào bên trong {selected_folder}/")
            sys.exit(1)

        # Ưu tiên file 'best' hơn file 'last'
        best_files = [c for c in candidates if "best" in os.path.basename(c)]
        model_path = best_files[0] if best_files else candidates[0]
        print(f"\n[INFO] Đã chọn model: {model_path}")

    if not os.path.exists(model_path):
        print(f"[ERROR] File không tồn tại: {model_path}")
        sys.exit(1)

    # Import kiến trúc từ train.py để rebuild rồi load weights
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from train import get_model

    # Tự động nhận diện dim từ tên file (vd: islr-fp16-192-... hoặc islr-fp16-384-...)
    dim = 192  # giá trị mặc định
    for candidate_dim in [384, 256, 192]:
        if f"-{candidate_dim}-" in os.path.basename(model_path):
            dim = candidate_dim
            break
    print(f"[INFO] Tự động nhận diện kiến trúc: dim={dim}")

    model = get_model(max_len=MAX_LEN, dropout_step=0, dim=dim)
    model.load_weights(model_path)
    print(f"[INFO] Model ready. Input={model.input_shape}, Output={model.output_shape}")
    return model


# ─── FRAME BUFFER & PREDICTOR ─────────────────────────────────────────────────

class SignPredictor:
    """
    Tích lũy frame từ camera vào buffer 60 frames.
    Khi đủ → chạy model → trả kết quả.
    """

    def __init__(self, model, label_map: dict, label_original: dict,
                 topk: int = 3, slide: int = 30):
        self.model          = model
        self.label_map      = label_map
        self.label_original = label_original
        self.topk           = topk
        self.slide          = slide

        self.preprocess = Preprocess(max_len=MAX_LEN)
        self.buffer     = []

        self.last_label         = ""
        self.last_original      = ""
        self.last_confidence    = 0.0
        self.last_topk          = []
        self.is_detecting       = False
        self.hand_detected      = False
        self._grace_count       = 0
        self._recent_labels     = []
        self._last_known_vec    = None

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
        seq    = np.stack(self.buffer[:BUFFER_LEN], axis=0)   # (60, 543, 3)
        seq_tf = tf.constant(seq, dtype=tf.float32)

        features = self.preprocess(seq_tf)
        features = tf.cast(features, tf.float32)

        T = tf.shape(features)[1]
        if T < MAX_LEN:
            pad = tf.fill([1, MAX_LEN - T, CHANNELS], PAD)
            features = tf.concat([features, pad], axis=1)

        logits = self.model(features, training=False)
        probs  = tf.nn.softmax(logits[0]).numpy()

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

        # Smoothing: majority vote trong 3 kết quả gần nhất
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
        self.buffer             = []
        self.last_label         = ""
        self.last_original      = ""
        self.last_confidence    = 0.0
        self.last_topk          = []
        self.is_detecting       = False
        self._grace_count       = 0
        self._recent_labels     = []
        self._last_known_vec    = None  # ← phải reset kểo grace period bơm data cũ vào buffer mới


# ─── DRAW OVERLAY ─────────────────────────────────────────────────────────────

def draw_overlay(frame: np.ndarray, predictor: SignPredictor,
                 buffer_size: int, topk: int):
    h, w, _ = frame.shape
    
    if predictor.last_label:
        conf_pct = predictor.last_confidence * 100
        
        # Sử dụng predictor.last_label để lấy từ không dấu (như bạn yêu cầu) thay vì last_original
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
        
        # Lấy kích thước chữ để căn giữa và làm nền đen
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        x = max(50, (w - text_width) // 2)
        y = h - 50
        
        # Vẽ nền mờ đen
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 15, y - text_height - 15), (x + text_width + 15, y + 15), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Vẽ viền chữ màu đen cho rõ nét, rồi mới vẽ màu thật đè lên trên
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
        
    return frame


# ─── MAIN CAMERA LOOP ─────────────────────────────────────────────────────────

def run_camera(model_path: str = "", camera_idx: int = 0,
               topk: int = 3, dataset_dir: str = DEFAULT_DATASET_DIR):
    label_map, label_original = load_label_map(dataset_dir)
    model = load_model(model_path)
    holistic_model, hand_model = init_mediapipe()   # Khởi tạo 2 mô hình (Hybrid)
    predictor  = SignPredictor(model, label_map, label_original, topk=topk)

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Không mở được camera {camera_idx}")
        holistic_model.close()
        hand_model.close()
        return

    # Đặt kích thước camera nhỏ lại để tăng FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    cv2.namedWindow("SignBridge AI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SignBridge AI", 640, 480)

    print("\n[INFO] Camera đang chạy chế độ HYBRID (Holistic + Hands).")
    os.makedirs("captures", exist_ok=True)

    frame_idx    = 0
    fps_time     = time.time()
    fps          = 0.0
    # Dùng perf_counter (ns) để tính timestamp thực tế — tránh trườt do FPS không chính xác
    t_start_ns   = time.perf_counter_ns()

    try:
        # Tạo thread pool 2 workers — mỗi worker đảm nhận 1 mô hình MediaPipe.
        # MediaPipe inference là C++ extension nên nó TỰ ĐỘNG nhả GIL trong lúc chạy,
        # cho phép 2 threads này thực sự chiếm 2 lõi CPU khác nhau cùng lúc.
        # Tạo executor 1 lần bên ngoài vòng lặp để tránh overhead tạo/xoá thread mỗi frame.
        tracker = HandTracker()   # EMA temporal tracker cho tay Trái/Phải
        with ThreadPoolExecutor(max_workers=2) as mp_executor:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % 30 == 0:
                    fps      = 30 / (time.time() - fps_time)
                    fps_time = time.time()

                # BƯỚC 1: XỬ LÝ KHUNG HÌNH GỐC (CHƯA LẬT)
                # MediaPipe sẽ xử lý frame gốc để lấy toạ độ chuẩn không bị ngược trái/phải
                # (khớp hoàn toàn với dữ liệu training).
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                
                # Timestamp thực tế tính từ perf_counter — được đảm bảo tăng đều, không bị trôi
                current_ms = (time.perf_counter_ns() - t_start_ns) // 1_000_000
                
                # Dispatch 2 mô hình sang 2 threads — chạy SONG SONG trên 2 CPU cores.
                # mp_image là read-only nên an toàn để 2 threads cùng đọc đồng thời.
                future_holistic = mp_executor.submit(holistic_model.detect_for_video, mp_image, current_ms)
                future_hands    = mp_executor.submit(hand_model.detect_for_video,    mp_image, current_ms)

                # Chờ cả 2 kết quả (thread nào xong trước thì chờ thread còn lại)
                res_holistic = future_holistic.result()
                res_hands    = future_hands.result()

                # Trích xuất 543 điểm chuẩn (Hybrid + EMA Temporal Tracking)
                vec = extract_hybrid(res_holistic, res_hands, tracker)
                tracker.update(vec)   # Cập nhật EMA ngay sau mỗi frame
                predictor.push_frame(vec)

                # Vẽ skeleton lên frame CHƯA LẬT (vẽ trực tiếp từ mảng vec)
                draw_landmarks_cv2(frame, vec)

                # BƯỚC 2: LẬT KHUNG HÌNH (MIRROR) ĐỂ HIỂN THỊ CHO NGƯỜI DÙNG DỄ NHÌN
                frame = cv2.flip(frame, 1)

                # BƯỚC 3: VẼ GIAO DIỆN CHỮ LÊN KHUNG HÌNH (lúc này chữ không bị lật ngược)
                frame = draw_overlay(frame, predictor, len(predictor.buffer), topk)
                cv2.putText(frame, f"FPS: {fps:.0f} (HYBRID)", (frame.shape[1] - 150, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                # ── Status bar: tay L/R và buffer (góc trái trên, sau khi flip) ──
                l_ok = not np.isnan(vec[LHAND_START, 0])
                r_ok = not np.isnan(vec[RHAND_START, 0])
                hand_status = f"L:{'OK' if l_ok else '--'}  R:{'OK' if r_ok else '--'}  buf:{len(predictor.buffer):02d}/{BUFFER_LEN}"
                cv2.putText(frame, hand_status, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

                cv2.imshow("SignBridge AI", frame)

                # Hỗ trợ bấm nút X trên cửa sổ để tắt
                if cv2.getWindowProperty("SignBridge AI", cv2.WND_PROP_VISIBLE) < 1:
                    break

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('r'):
                    predictor.reset()
                elif key == ord('s'):
                    fname = f"captures/capture_{int(time.time())}.png"
                    cv2.imwrite(fname, frame)
    finally:
        cap.release()
        holistic_model.close()
        hand_model.close()
        cv2.destroyAllWindows()



def main():
    parser = argparse.ArgumentParser(description="SignBridge - Nhận diện ký hiệu tay realtime")
    parser.add_argument("--model", type=str, default="",
                        help="Đường dẫn file .weights.h5")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_DIR)
    args = parser.parse_args()

    run_camera(
        model_path  = args.model,
        camera_idx  = args.camera,
        topk        = args.topk,
        dataset_dir = args.dataset,
    )

if __name__ == "__main__":
    main()
