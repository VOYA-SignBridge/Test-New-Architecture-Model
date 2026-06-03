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
import csv
import sys
import time
import argparse
import glob
import numpy as np

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

MP_MODEL_PATH = "holistic_landmarker.task"
MP_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"

def _ensure_mp_model():
    """Tải model MediaPipe nếu chưa có."""
    if not os.path.exists(MP_MODEL_PATH):
        print(f"[INFO] Đang tải model MediaPipe ({MP_MODEL_PATH})...")
        urllib.request.urlretrieve(MP_MODEL_URL, MP_MODEL_PATH)
        print("[INFO] Tải xong!")


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
    """Khởi tạo HolisticLandmarker dùng MediaPipe Tasks API mới (>=0.10)."""
    _ensure_mp_model()
    base_options = mp_python.BaseOptions(model_asset_path=MP_MODEL_PATH)
    options = mp_vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,  # 0.5 giúp detect tốt hơn khi tay co lại
        output_face_blendshapes=False,
    )
    return mp_vision.HolisticLandmarker.create_from_options(options)


def extract_holistic(results) -> np.ndarray:
    """
    Trích xuất vector (543, 3) từ kết quả Tasks API.
    Cấu trúc y hệt mảng 543 điểm chuẩn:
      0-467: Face  |  468-488: Left Hand  |  489-521: Pose  |  522-542: Right Hand
    Tasks API trả về list NormalizedLandmark trực tiếp (không có .landmark).
    """
    frame_data = np.full((543, 3), np.nan, dtype=np.float32)

    if results.face_landmarks:
        for i, lm in enumerate(results.face_landmarks):
            if i < 468:
                frame_data[i] = [lm.x, lm.y, lm.z]

    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks):
            frame_data[468 + i] = [lm.x, lm.y, lm.z]

    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks):
            frame_data[489 + i] = [lm.x, lm.y, lm.z]

    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks):
            frame_data[522 + i] = [lm.x, lm.y, lm.z]

    return frame_data


def draw_landmarks_cv2(frame: np.ndarray, results) -> np.ndarray:
    """
    Vẽ landmarks bằng OpenCV thuần (Tasks API không có mp_drawing).
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

    def _draw_hand(landmarks, dot_color, line_color, radius=4, thickness=2):
        if not landmarks: return
        
        pts = []
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            pts.append((cx, cy))
            
        # Vẽ các đoạn thẳng (xương)
        for p1, p2 in HAND_CONNECTIONS:
            if p1 < len(pts) and p2 < len(pts):
                cv2.line(frame, pts[p1], pts[p2], line_color, thickness)
                
        # Vẽ các chấm (khớp)
        for p in pts:
            cv2.circle(frame, p, radius, dot_color, -1)

    _draw_hand(results.left_hand_landmarks,  (0, 255, 0), (144, 238, 144))  # tay trái — xanh lá
    _draw_hand(results.right_hand_landmarks, (0, 0, 255), (128, 128, 255))  # tay phải — đỏ
    
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
        self.slide          = slide  # slide = 30: predict mỗi 30 frame, đủ để tín hiệu ổn định

        self.preprocess = Preprocess(max_len=MAX_LEN)
        self.buffer     = []  # list of ndarray (543, 3)

        self.last_label         = ""
        self.last_original      = ""
        self.last_confidence    = 0.0
        self.last_topk          = []
        self.is_detecting       = False
        self.hand_detected      = False
        self._grace_count       = 0   # đếm số frame liên tiếp không có tay
        self._recent_labels     = []  # lưu 3 kết quả gần nhất để smoothing
        self._last_known_vec    = None  # vị trí tay lần cuối detect được

    GRACE_MAX   = 8   # số frame tay tạm khuất vẫn giữ buffer (tăng từ 5 lên 8)
    CLEAR_AFTER = 20  # số frame sau khi tay biến hẳn mới xóa buffer

    def push_frame(self, vec_543: np.ndarray) -> bool:
        lhand = vec_543[468:489]
        rhand = vec_543[522:543]
        has_hand = not (np.isnan(lhand).all() and np.isnan(rhand).all())

        self.hand_detected = has_hand

        if has_hand:
            self.is_detecting    = True
            self._grace_count    = 0
            self._last_known_vec = vec_543.copy()  # lưu vị trí mới nhất
            self.buffer.append(vec_543)
        elif self._grace_count < self.GRACE_MAX and self._last_known_vec is not None:
            # ── "LAST KNOWN POSITION" TRACKING ──────────────────────────
            # Tay tạm khuất (nằm ngang, co lại, bị che khuất...) nhưng
            # ta vẫn biết tay đang ở đâu (lần cuối detect được).
            # Điền vị trí cũ vào buffer thay vì NaN để:
            #   1) buffer không bị đứt → signal liên tục
            #   2) model không nhận toàn NaN → predict ít nhiễu hơn
            self._grace_count += 1
            self.buffer.append(self._last_known_vec)
        else:
            # Tay mất hẳn quá lâu
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

        # Smoothing: chỉ cập nhật kết quả nếu cùng tên 2/3 lần gần nhất
        self._recent_labels.append(raw_label)
        if len(self._recent_labels) > 3:
            self._recent_labels.pop(0)

        # Majority vote trong 3 kết quả gần nhất
        from collections import Counter
        vote = Counter(self._recent_labels).most_common(1)[0]
        if vote[1] >= 2:  # ít nhất 2/3 lần cùng tên mới hiện
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
    landmarker = init_mediapipe()   # HolisticLandmarker (Tasks API)
    predictor  = SignPredictor(model, label_map, label_original, topk=topk)

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Không mở được camera {camera_idx}")
        landmarker.close()
        return

    # Đặt kích thước camera nhỏ lại
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    
    cv2.namedWindow("SignBridge AI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SignBridge AI", 800, 600)

    print("\n[INFO] Camera đang chạy (Tasks API MediaPipe 0.10+).")
    os.makedirs("captures", exist_ok=True)

    frame_count   = 0
    fps_time      = time.time()
    fps           = 0.0
    current_ms    = 0          # timestamp tăng dần cho detect_for_video
    cap_fps       = cap.get(cv2.CAP_PROP_FPS)
    frame_ms      = int(1000 / cap_fps) if cap_fps > 0 else 33  # ~30fps

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 30 == 0:
                fps      = 30 / (time.time() - fps_time)
                fps_time = time.time()

            # BƯỚC 1: XỬ LÝ KHUNG HÌNH GỐC (CHƯA LẬT)
            # MediaPipe sẽ xử lý frame gốc để lấy toạ độ chuẩn không bị ngược trái/phải
            # (khớp hoàn toàn với dữ liệu training).
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results  = landmarker.detect_for_video(mp_image, current_ms)
            current_ms += frame_ms

            # Trích xuất 543 điểm chuẩn để đưa vào model
            vec = extract_holistic(results)
            predictor.push_frame(vec)

            # Vẽ skeleton lên frame CHƯA LẬT (để toạ độ vẽ khớp với ảnh gốc)
            draw_landmarks_cv2(frame, results)

            # BƯỚC 2: LẬT KHUNG HÌNH (MIRROR) ĐỂ HIỂN THỊ CHO NGƯỜI DÙNG DỄ NHÌN
            frame = cv2.flip(frame, 1)

            # BƯỚC 3: VẼ GIAO DIỆN CHỮ LÊN KHUNG HÌNH (lúc này chữ không bị lật ngược)
            frame = draw_overlay(frame, predictor, len(predictor.buffer), topk)
            cv2.putText(frame, f"FPS: {fps:.0f}", (frame.shape[1] - 90, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

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
        landmarker.close()
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
