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
except ImportError:
    print("[ERROR] Thiếu mediapipe. Chạy: pip install mediapipe")
    sys.exit(1)

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


# ─── MEDIAPIPE SETUP ──────────────────────────────────────────────────────────
def init_mediapipe():
    """Khởi tạo MediaPipe Holistic."""
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    return holistic, mp_holistic, mp_drawing, mp_drawing_styles


def extract_holistic(results) -> np.ndarray:
    """
    Trích xuất vector (543, 3) từ kết quả MediaPipe Holistic.
    Cấu trúc y hệt mảng 543 điểm chuẩn:
      0-467: Face
      468-488: Left Hand
      489-521: Pose
      522-542: Right Hand
    Nếu không detect được → NaN.
    """
    frame_data = np.full((543, 3), np.nan, dtype=np.float32)
    
    if results.face_landmarks:
        for i, lm in enumerate(results.face_landmarks.landmark):
            if i < 468: 
                frame_data[i] = [lm.x, lm.y, lm.z]
                
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            frame_data[468 + i] = [lm.x, lm.y, lm.z]
            
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            frame_data[489 + i] = [lm.x, lm.y, lm.z]
            
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            frame_data[522 + i] = [lm.x, lm.y, lm.z]

    return frame_data


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
            
        model_path = candidates[0]
        print(f"\n[INFO] Đã chọn model: {model_path}")

    if not os.path.exists(model_path):
        print(f"[ERROR] File không tồn tại: {model_path}")
        sys.exit(1)

    # Import kiến trúc từ train.py để rebuild rồi load weights
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from train import get_model

    # Tạo model khớp với lúc train (dim=192, MAX_LEN=384)
    model = get_model(max_len=MAX_LEN, dropout_step=0, dim=192)
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
                 topk: int = 3, slide: int = 15):
        self.model          = model
        self.label_map      = label_map
        self.label_original = label_original
        self.topk           = topk
        self.slide          = slide     # số frame xóa sau mỗi predict (slide = 15 để predict nhanh)

        self.preprocess = Preprocess(max_len=MAX_LEN)
        self.buffer     = []            # list of ndarray (543, 3)

        self.last_label      = ""
        self.last_original   = ""
        self.last_confidence = 0.0
        self.last_topk       = []
        self.is_detecting    = False
        self.hand_detected   = False

    def push_frame(self, vec_543: np.ndarray) -> bool:
        # Check xem có tay trong khung hình không
        lhand = vec_543[468:489]
        rhand = vec_543[522:543]
        has_hand = not (np.isnan(lhand).all() and np.isnan(rhand).all())
        
        self.hand_detected = has_hand

        if has_hand:
            self.is_detecting = True
            self.buffer.append(vec_543)

        # Đủ BUFFER_LEN (vd 60 frame) → predict
        if len(self.buffer) >= BUFFER_LEN:
            self._predict()
            # Sliding: xóa slide frame đầu
            self.buffer = self.buffer[self.slide:]
            return True

        return False

    def _predict(self):
        seq = np.stack(self.buffer[:BUFFER_LEN], axis=0)   # (60, 543, 3)
        seq_tf = tf.constant(seq, dtype=tf.float32)

        # Preprocess → (1, 60, 708)
        features = self.preprocess(seq_tf)
        features = tf.cast(features, tf.float32)

        # Padding lên đủ MAX_LEN (384) để khớp với đầu vào model
        T = tf.shape(features)[1]
        if T < MAX_LEN:
            pad = tf.fill([1, MAX_LEN - T, CHANNELS], PAD)
            features = tf.concat([features, pad], axis=1)

        # Inference
        logits = self.model(features, training=False)
        probs  = tf.nn.softmax(logits[0]).numpy()

        # Top-K
        top_idx   = np.argsort(probs)[::-1][:self.topk]
        self.last_topk = [
            (self.label_map.get(i, f"class_{i}"),
             self.label_original.get(i, f"class_{i}"),
             float(probs[i]))
            for i in top_idx
        ]

        best_idx              = top_idx[0]
        self.last_label       = self.label_map.get(best_idx, f"class_{best_idx}")
        self.last_original    = self.label_original.get(best_idx, self.last_label)
        self.last_confidence  = float(probs[best_idx])

    def reset(self):
        self.buffer          = []
        self.last_label      = ""
        self.last_original   = ""
        self.last_confidence = 0.0
        self.last_topk       = []
        self.is_detecting    = False


# ─── DRAW OVERLAY ─────────────────────────────────────────────────────────────

def draw_overlay(frame: np.ndarray, predictor: SignPredictor,
                 buffer_size: int, topk: int):
    h, w, _ = frame.shape
    panel_w = 300
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (15, 15, 15), -1)
    alpha = 0.85
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.putText(frame, "SignBridge AI", (10, 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    cv2.line(frame, (10, 50), (panel_w - 10, 50), (80, 80, 80), 1)

    y = 80
    cv2.putText(frame, "STATUS:", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    y += 25

    if predictor.hand_detected:
        cv2.putText(frame, "DETECTING", (10, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
    else:
        cv2.putText(frame, "WAITING...", (10, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 140, 255), 1)
    y += 35

    bar_w = panel_w - 20
    fill_w = int((buffer_size / BUFFER_LEN) * bar_w)
    cv2.rectangle(frame, (10, y), (10 + bar_w, y + 10), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, y), (10 + fill_w, y + 10), (0, 255, 0), -1)
    y += 40

    cv2.putText(frame, "PREDICTION:", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    y += 35

    if predictor.last_label:
        conf_pct = predictor.last_confidence * 100
        if conf_pct > 70: label_color = (0, 255, 0)
        elif conf_pct > 40: label_color = (0, 200, 255)
        else: label_color = (0, 0, 255)

        label_display = predictor.last_original[:15]
        cv2.putText(frame, label_display, (10, y),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, label_color, 2)
        y += 45
        cv2.putText(frame, f"({predictor.last_label})  {conf_pct:.1f}%",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1)
    else:
        cv2.putText(frame, "---", (10, y),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (100, 100, 100), 2)

    y += 35
    cv2.line(frame, (10, y), (panel_w - 10, y), (80, 80, 80), 1)
    y += 15

    if predictor.last_topk:
        cv2.putText(frame, f"TOP {topk}:", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
        y += 22

        for rank, (slug, orig, conf) in enumerate(predictor.last_topk[:topk]):
            bar_fill  = int((panel_w - 20) * conf)
            bar_color = (0, 140, 255) if rank == 0 else (60, 100, 160)

            cv2.rectangle(frame, (10, y), (10 + bar_fill, y + 18), bar_color, -1)
            cv2.rectangle(frame, (10, y), (panel_w - 10, y + 18), (80, 80, 80), 1)

            label_txt = f"{rank+1}. {orig}  {conf*100:.0f}%"
            cv2.putText(frame, label_txt, (14, y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y += 25

    guides = ["[R] Reset buffer", "[Q] Thoat", "[S] Chup man hinh"]
    for i, g in enumerate(guides):
        cv2.putText(frame, g, (10, h - 60 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 130), 1)

    # ─── CHỮ DỰ ĐOÁN SIÊU TO Ở GÓC DƯỚI BÊN PHẢI ───────────────────────────────
    if predictor.last_label and predictor.last_confidence > 0.3:
        conf_pct = predictor.last_confidence * 100
        text = predictor.last_original
        conf_text = f"{conf_pct:.0f}%"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.8
        thickness = 3
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, 0.9, 2)
        
        margin_x = 30
        margin_y = 30
        x_pos = w - text_w - margin_x
        y_pos = h - margin_y - conf_h - 10

        # Nền mờ đằng sau chữ để dễ đọc
        bg_x1 = max(0, x_pos - 15)
        bg_y1 = max(0, y_pos - text_h - 15)
        bg_x2 = w
        bg_y2 = h
        bg_overlay = frame.copy()
        cv2.rectangle(bg_overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        frame = cv2.addWeighted(bg_overlay, 0.6, frame, 0.4, 0)

        # Màu sắc theo mức độ tự tin
        if conf_pct > 70:
            label_color = (0, 255, 0)      # Xanh lá
        elif conf_pct > 50:
            label_color = (0, 200, 255)    # Vàng cam
        else:
            label_color = (0, 140, 255)    # Cam

        # Vẽ tên từ vựng
        cv2.putText(frame, text, (x_pos, y_pos), font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(frame, text, (x_pos, y_pos), font, font_scale, label_color, thickness)
        # Vẽ % phía dưới từ vựng
        cv2.putText(frame, conf_text, (w - conf_w - margin_x, h - margin_y),
                    font, 0.9, label_color, 2)

    return frame


# ─── MAIN CAMERA LOOP ─────────────────────────────────────────────────────────

def run_camera(model_path: str = "", camera_idx: int = 0,
               topk: int = 3, dataset_dir: str = DEFAULT_DATASET_DIR):
    label_map, label_original = load_label_map(dataset_dir)
    model = load_model(model_path)
    holistic, mp_holistic, mp_drawing, mp_drawing_styles = init_mediapipe()
    predictor = SignPredictor(model, label_map, label_original, topk=topk)

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Không mở được camera {camera_idx}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n[INFO] Camera đang chạy.")
    
    os.makedirs("captures", exist_ok=True)
    frame_count = 0
    fps_time    = time.time()
    fps         = 0.0

    with holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_time)
                fps_time = time.time()

            # BƯỚC 1: XỬ LÝ KHUNG HÌNH GỐC (CHƯA LẬT) ĐỂ KHÔNG BỊ NGƯỢC TAY!
            # Mô hình train trên tay phải/trái gốc, nếu lật trước khi đưa vào MediaPipe, 
            # tay phải của bạn sẽ bị nhận diện nhầm thành tay trái (Mirror Effect)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            # Trích xuất 543 điểm chuẩn để đưa vào model
            vec = extract_holistic(results)
            predictor.push_frame(vec)

            # Vẽ skeleton lên frame CHƯA LẬT
            if results.face_landmarks:
                mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                    mp_drawing_styles.get_default_face_mesh_tesselation_style())
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    mp_drawing_styles.get_default_pose_landmarks_style())
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style())
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style())

            # BƯỚC 2: LẬT KHUNG HÌNH (MIRROR) ĐỂ HIỂN THỊ CHO NGƯỜI DÙNG DỄ NHÌN
            # Lúc này ảnh và skeleton sẽ lật lại như soi gương
            frame = cv2.flip(frame, 1)

            # BƯỚC 3: VẼ GIAO DIỆN CHỮ LÊN KHUNG HÌNH (Lúc này chữ không bị lật ngược)
            frame = draw_overlay(frame, predictor, len(predictor.buffer), topk)
            cv2.putText(frame, f"FPS: {fps:.0f}", (frame.shape[1] - 90, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("SignBridge AI", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                predictor.reset()
            elif key == ord('s'):
                fname = f"captures/capture_{int(time.time())}.png"
                cv2.imwrite(fname, frame)

    cap.release()
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
