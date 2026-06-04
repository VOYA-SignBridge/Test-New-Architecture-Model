import os
import glob
import cv2
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# CẤU HÌNH: Số lượng video và bộ lọc
# ============================================================================
MAX_VIDEOS = 5          # None = tất cả video; số nguyên = số video muốn xử lý
FILTER_SUFFIX = "N"       # Lọc video có chữ này ở cuối tên (ví dụ: "N" -> D0001N)
                           # Đặt = None để bỏ qua bộ lọc

# Cấu hình Mô hình
HOLISTIC_MODEL_PATH = "holistic_landmarker.task"
HOLISTIC_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"

HANDS_MODEL_PATH = "hand_landmarker.task"
HANDS_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# ============================================================================
# Bố cục mảng 543 điểm (phải khớp với prepare_tfrecords.py và train.py)
# ============================================================================
# [0   : 467]  Face Mesh      (468 điểm)
# [468 : 488]  Left Hand      (21 điểm)  <- Tay TRÁI của NGƯỜI DÙNG
# [489 : 521]  Pose           (33 điểm)
# [522 : 542]  Right Hand     (21 điểm)  <- Tay PHẢI của NGƯỜI DÙNG
FACE_START,  FACE_END   = 0,   468
LHAND_START, LHAND_END  = 468, 489
POSE_START,  POSE_END   = 489, 522
RHAND_START, RHAND_END  = 522, 543

def download_models():
    """Tải xuống cả 2 mô hình cần thiết cho kiến trúc Hybrid."""
    if not os.path.exists(HOLISTIC_MODEL_PATH):
        print(f"Đang tải file mô hình Holistic ({HOLISTIC_MODEL_PATH})...")
        urllib.request.urlretrieve(HOLISTIC_URL, HOLISTIC_MODEL_PATH)
        
    if not os.path.exists(HANDS_MODEL_PATH):
        print(f"Đang tải file mô hình Hand ({HANDS_MODEL_PATH})...")
        urllib.request.urlretrieve(HANDS_URL, HANDS_MODEL_PATH)
        
    print("Đã kiểm tra và tải đủ mô hình!")

def create_landmarker_options():
    """Tạo options cho cả Holistic và Hand Landmarker."""
    # 1. Options cho Holistic (Lấy Face và Pose; tay của Holistic dùng làm Fallback)
    holistic_base = python.BaseOptions(model_asset_path=HOLISTIC_MODEL_PATH)
    holistic_options = vision.HolisticLandmarkerOptions(
        base_options=holistic_base,
        running_mode=vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
        output_face_blendshapes=False
    )
    
    # 2. Options cho Hands (Nguồn tay chính - chất lượng cao hơn Holistic)
    hands_base = python.BaseOptions(model_asset_path=HANDS_MODEL_PATH)
    hands_options = vision.HandLandmarkerOptions(
        base_options=hands_base,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    return holistic_options, hands_options

# ─── EMA TEMPORAL HAND TRACKER ───────────────────────────────────────────────
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

def extract_landmarks(video_path, holistic_opts, hands_opts):
    """
    Trích xuất Hybrid vector (N, 543, 3) từ video.
    Layout kết quả:
      [0:468]   = Face (từ Holistic)
      [468:489] = Left Hand  (Tay TRÁI của người dùng)
      [489:522] = Pose (từ Holistic)
      [522:543] = Right Hand (Tay PHẢI của người dùng)

    Chiều tay xác định bằng Holistic-Anchor Assignment + EMA Temporal Tracker.
    Không dùng category_name của HandLandmarker (đã bỏ từ phiên bản này).
    """
    frames_landmarks = []
    tracker = HandTracker()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        
        # Dùng float để tránh drift timestamp khi fps không chia đều (vd: 29.97)
        frame_duration_ms = 1000.0 / fps

        # Khởi tạo đồng thời 2 landmarker — mỗi video cần instance riêng
        # vì timestamp phải tăng đơn điệu trong suốt vòng đời của 1 instance
        with vision.HolisticLandmarker.create_from_options(holistic_opts) as holistic_model, \
             vision.HandLandmarker.create_from_options(hands_opts) as hand_model:
            
            frame_idx = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Tính timestamp chính xác từ frame_idx thay vì cộng dồn int
                current_time_ms = int(frame_idx * frame_duration_ms)
                frame_idx += 1
                    
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                
                # Chạy 2 mô hình trên cùng 1 khung hình
                res_holistic = holistic_model.detect_for_video(mp_image, current_time_ms)
                res_hands    = hand_model.detect_for_video(mp_image, current_time_ms)

                # Trích xuất 543 điểm (logic giống hệt camera_demo.py)
                frame_data = extract_hybrid(res_holistic, res_hands, tracker)
                tracker.update(frame_data)
                        
                frames_landmarks.append(frame_data)
                
    finally:
        cap.release()
    
    if len(frames_landmarks) == 0:
        return None
        
    return np.array(frames_landmarks, dtype=np.float32)

def main():
    download_models()
    
    dataset_dir = os.path.join("dataset", "Vietnamese")
    output_dir = os.path.join("data", "Vietnamese", "mediapipe")
    
    os.makedirs(output_dir, exist_ok=True)
    video_files = glob.glob(os.path.join(dataset_dir, "*.mp4"))
    
    if not video_files:
        print(f"Không tìm thấy file .mp4 nào trong {dataset_dir}")
        return
    
    # Lọc video theo suffix nếu được chỉ định
    if FILTER_SUFFIX is not None:
        original_count = len(video_files)
        video_files = [v for v in video_files if Path(v).stem.endswith(FILTER_SUFFIX)]
        print(f"ℹ️  Bộ lọc: Chỉ xử lý video có chữ '{FILTER_SUFFIX}' ở cuối tên")
        print(f"   Từ {original_count} video → {len(video_files)} video")
    
    # Giới hạn số lượng video nếu được đặt
    if MAX_VIDEOS is not None and len(video_files) > MAX_VIDEOS:
        video_files = video_files[:MAX_VIDEOS]
        print(f"ℹ️  Giới hạn: Chỉ xử lý {MAX_VIDEOS} video đầu tiên")
        
    print(f"\n🎬 Bắt đầu trích xuất MediaPipe (Hybrid Pipeline) cho {len(video_files)} video...")
    print("   Nguồn chính: HandLandmarker (bàn tay) | Fallback: Holistic")
    print("   Face + Pose: Holistic\n")
    
    holistic_opts, hands_opts = create_landmarker_options()
    
    for video_path in tqdm(video_files, desc="Đang trích xuất (Hybrid)"):
        video_name = Path(video_path).stem
        output_path = os.path.join(output_dir, f"{video_name}_mediapipe.npy")
        
        # Bỏ qua nếu đã trích xuất trước đó (Resume)
        if os.path.exists(output_path):
            continue
            
        landmarks_data = extract_landmarks(video_path, holistic_opts, hands_opts)
        
        if landmarks_data is not None:
            np.save(output_path, landmarks_data)

if __name__ == "__main__":
    main()
