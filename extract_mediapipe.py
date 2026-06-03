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

MODEL_PATH = "holistic_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Đang tải file mô hình MediaPipe ({MODEL_PATH})...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Tải xong!")

def create_landmarker_options():
    """Tạo options cho HolisticLandmarker (dùng lại cho mỗi video)."""
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    return vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
        output_face_blendshapes=False
    )

def extract_landmarks(video_path, options):
    """
    Đọc video và trích xuất ma trận (N, 543, 3) chứa toạ độ x, y, z.
    Tạo một landmarker mới cho mỗi video để tránh lỗi timestamp.
    """
    frames_landmarks = []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    try:
        # Tạo landmarker mới cho mỗi video (timestamp phải tăng đơn điệu
        # trong suốt vòng đời của 1 landmarker, nên không thể dùng chung)
        with vision.HolisticLandmarker.create_from_options(options) as landmarker:
            # Lấy FPS để tính timestamp cho Tasks API
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 30.0
            frame_ms = int(1000 / fps)
            current_time_ms = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Đổi hệ màu BGR sang RGB cho MediaPipe
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                
                # Xử lý trích xuất điểm ảnh (chế độ VIDEO bắt buộc truyền timestamp)
                results = landmarker.detect_for_video(mp_image, current_time_ms)
                current_time_ms += frame_ms
                
                # Khởi tạo mảng NaN cho 543 điểm: Face (468), LHand (21), Pose (33), RHand (21)
                frame_data = np.full((543, 3), np.nan, dtype=np.float32)
                
                # 1. Khuôn mặt (Face - 0 đến 467)
                if results.face_landmarks:
                    for i, lm in enumerate(results.face_landmarks):
                        if i < 468: 
                            frame_data[i] = [lm.x, lm.y, lm.z]
                            
                # 2. Tay trái (Left Hand - 468 đến 488)
                if results.left_hand_landmarks:
                    for i, lm in enumerate(results.left_hand_landmarks):
                        frame_data[468 + i] = [lm.x, lm.y, lm.z]
                        
                # 3. Tư thế (Pose - 489 đến 521)
                if results.pose_landmarks:
                    for i, lm in enumerate(results.pose_landmarks):
                        frame_data[489 + i] = [lm.x, lm.y, lm.z]
                        
                # 4. Tay phải (Right Hand - 522 đến 542)
                if results.right_hand_landmarks:
                    for i, lm in enumerate(results.right_hand_landmarks):
                        frame_data[522 + i] = [lm.x, lm.y, lm.z]
                        
                frames_landmarks.append(frame_data)
    finally:
        cap.release()
    
    if len(frames_landmarks) == 0:
        return None
        
    return np.array(frames_landmarks, dtype=np.float32)

def main():
    download_model()
    
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
        
    print(f"\n🎬 Bắt đầu trích xuất MediaPipe cho {len(video_files)} video...")
    
    # Tạo options 1 lần, nhưng landmarker sẽ được tạo mới cho từng video
    options = create_landmarker_options()
    
    for video_path in tqdm(video_files, desc="Đang trích xuất MediaPipe"):
        video_name = Path(video_path).stem
        output_path = os.path.join(output_dir, f"{video_name}_mediapipe.npy")
        
        # Bỏ qua nếu đã trích xuất trước đó (Resume)
        if os.path.exists(output_path):
            continue
            
        landmarks_data = extract_landmarks(video_path, options)
        
        if landmarks_data is not None:
            np.save(output_path, landmarks_data)

if __name__ == "__main__":
    main()
