import os
import glob
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def extract_frames(video_path, output_dir, video_name, fps=30):
    """
    Trích xuất frame từ video sử dụng FFmpeg.
    """
    # Tạo thư mục output nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    # Định nghĩa cấu trúc tên file xuất ra (vd: D0001N_1.jpg, D0001N_2.jpg...)
    output_pattern = os.path.join(output_dir, f"{video_name}_%d.jpg")
    
    # Lệnh ffmpeg:
    # -y: Ghi đè file nếu đã tồn tại (không hỏi lại)
    # -i: Input video
    # -r 30: Đặt FPS (số khung hình trên giây) là 30
    # -q:v 2: Giữ chất lượng ảnh (JPEG quality, 1-31, số càng nhỏ càng đẹp)
    command = [
        "ffmpeg",
        "-y",
        "-hwaccel", "cuda",  # Kích hoạt tăng tốc phần cứng bằng GPU Nvidia
        "-i", video_path,
        "-r", str(fps),
        "-q:v", "2",
        output_pattern
    ]
    
    try:
        # Chạy lệnh, ẩn log chi tiết bằng cách chuyển stdout và stderr vào PIPE
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[LỖI] Không thể xử lý video {video_path}: {e.stderr.decode('utf-8')}")
        return False

# ============================================================================
# CẤU HÌNH SỐ LƯỢNG VIDEO CẦN TRÍCH XUẤT
# ============================================================================
MAX_VIDEOS = None  # Đổi None thành số nguyên (ví dụ: 10, 50) để giới hạn số video trích xuất. None = tất cả

def main():
    dataset_dir = os.path.join("dataset", "Vietnamese")
    data_dir = os.path.join("data", "Vietnamese", "frames")
    
    # Quét tất cả các file mp4 trong dataset/Vietnamese
    video_files = glob.glob(os.path.join(dataset_dir, "*.mp4"))
    
    if not video_files:
        print(f"Không tìm thấy file .mp4 nào trong thư mục {dataset_dir}")
        return
        
    # Áp dụng giới hạn số lượng video nếu được thiết lập
    if MAX_VIDEOS is not None:
        video_files = video_files[:MAX_VIDEOS]
        
    print(f"Tìm thấy {len(video_files)} video để xử lý. Bắt đầu trích xuất frames ở tốc độ 30 FPS...")
    
    # Hàm đóng gói để truyền vào luồng xử lý
    def process_video(video_path):
        video_name = Path(video_path).stem  # Lấy tên file không có đuôi (VD: D0001N)
        output_dir = os.path.join(data_dir, video_name)
        extract_frames(video_path, output_dir, video_name, fps=30)
        
    # Xử lý song song bằng đa luồng (Multi-threading) để tiết kiệm thời gian
    # Số luồng tự động dựa trên số lõi CPU của máy
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Sử dụng tqdm để hiển thị thanh tiến trình (progress bar)
        list(tqdm(executor.map(process_video, video_files), total=len(video_files), desc="Tiến trình cắt frame"))
        
    print(f"\nĐã hoàn tất! Các frame được lưu tại thư mục: {data_dir}")

if __name__ == "__main__":
    main()
