import os
import glob
import csv
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

def bytes_feature(value):
    """Trả về bytes_list từ kiểu chuỗi/byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
    """Trả về int64_list từ kiểu int/bool."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def main():
    pt_dir = os.path.join("data", "Vietnamese", "mediapipe")
    csv_path = os.path.join("dataset", "Vietnamese", "label", "label_with_folders.csv")
    out_dir = os.path.join("data", "Vietnamese", "TFRecord")
    
    # Tạo thư mục đầu ra
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Đọc file CSV và tạo từ điển tra cứu Video -> Nhãn (text)
    video_to_label = {}
    if not os.path.exists(csv_path):
        print(f"Lỗi: Không tìm thấy file CSV tại {csv_path}")
        return
        
    # Dùng utf-8-sig để tránh lỗi BOM (Byte Order Mark) nếu CSV lưu từ Excel
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Tự động lấy tên file bỏ đuôi mở rộng (tránh phân biệt .mp4 hay .MP4)
            video_name = Path(row['VIDEO']).stem
            video_to_label[video_name] = row['LABEL']
            
    # 2. Quét các file .npy đã trích xuất MediaPipe
    npy_files = glob.glob(os.path.join(pt_dir, "*.npy"))
    if not npy_files:
        print(f"Không tìm thấy file .npy nào trong {pt_dir}")
        return

    print(f"Tìm thấy {len(npy_files)} file .npy để ghép.")

    # 3. Ghép .npy với nhãn từ CSV
    #    Nhiều video khác nhau có thể cùng 1 nhãn (ví dụ D0001B và D0023B cùng là "địa chỉ (bắc)")
    #    → Đếm số nhãn UNIQUE trong các video có .npy mới là NUM_CLASSES đúng
    valid_videos  = []
    present_labels = set()

    for npy_file in npy_files:
        video_name = Path(npy_file).stem.replace("_mediapipe", "")
        if video_name in video_to_label:
            label_str = video_to_label[video_name]
            valid_videos.append((npy_file, label_str))
            present_labels.add(label_str)          # ← chỉ thêm nhãn mới nếu chưa có
        else:
            print(f"[Cảnh báo] Video {video_name} không có trong file label CSV. Đã bỏ qua!")

    # Sắp xếp A-Z để ID ổn định, sau đó gắn số thứ tự 0..N-1
    sorted_labels = sorted(present_labels)
    label_to_id   = {label: idx for idx, label in enumerate(sorted_labels)}

    print(f"Số nhãn UNIQUE từ {len(valid_videos)} video có .npy: {len(label_to_id)} classes")

    # Lưu label_map.json
    map_file = os.path.join(out_dir, "label_map.json")
    with open(map_file, "w", encoding="utf-8") as f:
        json.dump(label_to_id, f, ensure_ascii=False, indent=4)

    # Cập nhật ID cho dataset
    dataset = [(pt_file, label_to_id[label_str]) for pt_file, label_str in valid_videos]
    
    # 4. Trộn ngẫu nhiên và chia thành 5 folds (để Cross Validation)
    import random
    random.seed(42)
    random.shuffle(dataset)
    
    n_splits = 5
    # Cắt mảng thành n phần đều nhau
    folds = [dataset[i::n_splits] for i in range(n_splits)]
    
    # 5. Ghi ra TFRecord theo đúng chuẩn của Kaggle
    for fold_idx, fold_data in enumerate(folds):
        count = len(fold_data)
        if count == 0: continue
            
        # Tên file rất quan trọng: train.py dùng hàm count_data_items chứa biểu thức chính quy (Regex)
        # r"-([0-9]*)\." để đếm số lượng dựa trên TÊN FILE. VD: fold_0-110.tfrecords
        tfrec_name = os.path.join(out_dir, f"fold_{fold_idx}-{count}.tfrecords")
        
        with tf.io.TFRecordWriter(tfrec_name) as writer:
            for npy_file, label_id in tqdm(fold_data, desc=f"Writing fold {fold_idx}"):
                # Đọc ma trận numpy từ file .npy
                np_array = np.load(npy_file).astype(np.float32)
                np_array = np.ascontiguousarray(np_array)
                
                # Chuyển thành Features
                feature = {
                    'coordinates': bytes_feature(np_array.tobytes()),
                    'sign': int64_feature(label_id)
                }
                
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
                
        print(f"Đã lưu: {tfrec_name} ({count} videos)")
        
    print("\n✅ [HOÀN TẤT] Dữ liệu đã sẵn sàng để train!")
    print(f"   Số lượng classes: {len(label_to_id)} → train.py sẽ tự động đọc từ label_map.json")

if __name__ == "__main__":
    main()
