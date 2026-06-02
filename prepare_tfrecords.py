import os
import glob
import csv
import json
import torch
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
        
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Bỏ đuôi .mp4 để so sánh cho dễ (VD: D0001N.mp4 -> D0001N)
            video_name = row['VIDEO'].replace('.mp4', '')
            video_to_label[video_name] = row['LABEL']
            
    # 2. Quét các file .pt đã trích xuất MediaPipe
    pt_files = glob.glob(os.path.join(pt_dir, "*.pt"))
    if not pt_files:
        print(f"Không tìm thấy file .pt nào trong {pt_dir}")
        return
        
    print(f"Tìm thấy {len(pt_files)} file .pt để ghép.")
    
    # 3. Lọc danh sách và cấp mã ID cho các Label có tồn tại
    valid_videos = []
    present_labels = set()
    
    for pt_file in pt_files:
        video_name = Path(pt_file).stem.replace("_mediapipe", "")
        if video_name in video_to_label:
            label_str = video_to_label[video_name]
            valid_videos.append((pt_file, label_str))
            present_labels.add(label_str)
        else:
            print(f"[Cảnh báo] Video {video_name} không có trong file label CSV. Đã bỏ qua!")
            
    # Sắp xếp theo thứ tự A-Z và gắn số thứ tự từ 0 đến N-1
    sorted_labels = sorted(list(present_labels))
    label_to_id = {label: idx for idx, label in enumerate(sorted_labels)}
    
    # Lưu lại mapping để sau này chạy ứng dụng thực tế còn biết dự đoán số mấy là chữ gì
    map_file = os.path.join(out_dir, 'label_map.json')
    with open(map_file, 'w', encoding='utf-8') as f:
        json.dump(label_to_id, f, ensure_ascii=False, indent=4)
        
    print(f"Đã lập bản đồ cho {len(label_to_id)} nhãn (classes).")
    
    # Cập nhật ID cho valid_videos
    dataset = [(pt_file, label_to_id[label_str]) for pt_file, label_str in valid_videos]
    
    # 4. Trộn ngẫu nhiên và chia thành 5 folds (để Cross Validation)
    np.random.seed(42)
    np.random.shuffle(dataset)
    
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
            for pt_file, label_id in tqdm(fold_data, desc=f"Writing fold {fold_idx}"):
                # Đọc ma trận Tensor từ file .pt
                # Lưu ý: weight_onlys=True để tránh cảnh báo an toàn từ PyTorch bản mới
                tensor = torch.load(pt_file, weights_only=False)
                
                # Chuyển về float32 để khớp tuyệt đối với `decode_tfrec`
                np_array = tensor.numpy().astype(np.float32)
                
                # Chuyển thành Features
                feature = {
                    'coordinates': bytes_feature(np_array.tobytes()),
                    'sign': int64_feature(label_id)
                }
                
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
                
        print(f"Đã lưu: {tfrec_name} ({count} videos)")
        
    print("\n[HOÀN TẤT THÀNH CÔNG] Dữ liệu đã sẵn sàng để train!")
    print(f"⚠️ QUAN TRỌNG: Bạn BẮT BUỘC phải sửa biến NUM_CLASSES = {len(label_to_id)} trong file train.py trước khi chạy train.py nhé!")

if __name__ == "__main__":
    main()
