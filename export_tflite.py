import os
import sys
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

DEFAULT_MODEL_DIR = "./output"
MAX_LEN = 384

# Phải khớp đủng với train.py: POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE = 118 điểm
# CHANNELS = 6 * NUM_NODES = 6 * 118 = 708
NUM_NODES = 118
CHANNELS = 6 * NUM_NODES  # = 708

def load_model_path():
    """Hỏi người dùng chọn thư mục chứa model từ output/ và trả về đường dẫn file .weights.h5"""
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

    # Nhẫu ưu tiên file 'best' hơn file 'last'
    best_files = [c for c in candidates if "best" in os.path.basename(c)]
    model_path = best_files[0] if best_files else candidates[0]
    print(f"\n[INFO] Đã chọn model: {model_path}")
    
    # Tự động nhận diện dim từ comment trong tên file (vd: islr-fp16-192-... hoặc islr-fp16-384-...)
    dim = 192  # giá trị mặc định
    basename = os.path.basename(model_path)
    for candidate_dim in [384, 256, 192]:
        if f"-{candidate_dim}-" in basename:
            dim = candidate_dim
            break
    print(f"[INFO] Tự động nhận diện kiến trúc: dim={dim}")
    
    return model_path, selected_folder, dim

def export_tflite(model: tf.keras.Model, output_path: str, quantize: str = "float16"):
    print(f"\n[INFO] Đang nén mô hình sang TFLite (quantize={quantize})...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantize == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"[OK]   Đã xuất TFLite thành công: {output_path} ({size_mb:.2f} MB)")

def verify_tflite(tflite_path: str):
    print("[INFO] Đang kiểm tra file .tflite với dummy input...")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    dummy_input = np.zeros((1, MAX_LEN, CHANNELS), dtype=input_details[0]["dtype"])
    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    
    probs = tf.nn.softmax(output[0].astype(np.float32)).numpy()
    print(f"[OK]   Dummy Softmax: {probs.tolist()}")
    print("[OK]   TFLite sẵn sàng → chạy: python camera_demo_tflite.py\n")

def main():
    model_path, selected_folder, dim = load_model_path()
    
    # Import kiến trúc từ train.py để rebuild rồi load weights
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from train import get_model
    
    # Tạo model khớp với lúc train (dim tự detect, MAX_LEN=384)
    print(f"[INFO] Đang khôi phục kiến trúc mô hình (dim={dim}) và nạp trọng số...")
    model = get_model(max_len=MAX_LEN, dropout_step=0, dim=dim)
    model.load_weights(model_path)
    print(f"[INFO] Model ready. Input={model.input_shape}, Output={model.output_shape}")
    
    # Xuất TFLite — buộc phải reset về float32 trước khi convert
    # (TFLite converter không hỗ trợ mixed_bfloat16 / mixed_float16)
    try:
        mixed_precision.set_global_policy("float32")
    except:
        pass
        
    tflite_name = os.path.basename(model_path).replace(".weights.h5", "-float16.tflite")
    tflite_path = os.path.join(DEFAULT_MODEL_DIR, selected_folder, tflite_name)
    
    try:
        export_tflite(model, tflite_path, quantize="float16")
        verify_tflite(tflite_path)
    except Exception as e:
        print(f"[ERROR] Xuất TFLite thất bại: {e}")

if __name__ == "__main__":
    main()
