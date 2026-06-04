"""
ISLR 1st Place Solution - Local Training Script
Adapted from ISLR_1st_place_Hoyeol_Sohn.ipynb
(Reverted to original Kaggle parameters to match nb_extracted.txt)
"""

import os
import gc
import glob
import random
import datetime
import re
import numpy as np
import tensorflow as tf
import tensorflow.keras.mixed_precision as mixed_precision

try:
    import tensorflow_addons as tfa
    HAS_TFA = True
except ImportError:
    HAS_TFA = False
    print("[WARN] tensorflow-addons not found. Using Adam fallback.")

try:
    from tf_utils.schedules import OneCycleLR
    from tf_utils.callbacks import Snapshot, SWA
    from tf_utils.learners import FGM, AWP
    HAS_TF_UTILS = True
except ImportError:
    HAS_TF_UTILS = False
    print("[WARN] tf-utils not found. Using CosineDecay fallback.")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# Input : thư mục .tfrecords (tạo bằng prepare_tfrecords.py)
# Output: thư mục lưu model weights + logs
DATA_DIR   = "./data/Vietnamese/TFRecord"
OUTPUT_DIR = "./outputs"

# ============================================================================
# CẤU HÌNH: Số lượng video và bộ lọc
# ============================================================================
MAX_VIDEOS = 5          # None = tất cả video; số nguyên = số video muốn huấn luyện
FILTER_SUFFIX = None       # Lọc video có chữ này ở cuối tên (ví dụ: "N" -> D0001N)
                           # Đặt = None để bỏ qua bộ lọc

TRAIN_FILENAMES = glob.glob(os.path.join(DATA_DIR, "**/*.tfrecords"), recursive=True)

# Áp dụng bộ lọc suffix nếu được chỉ định
if FILTER_SUFFIX is not None:
    original_count = len(TRAIN_FILENAMES)
    TRAIN_FILENAMES = [f for f in TRAIN_FILENAMES if FILTER_SUFFIX in f]
    print(f"ℹ️  Bộ lọc: Chỉ xử lý video có chữ '{FILTER_SUFFIX}' trong tên")
    print(f"   Từ {original_count} file → {len(TRAIN_FILENAMES)} file")

# Giới hạn số lượng video nếu được đặt
if MAX_VIDEOS is not None:
    # Trích xuất danh sách video duy nhất từ tên file (fold_X-COUNT.tfrecords)
    video_counts = {}
    for fname in TRAIN_FILENAMES:
        # Tìm số lượng video trong file từ tên: fold_0-110 → 110
        match = re.search(r'-(\d+)\.tfrecords', fname)
        if match:
            count = int(match.group(1))
            fold = re.search(r'fold_(\d+)', fname)
            fold_idx = int(fold.group(1)) if fold else 0
            if fold_idx not in video_counts:
                video_counts[fold_idx] = 0
            video_counts[fold_idx] += count
    
    # Tính tổng số video hiện tại
    total_videos = sum(video_counts.values())
    
    if total_videos > MAX_VIDEOS:
        # Lọc để chỉ giữ MAX_VIDEOS video đầu tiên
        videos_kept = 0
        filtered_files = []
        for fname in sorted(TRAIN_FILENAMES):
            match = re.search(r'-(\d+)\.tfrecords', fname)
            if match:
                count = int(match.group(1))
                if videos_kept + count <= MAX_VIDEOS:
                    filtered_files.append(fname)
                    videos_kept += count
        
        TRAIN_FILENAMES = filtered_files
        print(f"ℹ️  Giới hạn: Chỉ xử lý {MAX_VIDEOS} video đầu tiên")
        print(f"   Từ {total_videos} video → {videos_kept} video")

# ─── HYPERPARAMETERS ──────────────────────────────────────────────────────────
ROWS_PER_FRAME = 543
MAX_LEN = 384
CROP_LEN = MAX_LEN
PAD = -100.

# Tự động đọc NUM_CLASSES từ label_map.json được tạo bởi prepare_tfrecords.py
_label_map_path = os.path.join(DATA_DIR, "label_map.json")
if os.path.exists(_label_map_path):
    import json as _json
    with open(_label_map_path, "r", encoding="utf-8") as _f:
        _label_data = _json.load(_f)
    NUM_CLASSES = len(_label_data)
    print(f"[INFO] Tự động phát hiện NUM_CLASSES = {NUM_CLASSES} (từ {_label_map_path})")
else:
    NUM_CLASSES = 4
    print(f"[WARN] Không tìm thấy {_label_map_path}. Dùng NUM_CLASSES mặc định = {NUM_CLASSES}")


# ─── LANDMARK GROUPS ──────────────────────────────────────────────────────────
NOSE   = [1, 2, 98, 327]
LNOSE  = [98]
RNOSE  = [327]

LIP = [
    0,
    61, 185, 40, 39, 37, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
LLIP = [84,181,91,146,61,185,40,39,37,87,178,88,95,78,191,80,81,82]
RLIP = [314,405,321,375,291,409,270,269,267,317,402,318,324,308,415,310,311,312]

POSE  = [500, 502, 504, 501, 503, 505, 512, 513]
LPOSE = [513, 505, 503, 501]
RPOSE = [512, 504, 502, 500]

REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133,
        246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362,
        466, 388, 387, 386, 385, 384, 398]

LHAND = np.arange(468, 489).tolist()
RHAND = np.arange(522, 543).tolist()

POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE #+POSE
NUM_NODES       = len(POINT_LANDMARKS)
CHANNELS        = 6 * NUM_NODES

print(f"NUM_NODES : {NUM_NODES}")
print(f"CHANNELS  : {CHANNELS}")
print(f"MAX_LEN   : {MAX_LEN}")

# ─── UTILITIES ────────────────────────────────────────────────────────────────
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename.split('/')[-1]).group(1)) for filename in filenames]
    return np.sum(n)

# ─── NAN HELPERS ──────────────────────────────────────────────────────────────
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

# ─── INTERPOLATION ────────────────────────────────────────────────────────────
def interp1d_(x, target_len, method='random'):
    length = tf.shape(x)[1]
    target_len = tf.maximum(1, target_len)
    if method == 'random':
        if tf.random.uniform(()) < 0.33:
            x = tf.image.resize(x, (target_len, tf.shape(x)[1]), 'bilinear')
        else:
            if tf.random.uniform(()) < 0.5:
                x = tf.image.resize(x, (target_len, tf.shape(x)[1]), 'bicubic')
            else:
                x = tf.image.resize(x, (target_len, tf.shape(x)[1]), 'nearest')
    else:
        x = tf.image.resize(x, (target_len, tf.shape(x)[1]), method)
    return x

# ─── PREPROCESS LAYER ─────────────────────────────────────────────────────────
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

        mean = tf_nan_mean(tf.gather(x, [17], axis=2), axis=[1,2], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5, x.dtype), mean)
        x    = tf.gather(x, self.point_landmarks, axis=2)
        std  = tf_nan_std(x, center=mean, axis=[1, 2], keepdims=True)

        x = (x - mean) / std

        if self.max_len is not None:
            x = x[:, :self.max_len]
        length = tf.shape(x)[1]
        x = x[..., :2]

        dx  = tf.cond(
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

# ─── AUGMENTATION ─────────────────────────────────────────────────────────────
def flip_lr(x):
    x, y, z = tf.unstack(x, axis=-1)
    x     = 1 - x
    new_x = tf.stack([x, y, z], -1)
    new_x = tf.transpose(new_x, [1, 0, 2])
    lhand = tf.gather(new_x, LHAND, axis=0)
    rhand = tf.gather(new_x, RHAND, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LHAND)[...,None], rhand)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(RHAND)[...,None], lhand)
    llip = tf.gather(new_x, LLIP, axis=0)
    rlip = tf.gather(new_x, RLIP, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LLIP)[...,None], rlip)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(RLIP)[...,None], llip)
    lpose = tf.gather(new_x, LPOSE, axis=0)
    rpose = tf.gather(new_x, RPOSE, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LPOSE)[...,None], rpose)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(RPOSE)[...,None], lpose)
    leye = tf.gather(new_x, LEYE, axis=0)
    reye = tf.gather(new_x, REYE, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LEYE)[...,None], reye)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(REYE)[...,None], leye)
    lnose = tf.gather(new_x, LNOSE, axis=0)
    rnose = tf.gather(new_x, RNOSE, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LNOSE)[...,None], rnose)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(RNOSE)[...,None], lnose)
    new_x = tf.transpose(new_x, [1, 0, 2])
    return new_x

def resample(x, rate=(0.8, 1.2)):
    rate     = tf.random.uniform((), rate[0], rate[1])
    length   = tf.shape(x)[0]
    new_size = tf.cast(rate * tf.cast(length, tf.float32), tf.int32)
    return interp1d_(x, new_size)

def spatial_random_affine(xyz, scale=(.8,1.2), shear=(-.15,.15),
                          shift=(-.1,.1), degree=(-30,30)):
    center = tf.constant([0.5, 0.5])
    if scale is not None:
        s   = tf.random.uniform((), *scale)
        xyz = s * xyz
    if shear is not None:
        xy        = xyz[..., :2]; z = xyz[..., 2:]
        shear_x = shear_y = tf.random.uniform((), *shear)
        if tf.random.uniform(()) < 0.5:
            shear_x = 0.
        else:
            shear_y = 0.
        sm  = tf.identity([[1., shear_x], [shear_y, 1.]])
        xy  = xy @ sm
        center = center + [shear_y, shear_x]
        xyz = tf.concat([xy, z], axis=-1)
    if degree is not None:
        xy  = xyz[..., :2]; z = xyz[..., 2:]
        xy -= center
        deg = tf.random.uniform((), *degree)
        rad = deg / 180 * np.pi
        c   = tf.math.cos(rad); s = tf.math.sin(rad)
        rm  = tf.identity([[c, s], [-s, c]])
        xy  = xy @ rm + center
        xyz = tf.concat([xy, z], axis=-1)
    if shift is not None:
        sh  = tf.random.uniform((), *shift)
        xyz = xyz + sh
    return xyz

def temporal_crop(x, length=MAX_LEN):
    l      = tf.shape(x)[0]
    offset = tf.random.uniform((), 0, tf.clip_by_value(l - length, 1, length), dtype=tf.int32)
    return x[offset:offset + length]

def temporal_mask(x, size=(.2,.4), mask_value=float("nan")):
    l           = tf.shape(x)[0]
    mask_size   = tf.cast(tf.cast(l, tf.float32) * tf.random.uniform((), *size), tf.int32)
    mask_offset = tf.random.uniform((), 0, tf.clip_by_value(l - mask_size, 1, l), dtype=tf.int32)
    x = tf.tensor_scatter_nd_update(
        x,
        tf.range(mask_offset, mask_offset + mask_size)[..., None],
        tf.fill([mask_size, ROWS_PER_FRAME, 3], mask_value)
    )
    return x

def spatial_mask(x, size=(.2,.4), mask_value=float("nan")):
    mox  = tf.random.uniform(())
    moy  = tf.random.uniform(())
    ms   = tf.random.uniform((), *size)
    mask = ((mox < x[..., 0]) & (x[..., 0] < mox + ms) &
            (moy < x[..., 1]) & (x[..., 1] < moy + ms))
    return tf.where(mask[..., None], mask_value, x)

def augment_fn(x, always=False, max_len=None):
    if tf.random.uniform(()) < 0.8 or always:
        x = resample(x, (0.5, 1.5))
    if tf.random.uniform(()) < 0.5 or always:
        x = flip_lr(x)
    if max_len is not None:
        x = temporal_crop(x, max_len)
    if tf.random.uniform(()) < 0.75 or always:
        x = spatial_random_affine(x)
    if tf.random.uniform(()) < 0.5 or always:
        x = temporal_mask(x)
    if tf.random.uniform(()) < 0.5 or always:
        x = spatial_mask(x)
    return x

# ─── DATASET ──────────────────────────────────────────────────────────────────
def decode_tfrec(record_bytes):
    features = tf.io.parse_single_example(record_bytes, {
        "coordinates": tf.io.FixedLenFeature([], tf.string),
        "sign":        tf.io.FixedLenFeature([], tf.int64),
    })
    out = {}
    out["coordinates"] = tf.reshape(
        tf.io.decode_raw(features["coordinates"], tf.float32),
        (-1, ROWS_PER_FRAME, 3)
    )
    out["sign"] = features["sign"]
    return out

def filter_nans_tf(x, ref_point=POINT_LANDMARKS):
    mask = tf.math.logical_not(tf.reduce_all(tf.math.is_nan(tf.gather(x,ref_point,axis=1)), axis=[-2,-1]))
    x = tf.boolean_mask(x, mask, axis=0)
    return x

def preprocess(x, augment=False, max_len=MAX_LEN):
    coord = x["coordinates"]
    coord = filter_nans_tf(coord)
    if augment:
        coord = augment_fn(coord, max_len=max_len)
    coord = tf.ensure_shape(coord, (None, ROWS_PER_FRAME, 3))
    return (
        tf.cast(Preprocess(max_len=max_len)(coord)[0], tf.float32),
        tf.one_hot(x["sign"], NUM_CLASSES)
    )

def get_tfrec_dataset(tfrecords, batch_size=64, max_len=64,
                      drop_remainder=False, augment=False,
                      shuffle=False, repeat=False):
    ds = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(decode_tfrec, tf.data.AUTOTUNE)
    ds = ds.map(lambda x: preprocess(x, augment=augment, max_len=max_len),
                tf.data.AUTOTUNE)
    if repeat:
        ds = ds.repeat()
    if shuffle:
        ds = ds.shuffle(shuffle)
        opts = tf.data.Options()
        opts.experimental_deterministic = False
        ds = ds.with_options(opts)
    if batch_size:
        ds = ds.padded_batch(
            batch_size,
            padding_values=PAD,
            padded_shapes=([max_len, CHANNELS], [NUM_CLASSES]),
            drop_remainder=drop_remainder
        )
    return ds.prefetch(tf.data.AUTOTUNE)

# ─── MODEL COMPONENTS ─────────────────────────────────────────────────────────
class ECA(tf.keras.layers.Layer):
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.kernel_size      = kernel_size
        self.conv = tf.keras.layers.Conv1D(1, kernel_size=kernel_size,
                                           strides=1, padding="same", use_bias=False)

    def call(self, inputs, mask=None):
        nn = tf.keras.layers.GlobalAveragePooling1D()(inputs, mask=mask)
        nn = tf.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        nn = nn[:, None, :]
        return inputs * nn

class LateDropout(tf.keras.layers.Layer):
    def __init__(self, rate, noise_shape=None, start_step=0, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.rate             = rate
        self.start_step       = start_step
        self.dropout          = tf.keras.layers.Dropout(rate, noise_shape=noise_shape)

    def build(self, input_shape):
        super().build(input_shape)
        self._train_counter = tf.Variable(0, dtype="int64",
                                          aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                                          trainable=False)

    def call(self, inputs, training=False):
        x = tf.cond(
            self._train_counter < self.start_step,
            lambda: inputs,
            lambda: self.dropout(inputs, training=training)
        )
        if training:
            self._train_counter.assign_add(1)
        return x

class CausalDWConv1D(tf.keras.layers.Layer):
    def __init__(self, kernel_size=17, dilation_rate=1, use_bias=False,
                 depthwise_initializer="glorot_uniform", name="", **kwargs):
        super().__init__(name=name, **kwargs)
        self.causal_pad = tf.keras.layers.ZeroPadding1D(
            (dilation_rate * (kernel_size - 1), 0), name=name + "_pad")
        self.dw_conv = tf.keras.layers.DepthwiseConv1D(
            kernel_size, strides=1, dilation_rate=dilation_rate,
            padding="valid", use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            name=name + "_dwconv")
        self.supports_masking = True

    def call(self, inputs):
        return self.dw_conv(self.causal_pad(inputs))

def Conv1DBlock(channel_size, kernel_size, dilation_rate=1, drop_rate=0.0,
                expand_ratio=2, se_ratio=0.25, activation="swish", name=None):
    if name is None:
        name = str(tf.keras.backend.get_uid("mbblock"))

    def apply(inputs):
        channels_in     = tf.keras.backend.int_shape(inputs)[-1]
        channels_expand = channels_in * expand_ratio
        skip = inputs
        x = tf.keras.layers.Dense(channels_expand, use_bias=True,
                                   activation=activation,
                                   name=name + "_expand_conv")(inputs)
        x = CausalDWConv1D(kernel_size, dilation_rate=dilation_rate,
                            use_bias=False, name=name + "_dwconv")(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.95, name=name + "_bn")(x)
        x = ECA()(x)
        x = tf.keras.layers.Dense(channel_size, use_bias=True,
                                   name=name + "_project_conv")(x)
        if drop_rate > 0:
            x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1),
                                         name=name + "_drop")(x)
        if channels_in == channel_size:
            x = tf.keras.layers.add([x, skip], name=name + "_add")
        return x
    return apply

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, dim=256, num_heads=4, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.dim       = dim
        self.scale     = dim ** -0.5
        self.num_heads = num_heads
        self.qkv       = tf.keras.layers.Dense(3 * dim, use_bias=False)
        self.drop1     = tf.keras.layers.Dropout(dropout)
        self.proj      = tf.keras.layers.Dense(dim, use_bias=False)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        qkv = self.qkv(inputs)
        qkv = tf.keras.layers.Permute((2,1,3))(
            tf.keras.layers.Reshape((-1, self.num_heads, self.dim * 3 // self.num_heads))(qkv))
        q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)
        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        if mask is not None:
            mask = mask[:, None, None, :]
        attn = tf.keras.layers.Softmax(axis=-1)(attn, mask=mask)
        attn = self.drop1(attn)
        x = attn @ v
        x = tf.keras.layers.Reshape((-1, self.dim))(
            tf.keras.layers.Permute((2,1,3))(x))
        return self.proj(x)

def TransformerBlock(dim=256, num_heads=4, expand=4,
                     attn_dropout=0.2, drop_rate=0.2, activation="swish"):
    def apply(inputs):
        x = inputs
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = MultiHeadSelfAttention(dim=dim, num_heads=num_heads, dropout=attn_dropout)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
        x = tf.keras.layers.Add()([inputs, x])
        attn_out = x
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = tf.keras.layers.Dense(dim * expand, use_bias=False, activation=activation)(x)
        x = tf.keras.layers.Dense(dim, use_bias=False)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
        x = tf.keras.layers.Add()([attn_out, x])
        return x
    return apply

def get_model(max_len=64, dropout_step=0, dim=192):
    inp   = tf.keras.Input((max_len, CHANNELS))
    x     = tf.keras.layers.Masking(mask_value=PAD,
                                     input_shape=(max_len, CHANNELS))(inp)
    ksize = 17
    x = tf.keras.layers.Dense(dim, use_bias=False, name="stem_conv")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.95, name="stem_bn")(x)

    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = TransformerBlock(dim, expand=2)(x)

    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = TransformerBlock(dim, expand=2)(x)

    if dim == 384:  # for the 4x sized model
        x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
        x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
        x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
        x = TransformerBlock(dim, expand=2)(x)

        x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
        x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
        x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
        x = TransformerBlock(dim, expand=2)(x)

    x = tf.keras.layers.Dense(dim * 2, activation=None, name="top_conv")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = LateDropout(0.8, start_step=dropout_step)(x)
    x = tf.keras.layers.Dense(NUM_CLASSES, name="classifier")(x)
    return tf.keras.Model(inp, x)

# ─── TRAINING CONFIG ──────────────────────────────────────────────────────────
class CFG:
    n_splits = 5
    save_output = True

    _now = datetime.datetime.now()
    output_dir = f"output/{_now.hour}-{_now.minute}-{_now.day}-{_now.month}-{_now.year}"
    # ⚠️ os.makedirs được gọi trong train_fold() thay vì ở đây
    # để tránh tạo thư mục mỗi lần import train.py (vd: từ camera_demo.py)
    
    seed = 42
    verbose = 2
    
    max_len = 384
    # QUAN TRọNG: replicas/batch_size/lr sẽ được cập nhật động
    # dựa vào số GPU thực tế sau khi khởi tạo STRATEGY.
    # Các giá trị dưới đây là mặc định cho CPU (1 replica)
    replicas = 1
    lr = 5e-4 * replicas
    weight_decay = 0.1
    lr_min = 1e-6
    epoch = 50
    warmup = 0
    batch_size = 64 * replicas
    snapshot_epochs = []
    swa_epochs = []
    
    fp16 = True
    fgm = False
    awp = True
    awp_lambda = 0.2
    awp_start_epoch = 15
    dropout_start_epoch = 15
    resume = 0
    decay_type = 'cosine'
    dim = 192
    comment = f'islr-fp16-192-8-seed{seed}'

def get_strategy(device='GPU'):
    IS_TPU = False
    
    if "TPU" in device:
        try:
            tpu = 'local' if device=='TPU-VM' else None
            print("[INFO] Đang kết nối tới TPU...")
            tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu=tpu)
            strategy = tf.distribute.TPUStrategy(tpu_resolver)
            IS_TPU = True
            print("[INFO] Sử dụng TPU Strategy")
        except Exception as e:
            print(f"[WARN] Không kết nối được TPU ({e}). Chuyển sang GPU/CPU.")
            device = "GPU"

    if device == "GPU" or device == "CPU":
        # 1. Lấy danh sách GPU thực tế
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # 2. Bật Memory Growth để tránh lỗi OOM (Out Of Memory)
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"[INFO] ✅ Đã tìm thấy {len(gpus)} GPU. Đã kích hoạt Memory Growth.")
            except RuntimeError as e:
                print(f"[WARN] Lỗi khi cài đặt Memory Growth: {e}")
                
            if len(gpus) > 1:
                print("[INFO] Sử dụng Multi-GPU (MirroredStrategy)")
                strategy = tf.distribute.MirroredStrategy()
            else:
                print("[INFO] Sử dụng Single GPU (Default Strategy)")
                # Default strategy tự động dùng GPU:0 nếu có
                strategy = tf.distribute.get_strategy() 
        else:
            print("[INFO] ⚠️ Không tìm thấy GPU tương thích. Hệ thống sẽ huấn luyện bằng CPU.")
            strategy = tf.distribute.get_strategy()

    REPLICAS = strategy.num_replicas_in_sync
    print(f'[INFO] Số lượng REPLICAS (Luồng đồng bộ): {REPLICAS}')
    
    return strategy, REPLICAS, IS_TPU

# Khởi tạo chiến lược phân phối (Mặc định ưu tiên GPU)
try:
    STRATEGY, N_REPLICAS, IS_TPU = get_strategy(device="GPU")
except Exception as e:
    print(f"[ERROR] Lỗi khởi tạo phân phối: {e}")
    STRATEGY = tf.distribute.get_strategy()
    N_REPLICAS = 1
    IS_TPU = False

# ↺ Cập nhật CFG theo số replica thực tế — đảm bảo batch_size/lr đúng trên mọi nền tảng
CFG.replicas   = N_REPLICAS
CFG.batch_size = 64 * N_REPLICAS
CFG.lr         = 5e-4 * N_REPLICAS
print(f"[INFO] Cấu hình: replicas={CFG.replicas} | batch_size={CFG.batch_size} | lr={CFG.lr:.2e}")

# ─── TRAINING LOOP ────────────────────────────────────────────────────────────
def train_fold(CFG, fold, train_files, valid_files=None, strategy=STRATEGY, summary=True):
    seed_everything(CFG.seed)
    os.makedirs(CFG.output_dir, exist_ok=True)   # Tạo thư mục output khi bắt đầu train
    tf.keras.backend.clear_session()
    gc.collect()
    # XLA JIT: Chỉ bật trên TPU/Linux (ổn định). Trên Windows GPU có thể gây crash im lặng.
    if IS_TPU:
        tf.config.optimizer.set_jit(True)

    if CFG.fp16:
        try:
            policy = mixed_precision.Policy("mixed_bfloat16")
            mixed_precision.set_global_policy(policy)
        except:
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_global_policy(policy)
    else:
        policy = mixed_precision.Policy("float32")
        mixed_precision.set_global_policy(policy)

    if fold != "all":
        train_ds = get_tfrec_dataset(train_files, batch_size=CFG.batch_size,
                                     max_len=CFG.max_len, drop_remainder=True,
                                     augment=True, repeat=True, shuffle=32768)
        valid_ds = get_tfrec_dataset(valid_files, batch_size=CFG.batch_size,
                                     max_len=CFG.max_len, drop_remainder=False,
                                     repeat=False, shuffle=False)
    else:
        train_ds = get_tfrec_dataset(train_files, batch_size=CFG.batch_size,
                                     max_len=CFG.max_len, drop_remainder=False,
                                     augment=True, repeat=True, shuffle=32768)
        valid_ds = None
        valid_files = []

    num_train = count_data_items(train_files)
    num_valid = count_data_items(valid_files)
    steps_per_epoch = max(1, num_train // CFG.batch_size)

    with strategy.scope():
        dropout_step = CFG.dropout_start_epoch * steps_per_epoch
        model = get_model(max_len=CFG.max_len, dropout_step=dropout_step, dim=CFG.dim)

        if HAS_TF_UTILS:
            schedule = OneCycleLR(
                CFG.lr, CFG.epoch,
                warmup_epochs=CFG.epoch * CFG.warmup,
                steps_per_epoch=steps_per_epoch,
                resume_epoch=CFG.resume,
                decay_epochs=CFG.epoch,
                lr_min=CFG.lr_min,
                decay_type=CFG.decay_type,
                warmup_type="linear",
            )
            decay_schedule = OneCycleLR(
                CFG.lr * CFG.weight_decay, CFG.epoch,
                warmup_epochs=CFG.epoch * CFG.warmup,
                steps_per_epoch=steps_per_epoch,
                resume_epoch=CFG.resume,
                decay_epochs=CFG.epoch,
                lr_min=CFG.lr_min * CFG.weight_decay,
                decay_type=CFG.decay_type,
                warmup_type="linear",
            )
        else:
            schedule = tf.keras.optimizers.schedules.CosineDecay(
                CFG.lr, decay_steps=CFG.epoch * steps_per_epoch,
                alpha=CFG.lr_min / CFG.lr
            )
            decay_schedule = None

        awp_step = CFG.awp_start_epoch * steps_per_epoch
        if HAS_TFA and HAS_TF_UTILS:
            if CFG.fgm:
                model = FGM(model.input, model.output,
                            delta=CFG.awp_lambda, eps=0., start_step=awp_step)
            elif CFG.awp:
                model = AWP(model.input, model.output,
                            delta=CFG.awp_lambda, eps=0., start_step=awp_step)
            opt = tfa.optimizers.RectifiedAdam(
                learning_rate=schedule,
                weight_decay=decay_schedule,
                sma_threshold=4,
            )
            opt = tfa.optimizers.Lookahead(opt, sync_period=5)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=schedule)

        model.compile(
            optimizer=opt,
            loss=[tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)],
            metrics=[[tf.keras.metrics.CategoricalAccuracy()]],
            steps_per_execution=steps_per_epoch,
        )

    if summary:
        print()
        model.summary()
        print()
        print(train_ds, valid_ds)
        print()
        if hasattr(schedule, 'plot'):
            schedule.plot()
        print()
        init = False

    print(f"---------fold{fold}---------")
    print(f"train:{num_train} valid:{num_valid}")
    print()

    if CFG.resume:
        print(f"resume from epoch{CFG.resume}")
        model.load_weights(f"{CFG.output_dir}/{CFG.comment}-fold{fold}-last.weights.h5")
        if train_ds is not None:
            model.evaluate(train_ds.take(steps_per_epoch))
        if valid_ds is not None:
            model.evaluate(valid_ds)

    logger = tf.keras.callbacks.CSVLogger(
        f"{CFG.output_dir}/{CFG.comment}-fold{fold}-logs.csv"
    )
    sv_loss = tf.keras.callbacks.ModelCheckpoint(
        f"{CFG.output_dir}/{CFG.comment}-fold{fold}-best.weights.h5",
        monitor="val_loss", verbose=0, save_best_only=True,
        save_weights_only=True, mode="min", save_freq="epoch",
    )

    callbacks = []
    if HAS_TF_UTILS:
        snap = Snapshot(f"{CFG.output_dir}/{CFG.comment}-fold{fold}", CFG.snapshot_epochs)
        swa = SWA(f"{CFG.output_dir}/{CFG.comment}-fold{fold}", CFG.swa_epochs, strategy=strategy,
                  train_ds=train_ds, valid_ds=valid_ds,
                  valid_steps=-(num_valid // -CFG.batch_size))
                  
    if CFG.save_output:
        callbacks.append(logger)
        if HAS_TF_UTILS:
            callbacks.append(snap)
            callbacks.append(swa)
        if fold != "all":
            callbacks.append(sv_loss)

    history = model.fit(
        train_ds,
        epochs=CFG.epoch - CFG.resume,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        validation_data=valid_ds,
        verbose=CFG.verbose,
        validation_steps=-(num_valid // -CFG.batch_size)
    )

    if CFG.save_output:
        best_weights = f"{CFG.output_dir}/{CFG.comment}-fold{fold}-best.weights.h5"
        
        # 1. Nạp lại trọng số tốt nhất của fold này
        try:
            model.load_weights(best_weights)
        except Exception as e:
            print(f"[WARN] Không thể load best weights: {e}")

    if fold != "all":
        cv = model.evaluate(valid_ds, verbose=CFG.verbose,
                            steps=-(num_valid // -CFG.batch_size))
    else:
        cv = None

    return model, cv, history


def train_folds(CFG, folds, strategy=STRATEGY, summary=True):
    for fold in folds:
        if fold != "all":
            all_files = TRAIN_FILENAMES
            train_files = [x for x in all_files if f"fold_{fold}" not in x]
            valid_files  = [x for x in all_files if f"fold_{fold}" in x]
        else:
            train_files = TRAIN_FILENAMES
            valid_files  = None
        train_fold(CFG, fold, train_files, valid_files, strategy=strategy, summary=summary)
    return


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_folds(CFG, [0])

