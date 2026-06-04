import tensorflow as tf
import numpy as np
import os, glob

tfrecord_dir = os.path.join('data', 'Vietnamese', 'TFRecord')
files = sorted(glob.glob(os.path.join(tfrecord_dir, '*.tfrecords')))

label_names = {0: 'Miama', 1: 'ghen ti', 2: 'lung tung', 3: 'dia chi'}
LHAND = list(range(468, 489))
RHAND = list(range(522, 543))
POINT_LANDMARKS = (
    [0,61,185,40,39,37,267,269,270,409,
     291,146,91,181,84,17,314,405,321,375,
     78,191,80,81,82,13,312,311,310,415,
     95,88,178,87,14,317,402,318,324,308]
    + list(range(468, 489))
    + list(range(522, 543))
    + [1,2,98,327]
    + [33,7,163,144,145,153,154,155,133,246,161,160,159,158,157,173]
    + [263,249,390,373,374,380,381,382,362,466,388,387,386,385,384,398]
)

print("="*80)
print("PHAN TICH CHI TIET TUNG SAMPLE TRONG DATASET")
print("="*80)

for f in files:
    ds = tf.data.TFRecordDataset(f)
    for raw in ds:
        feat = tf.io.parse_single_example(raw, {
            'coordinates': tf.io.FixedLenFeature([], tf.string),
            'sign':        tf.io.FixedLenFeature([], tf.int64),
        })
        sign   = int(feat['sign'].numpy())
        coords = tf.reshape(tf.io.decode_raw(feat['coordinates'], tf.float32), (-1, 543, 3)).numpy()
        name   = label_names.get(sign, '?')

        lhand_active = (~np.isnan(coords[:, LHAND, :]).all(axis=(1,2))).sum()
        rhand_active = (~np.isnan(coords[:, RHAND, :]).all(axis=(1,2))).sum()
        nframes = coords.shape[0]

        # After preprocessing: which channels carry actual signal vs zeros?
        # gather point_landmarks, keep only x,y (axis 2 idx :2)
        gathered = coords[:, POINT_LANDMARKS, :2]   # (T, 118, 2)
        has_signal = ~np.isnan(gathered)
        signal_ratio = has_signal.mean()

        print(f"\n  Class {sign} ({name}):")
        print(f"    Total frames : {nframes}")
        print(f"    LHAND active : {lhand_active} frames ({100*lhand_active/nframes:.1f}%)")
        print(f"    RHAND active : {rhand_active} frames ({100*rhand_active/nframes:.1f}%)")
        print(f"    Signal ratio : {100*signal_ratio:.1f}% of coords have real values (not NaN)")
        
        # Per-hand column analysis in POINT_LANDMARKS
        # LHAND occupies positions 40:61 (21 pts x 2 coords = 42 channels)
        # RHAND occupies positions 61:82 (21 pts x 2 coords = 42 channels)  
        lhand_in_pl = slice(40, 61)  # 21 points
        rhand_in_pl = slice(61, 82)
        lhand_signal = (~np.isnan(gathered[:, 40:61, :])).mean()
        rhand_signal = (~np.isnan(gathered[:, 61:82, :])).mean()
        print(f"    LHAND signal : {100*lhand_signal:.1f}% (within POINT_LANDMARKS)")
        print(f"    RHAND signal : {100*rhand_signal:.1f}% (within POINT_LANDMARKS)")

print()
print("="*80)
print("CONCLUSION:")
print("  - Khi inference: NaN -> 0.0 (sau Preprocess)")
print("  - Model se so sanh pattern voi tung class da hoc")
print("  - Class nao co nhieu 0.0 nhat trong training -> 'nen' cua class do la nhieu 0")
print("  - 'lung tung' co it 0 nhat (tay day du) -> bat ky input co nhieu signal se match lung tung")
print("="*80)
