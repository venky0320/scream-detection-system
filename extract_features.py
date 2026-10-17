"""Step 1: turn the audio dataset into features.csv

Dataset layout:
    dataset/Screaming/*.wav
    dataset/NotScreaming/*.wav
"""
import os
import numpy as np
import pandas as pd
import librosa
from utils import SR, load_segments, extract_features, is_audio_file

DATASET_PATH = "dataset"
CATEGORIES = {"Screaming": 1, "NotScreaming": 0}
AUGMENT = True          # add noisy / pitch-shifted copies of TRAINING-type data
rng = np.random.default_rng(42)


def augment(segment):
    """Return a few modified copies of one 3 s segment."""
    noisy = segment + rng.normal(0, 0.005, segment.shape).astype(np.float32)
    shifted = librosa.effects.pitch_shift(segment, sr=SR, n_steps=float(rng.choice([-2, -1, 1, 2])))
    louder = np.clip(segment * float(rng.uniform(0.6, 1.4)), -1, 1)
    return [noisy, shifted, louder]


rows = []
for category, label in CATEGORIES.items():
    folder = os.path.join(DATASET_PATH, category)
    if not os.path.isdir(folder):
        raise SystemExit(f"Folder not found: {folder}")
    n_files = 0
    for root, _, files in os.walk(folder):
        for name in files:
            if not is_audio_file(name):
                continue
            path = os.path.join(root, name)
            try:
                segments = load_segments(path)
            except Exception as e:
                print(f"Skipping {path}: {e}")
                continue
            n_files += 1
            for seg in segments:
                rows.append((extract_features(seg), label, path, 0))
                if AUGMENT:
                    for aug in augment(seg):
                        rows.append((extract_features(aug), label, path, 1))
    print(f"{category}: {n_files} files processed")

if not rows:
    raise SystemExit("No audio found. Check the dataset folder.")

X = np.vstack([r[0] for r in rows])
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
df["label"] = [r[1] for r in rows]
df["group"] = [r[2] for r in rows]      # source file (prevents train/test leakage)
df["augmented"] = [r[3] for r in rows]
df.to_csv("features.csv", index=False)

orig = df[df.augmented == 0]
print("\nOriginal segments per class:")
print(orig["label"].map({1: "Screaming", 0: "NotScreaming"}).value_counts())
print(f"Total rows (incl. augmented): {len(df)}")
print("Saved features.csv")