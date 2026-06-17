import os
import random
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================
US101_NPZ_PATH = r"C:/CARLA_0.9.13/NGSIM/US101_all_samples_3s5s_fixed.npz"
I80_NPZ_PATH   = r"C:/CARLA_0.9.13/NGSIM/I80_all_samples_3s5s_fixed.npz"

TRAIN_SAVE_PATH = r"C:/CARLA_0.9.13/NGSIM/train_70_percent.npz"
VAL_SAVE_PATH   = r"C:/CARLA_0.9.13/NGSIM/val_10_percent.npz"
TEST_SAVE_PATH  = r"C:/CARLA_0.9.13/NGSIM/test_20_percent.npz"

SEED = 42
SAMPLES_PER_CLASS = 5000
TEST_SIZE = 0.20
VAL_SIZE_FROM_TEMP = 0.125   # because 0.125 * 80% = 10% of total

# =========================
# SEED
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(SEED)

# =========================
# LOAD US101
# =========================
us101_data = np.load(US101_NPZ_PATH)
X_us101 = us101_data["X"]
y_us101 = us101_data["y"]

if X_us101.shape[-1] > 23:
    X_us101 = X_us101[:, :, :23]

print("US101 shape:", X_us101.shape, y_us101.shape)
print("US101 distribution:", Counter(y_us101.tolist()))

# =========================
# LOAD I80
# =========================
i80_data = np.load(I80_NPZ_PATH)
X_i80 = i80_data["X"]
y_i80 = i80_data["y"]

if X_i80.shape[-1] > 23:
    X_i80 = X_i80[:, :, :23]

print("I80 shape:", X_i80.shape, y_i80.shape)
print("I80 distribution:", Counter(y_i80.tolist()))

# =========================
# MERGE US101 + I80
# =========================
X_all = np.concatenate([X_us101, X_i80], axis=0)
y_all = np.concatenate([y_us101, y_i80], axis=0)

print("Merged shape:", X_all.shape, y_all.shape)
print("Merged distribution:", Counter(y_all.tolist()))

# =========================
# BALANCED SAMPLING
# 5000 samples from each class
# =========================
def sample_balanced_subset(X, y, samples_per_class=5000, seed=42):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)

    selected_indices = []

    for c in classes:
        idx = np.where(y == c)[0]

        if len(idx) < samples_per_class:
            raise ValueError(
                f"Class {c} has only {len(idx)} samples, أقل من المطلوب {samples_per_class}."
            )

        chosen = rng.choice(idx, size=samples_per_class, replace=False)
        selected_indices.extend(chosen.tolist())

    rng.shuffle(selected_indices)
    selected_indices = np.array(selected_indices)

    return X[selected_indices], y[selected_indices]

X_bal, y_bal = sample_balanced_subset(
    X_all,
    y_all,
    samples_per_class=SAMPLES_PER_CLASS,
    seed=SEED
)

print("Balanced shape:", X_bal.shape, y_bal.shape)
print("Balanced distribution:", Counter(y_bal.tolist()))

# =========================
# SPLIT: 80% TEMP + 20% TEST
# =========================
X_temp, X_test, y_temp, y_test = train_test_split(
    X_bal,
    y_bal,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y_bal
)

# =========================
# SPLIT TEMP: 70% TRAIN + 10% VALIDATION
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=VAL_SIZE_FROM_TEMP,
    random_state=SEED,
    stratify=y_temp
)

# =========================
# PRINT FINAL SHAPES
# =========================
print("Train shape      :", X_train.shape, y_train.shape)
print("Validation shape :", X_val.shape, y_val.shape)
print("Test shape       :", X_test.shape, y_test.shape)

print("Train distribution      :", Counter(y_train.tolist()))
print("Validation distribution :", Counter(y_val.tolist()))
print("Test distribution       :", Counter(y_test.tolist()))

# =========================
# SAVE EACH PART SEPARATELY
# =========================
np.savez_compressed(TRAIN_SAVE_PATH, X=X_train, y=y_train)
np.savez_compressed(VAL_SAVE_PATH, X=X_val, y=y_val)
np.savez_compressed(TEST_SAVE_PATH, X=X_test, y=y_test)

print(f"Saved train set to      : {TRAIN_SAVE_PATH}")
print(f"Saved validation set to : {VAL_SAVE_PATH}")
print(f"Saved test set to       : {TEST_SAVE_PATH}")
