import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
TEST_NPZ_PATH = r"C:/CARLA_0.9.13/NGSIM/test_20_percent.npz"
MODEL_PATH = r"C:/CARLA_0.9.13/NGSIM/best_iim_model_same_v3.pth"
FIGURE_SAVE_PATH = r"C:/CARLA_0.9.13/NGSIM/test_confusion_like_paper.png"

BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# IMPORTANT:
# Make sure your class mapping is:
# 0 = Left, 1 = Straight, 2 = Right
CLASS_NAMES = ["Straight", "Left", "Right"]


# =========================
# DATASET
# =========================
class IntentionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================
# MODEL
# =========================
class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, x):
        _, (h_n, c_n) = self.lstm(x)
        return h_n, c_n


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, h_n, c_n):
        dec_in = h_n[-1].unsqueeze(1)          # [B, 1, H]
        out, _ = self.lstm(dec_in, (h_n, c_n)) # [B, 1, H]
        logits = self.fc(out.squeeze(1))       # [B, C]
        return logits


class IntentionIdentificationModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes):
        super().__init__()
        self.encoder = EncoderLSTM(input_size, hidden_size, num_layers, dropout)
        self.decoder = DecoderLSTM(hidden_size, num_layers, dropout, num_classes)

    def forward(self, x):
        h_n, c_n = self.encoder(x)
        logits = self.decoder(h_n, c_n)
        return logits


# =========================
# THRESHOLD LOGIC (PAPER STYLE)
# =========================
def apply_threshold_logic(prob_vector: np.ndarray) -> np.ndarray:
    """
    Input: softmax probabilities of shape [3]
    Output: hard one-hot vector of shape [3]


    """
    straight_p, left_p, right_p = prob_vector

    # Apply paper thresholds
    if left_p >= 0.8:
        return np.array([0, 1, 0], dtype=np.int64)

    elif right_p >= 0.8:
        return np.array([0, 0, 1], dtype=np.int64)

    elif straight_p >= 0.7:
        return np.array([1, 0, 0], dtype=np.int64)

    else:
        # Fallback to argmax if no threshold is met
        hard = np.zeros(3, dtype=np.int64)
        hard[np.argmax(prob_vector)] = 1
        return hard


# =========================
# LOAD MODEL CHECKPOINT
# =========================
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

hidden_size = checkpoint["hidden_size"]
num_layers = checkpoint["num_layers"]
dropout = checkpoint["dropout"]
num_classes = checkpoint["num_classes"]
feature_mean = checkpoint["feature_mean"]
feature_std = checkpoint["feature_std"]

# convert to numpy if loaded as tensors
if isinstance(feature_mean, torch.Tensor):
    feature_mean = feature_mean.cpu().numpy()
if isinstance(feature_std, torch.Tensor):
    feature_std = feature_std.cpu().numpy()

feature_std[feature_std < 1e-8] = 1.0


# =========================
# LOAD TEST DATA
# =========================
data = np.load(TEST_NPZ_PATH)
X_test = data["X"]
y_test = data["y"]

# only if extra features exist
if X_test.shape[-1] > 23:
    X_test = X_test[:, :, :23]

print("Test X shape:", X_test.shape)
print("Test y shape:", y_test.shape)

# standardize using TRAIN statistics only
X_test = (X_test - feature_mean) / feature_std

test_dataset = IntentionDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# =========================
# BUILD MODEL
# =========================
model = IntentionIdentificationModel(
    input_size=X_test.shape[-1],
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=dropout,
    num_classes=num_classes,
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded successfully.")


# =========================
# EVALUATION
# =========================
all_soft_probs = []
all_hard_probs = []
all_preds = []
all_labels = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        logits = model(xb)

        soft_probs = torch.softmax(logits, dim=1).cpu().numpy()
        hard_probs = np.array([apply_threshold_logic(p) for p in soft_probs])
        preds = np.argmax(hard_probs, axis=1)

        all_soft_probs.extend(soft_probs)
        all_hard_probs.extend(hard_probs)
        all_preds.extend(preds)
        all_labels.extend(yb.numpy())

all_soft_probs = np.array(all_soft_probs)
all_hard_probs = np.array(all_hard_probs)
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)


# =========================
# METRICS
# =========================
overall_acc = accuracy_score(all_labels, all_preds)

precision, recall, f1, support = precision_recall_fscore_support(
    all_labels,
    all_preds,
    labels=[0, 1, 2],
    zero_division=0
)

print("\n==============================")
print("OVERALL RESULTS")
print("==============================")
print(f"Overall Accuracy: {overall_acc:.4f}")

print("\n==============================")
print("PER-CLASS METRICS")
print("==============================")
for i, cls_name in enumerate(CLASS_NAMES):
    print(
        f"{cls_name}: "
        f"Precision={precision[i]:.4f}, "
        f"Recall={recall[i]:.4f}, "
        f"F1-score={f1[i]:.4f}, "
        f"Support={support[i]}"
    )

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")
print(
    classification_report(
        all_labels,
        all_preds,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)

cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])

print("==============================")
print("CONFUSION MATRIX")
print("==============================")
print(cm)


# =========================
# FIGURE SIMILAR TO PAPER FIGURE 5
# x-axis = Actual intention
# stacked bars = predicted distribution
# =========================
# cm rows = actual, cols = predicted
row_sums = cm.sum(axis=1, keepdims=True)
cm_norm = cm.astype(np.float64) / np.clip(row_sums, 1, None)

actual_positions = np.arange(len(CLASS_NAMES))

pred_straight = cm_norm[:, 0]
pred_left = cm_norm[:, 1]
pred_right = cm_norm[:, 2]

plt.figure(figsize=(8, 6))
plt.bar(actual_positions, pred_straight, label="Straight")
plt.bar(actual_positions, pred_left, bottom=pred_straight, label="Left")
plt.bar(actual_positions, pred_right, bottom=pred_straight + pred_left, label="Right")

plt.xticks(actual_positions, CLASS_NAMES)
plt.ylim(0, 1.05)
plt.ylabel("Prediction intention")
plt.xlabel("Actual intention")
plt.title("Confusion Matrix of Intention Identification")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURE_SAVE_PATH, dpi=300)
plt.show()

print(f"\nFigure saved to: {FIGURE_SAVE_PATH}")


# =========================
# EXAMPLES
# =========================
print("\nExample decisions for first 5 test samples:")
for i in range(min(5, len(all_soft_probs))):
    print(
        f"Sample {i}: "
        f"soft_probs={all_soft_probs[i]}, "
        f"hard_probs={all_hard_probs[i]}, "
        f"pred={all_preds[i]}, "
        f"true={all_labels[i]}"
    )