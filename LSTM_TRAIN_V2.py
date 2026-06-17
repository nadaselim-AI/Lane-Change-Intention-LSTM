import random
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
TRAIN_NPZ_PATH = r"C:/CARLA_0.9.13/NGSIM/train_70_percent.npz"
VAL_NPZ_PATH   = r"C:/CARLA_0.9.13/NGSIM/val_10_percent.npz"

MODEL_SAVE_PATH = r"C:/CARLA_0.9.13/NGSIM/best_iim_model_v2.pth"

SEED = 42

HIDDEN_SIZE = 128
NUM_LAYERS = 4
DROPOUT = 0.2
NUM_CLASSES = 3

LR = 5e-4
LR_DECAY_GAMMA = 0.9
BATCH_SIZE = 32
EPOCHS = 25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# REPRODUCIBILITY
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# =========================
# LOAD TRAIN + VALIDATION DATA
# =========================
train_data = np.load(TRAIN_NPZ_PATH)
val_data = np.load(VAL_NPZ_PATH)

X_train = train_data["X"]
y_train = train_data["y"]

X_val = val_data["X"]
y_val = val_data["y"]

# Keep first 23 features only if extra features exist
if X_train.shape[-1] > 23:
    X_train = X_train[:, :, :23]

if X_val.shape[-1] > 23:
    X_val = X_val[:, :, :23]

print("Train X shape:", X_train.shape)
print("Train y shape:", y_train.shape)
print("Train class distribution:", Counter(y_train.tolist()))

print("Val X shape:", X_val.shape)
print("Val y shape:", y_val.shape)
print("Val class distribution:", Counter(y_val.tolist()))


# =========================
# STANDARDIZATION USING TRAIN ONLY
# =========================
feature_mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
feature_std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)

feature_std[feature_std < 1e-8] = 1.0

X_train = (X_train - feature_mean) / feature_std
X_val = (X_val - feature_mean) / feature_std

print("Standardization done using training data only.")


# =========================
# DATASET / DATALOADER
# =========================
class IntentionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = IntentionDataset(X_train, y_train)
val_dataset = IntentionDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# =========================
# MODEL SAME AS VERSION 1
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
        dec_in = h_n[-1].unsqueeze(1)
        out, _ = self.lstm(dec_in, (h_n, c_n))
        logits = self.fc(out.squeeze(1))
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


model = IntentionIdentificationModel(
    input_size=X_train.shape[-1],
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    num_classes=NUM_CLASSES,
).to(DEVICE)

print(model)


# =========================
# LOSS / OPTIMIZER / SCHEDULER
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY_GAMMA)
'''
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=True
)
'''
# =========================
# TRAIN / EVAL FUNCTIONS
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # missing
        optimizer.step()

        running_loss += loss.item() * xb.size(0)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            running_loss += loss.item() * xb.size(0)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    return running_loss / total, correct / total


# =========================
# TRAIN LOOP WITH VALIDATION
# =========================
train_losses = []
train_accs = []
val_losses = []
val_accs = []

best_val_loss = float("inf")
patience = 7
no_improve = 0
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, DEVICE
    )

    val_loss, val_acc = evaluate(
        model, val_loader, criterion, DEVICE
    )

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0   # reset

        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,

            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "num_classes": NUM_CLASSES,

            "feature_mean": feature_mean,
            "feature_std": feature_std,
        }, MODEL_SAVE_PATH)

        print(f"Best model saved to: {MODEL_SAVE_PATH}")

    else:
        no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break


# =========================
# PLOTS TRAIN + VALIDATION
# =========================
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_accs, label="Train Accuracy")
plt.plot(epochs, val_accs, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

print(f"Training finished. Best model saved to: {MODEL_SAVE_PATH}")