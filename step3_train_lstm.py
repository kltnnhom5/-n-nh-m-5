# ============================================================
#  BƯỚC 3 — Train LSTM & BiLSTM (Baseline Models)
#  Đề tài: Phân tích cảm xúc phản hồi sinh viên với PhoBERT
# ============================================================

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ------------------------------------------------------------
# 1. CẤU HÌNH
# ------------------------------------------------------------
CACHE_DIR   = "cache"
MODEL_DIR   = "models"
OUTPUT_DIR  = "outputs"

VOCAB_SIZE  = 64000   # vocab PhoBERT
EMBED_DIM   = 128     # kích thước embedding
HIDDEN_DIM  = 256     # số unit LSTM
NUM_CLASSES = 3       # tiêu cực / trung tính / tích cực
NUM_LAYERS  = 2       # số lớp LSTM
DROPOUT     = 0.3
BATCH_SIZE  = 32
NUM_EPOCHS  = 10
LR          = 1e-3
MAX_LENGTH  = 256

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 55)
print("BƯỚC 3 — TRAIN LSTM & BiLSTM (BASELINE)")
print("=" * 55)
print(f"Thiết bị: {DEVICE}")

# ------------------------------------------------------------
# 2. LOAD CACHE TỪ BƯỚC 2
# ------------------------------------------------------------
print("\n[1/6] Load cache từ Bước 2...")

def load_cache(split_name):
    path = f"{CACHE_DIR}/{split_name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

cache_train = load_cache("train")
cache_val   = load_cache("val")
cache_test  = load_cache("test")

print(f"  Train : {len(cache_train):,} mẫu")
print(f"  Val   : {len(cache_val):,} mẫu")
print(f"  Test  : {len(cache_test):,} mẫu")

# ------------------------------------------------------------
# 3. DATASET & DATALOADER
# ------------------------------------------------------------
print("\n[2/6] Tạo Dataset và DataLoader...")

class SentimentDataset(Dataset):
    def __init__(self, cache):
        self.data = cache

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"],      dtype=torch.long),
            "label":     torch.tensor(item["label"],          dtype=torch.long),
        }

train_dataset = SentimentDataset(cache_train)
val_dataset   = SentimentDataset(cache_val)
test_dataset  = SentimentDataset(cache_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"  Batch size : {BATCH_SIZE}")
print(f"  Train batch: {len(train_loader)}")
print(f"  Val batch  : {len(val_loader)}")

# ------------------------------------------------------------
# 4. ĐỊNH NGHĨA MÔ HÌNH LSTM & BiLSTM
# ------------------------------------------------------------
print("\n[3/6] Định nghĩa kiến trúc mô hình...")

class LSTMClassifier(nn.Module):
    """LSTM đơn hướng"""
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_classes, num_layers, dropout, bidirectional=False):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.lstm       = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        self.dropout    = nn.Dropout(dropout)
        lstm_out_dim    = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids):
        # input_ids: [batch, seq_len]
        x = self.embedding(input_ids)          # [batch, seq_len, embed_dim]
        x = self.dropout(x)
        out, (hidden, _) = self.lstm(x)        # out: [batch, seq_len, hidden]

        if self.lstm.bidirectional:
            # Ghép hidden state của 2 chiều
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]                # [batch, hidden_dim]

        hidden = self.dropout(hidden)
        return self.classifier(hidden)         # [batch, num_classes]


# Khởi tạo 2 mô hình
lstm_model = LSTMClassifier(
    vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES,
    num_layers=NUM_LAYERS, dropout=DROPOUT,
    bidirectional=False
).to(DEVICE)

bilstm_model = LSTMClassifier(
    vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES,
    num_layers=NUM_LAYERS, dropout=DROPOUT,
    bidirectional=True
).to(DEVICE)

# Đếm tham số
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"  LSTM   params: {count_params(lstm_model):,}")
print(f"  BiLSTM params: {count_params(bilstm_model):,}")

# ------------------------------------------------------------
# 5. HÀM TRAIN & EVALUATE
# ------------------------------------------------------------
print("\n[4/6] Định nghĩa hàm train và evaluate...")

# Xử lý class imbalance bằng class weights
label_counts = [0, 0, 0]
for item in cache_train:
    label_counts[item["label"]] += 1
total = sum(label_counts)
weights = [total / (3 * c) for c in label_counts]
weight_tensor = torch.tensor(weights, dtype=torch.float).to(DEVICE)
print(f"  Class weights: {[round(w,3) for w in weights]}")
print(f"  (Trung tính được boost vì ít mẫu nhất)")

criterion = nn.CrossEntropyLoss(weight=weight_tensor)


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        labels    = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            labels    = batch["label"].to(DEVICE)
            logits    = model(input_ids)
            loss      = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1  = f1_score(all_labels, all_preds, average="macro")
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc, f1, all_labels, all_preds


def train_model(model, model_name, num_epochs=NUM_EPOCHS):
    print(f"\n  {'='*45}")
    print(f"  Training: {model_name}")
    print(f"  {'='*45}")

    optimizer    = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler    = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_f1      = 0
    best_path    = f"{MODEL_DIR}/{model_name}_best.pt"
    history      = {"train_loss":[], "val_loss":[], "val_f1":[], "val_acc":[]}

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)

        # Lưu model tốt nhất
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            mark = " ← best"
        else:
            mark = ""

        print(f"  Epoch {epoch:2d}/{num_epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val F1: {val_f1:.4f}{mark}")

    print(f"\n  Best Val F1: {best_f1:.4f} → {best_path}")
    return history, best_path


# ------------------------------------------------------------
# 6. TRAIN CẢ 2 MÔ HÌNH
# ------------------------------------------------------------
print("\n[5/6] Bắt đầu training...")

# Train LSTM
history_lstm, path_lstm = train_model(lstm_model, "LSTM")

# Train BiLSTM
history_bilstm, path_bilstm = train_model(bilstm_model, "BiLSTM")

# ------------------------------------------------------------
# 7. ĐÁNH GIÁ TRÊN TEST SET
# ------------------------------------------------------------
print("\n[6/6] Đánh giá trên Test Set...")
label_names = ["Tiêu cực", "Trung tính", "Tích cực"]

results = {}

for model, name, path in [
    (lstm_model,   "LSTM",   path_lstm),
    (bilstm_model, "BiLSTM", path_bilstm),
]:
    # Load weights tốt nhất
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    _, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader)

    print(f"\n  {'='*45}")
    print(f"  Kết quả {name} trên Test Set")
    print(f"  {'='*45}")
    print(f"  Accuracy  : {test_acc:.4f}")
    print(f"  F1-macro  : {test_f1:.4f}")
    print(classification_report(y_true, y_pred,
          target_names=label_names, digits=4))

    results[name] = {
        "acc": test_acc, "f1": test_f1,
        "y_true": y_true, "y_pred": y_pred
    }

# ------------------------------------------------------------
# 8. VẼ BIỂU ĐỒ LEARNING CURVE
# ------------------------------------------------------------
print("\nVẽ biểu đồ Learning Curve...")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Learning Curve — LSTM & BiLSTM", fontsize=13)

pairs = [
    (history_lstm,   "LSTM",   axes[0]),
    (history_bilstm, "BiLSTM", axes[1]),
]

for history, name, (ax_loss, ax_f1) in pairs:
    epochs = range(1, len(history["train_loss"]) + 1)

    ax_loss.plot(epochs, history["train_loss"], "o-",
                 label="Train Loss", color="#378ADD")
    ax_loss.plot(epochs, history["val_loss"],   "s--",
                 label="Val Loss",   color="#E24B4A")
    ax_loss.set_title(f"{name} — Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)
    ax_loss.spines[["top","right"]].set_visible(False)

    ax_f1.plot(epochs, history["val_f1"],  "^-",
               label="Val F1-macro", color="#1D9E75")
    ax_f1.plot(epochs, history["val_acc"], "D--",
               label="Val Accuracy", color="#BA7517")
    ax_f1.set_title(f"{name} — F1 & Accuracy")
    ax_f1.set_xlabel("Epoch")
    ax_f1.set_ylabel("Score")
    ax_f1.legend()
    ax_f1.grid(alpha=0.3)
    ax_f1.spines[["top","right"]].set_visible(False)

plt.tight_layout()
curve_path = f"{OUTPUT_DIR}/learning_curve_lstm_bilstm.png"
plt.savefig(curve_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Đã lưu: {curve_path}")

# ------------------------------------------------------------
# 9. LƯU KẾT QUẢ ĐỂ SO SÁNH Ở BƯỚC 6
# ------------------------------------------------------------
import json
summary = {
    name: {"accuracy": round(r["acc"],4), "f1_macro": round(r["f1"],4)}
    for name, r in results.items()
}
with open(f"{OUTPUT_DIR}/results_baseline.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 55)
print("BƯỚC 3 HOÀN THÀNH!")
print("=" * 55)
print("Các file đã tạo:")
print(f"  models/LSTM_best.pt")
print(f"  models/BiLSTM_best.pt")
print(f"  outputs/learning_curve_lstm_bilstm.png")
print(f"  outputs/results_baseline.json")
print(f"\nTóm tắt kết quả:")
for name, r in results.items():
    print(f"  {name:8s} → Acc: {r['acc']:.4f} | F1-macro: {r['f1']:.4f}")
print("\nBước tiếp theo: step4_train_cnn_lstm.py")