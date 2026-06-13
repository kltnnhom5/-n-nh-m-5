# ============================================================
#  BƯỚC 4 — Train CNN + LSTM (Baseline Model)
#  Đề tài: Phân tích cảm xúc phản hồi sinh viên với PhoBERT
# ============================================================

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ------------------------------------------------------------
# 1. CẤU HÌNH
# ------------------------------------------------------------
CACHE_DIR   = "cache"
MODEL_DIR   = "models"
OUTPUT_DIR  = "outputs"

VOCAB_SIZE   = 64000
EMBED_DIM    = 128
NUM_FILTERS  = 128    # số filter CNN
KERNEL_SIZES = [3, 5, 7]  # kích thước cửa sổ CNN (bigram, trigram, 4-gram)
HIDDEN_DIM   = 256    # số unit LSTM
NUM_CLASSES  = 3
NUM_LAYERS   = 2
DROPOUT      = 0.3
BATCH_SIZE   = 32
NUM_EPOCHS   = 10
LR           = 1e-3

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 55)
print("BƯỚC 4 — TRAIN CNN + LSTM (BASELINE)")
print("=" * 55)
print(f"Thiết bị : {DEVICE}")
print(f"CNN filters    : {NUM_FILTERS} × {KERNEL_SIZES}")
print(f"LSTM hidden    : {HIDDEN_DIM}")

# ------------------------------------------------------------
# 2. LOAD CACHE
# ------------------------------------------------------------
print("\n[1/6] Load cache từ Bước 2...")

def load_cache(name):
    with open(f"{CACHE_DIR}/{name}.pkl", "rb") as f:
        return pickle.load(f)

cache_train = load_cache("train")
cache_val   = load_cache("val")
cache_test  = load_cache("test")

print(f"  Train : {len(cache_train):,} | Val : {len(cache_val):,} | Test : {len(cache_test):,}")

# ------------------------------------------------------------
# 3. DATASET & DATALOADER
# ------------------------------------------------------------
print("\n[2/6] Tạo DataLoader...")

class SentimentDataset(Dataset):
    def __init__(self, cache):
        self.data = cache
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "label":     torch.tensor(item["label"],     dtype=torch.long),
        }

train_loader = DataLoader(SentimentDataset(cache_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(SentimentDataset(cache_val),   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(SentimentDataset(cache_test),  batch_size=BATCH_SIZE, shuffle=False)

print(f"  Train batch: {len(train_loader)} | Val batch: {len(val_loader)}")

# ------------------------------------------------------------
# 4. KIẾN TRÚC CNN + LSTM
# ------------------------------------------------------------
print("\n[3/6] Định nghĩa kiến trúc CNN + LSTM...")

class CNNLSTMClassifier(nn.Module):
    """
    Kiến trúc:
      Embedding → CNN (multi-scale) → MaxPooling → LSTM → Dropout → Linear
    
    CNN trích xuất đặc trưng cục bộ (n-gram features)
    LSTM nắm bắt phụ thuộc tuần tự dài hạn
    Kết hợp 2 ưu điểm → tốt hơn chỉ dùng LSTM hay CNN đơn thuần
    """
    def __init__(self, vocab_size, embed_dim, num_filters,
                 kernel_sizes, hidden_dim, num_classes, dropout):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.emb_dropout = nn.Dropout(dropout)

        # CNN layers — mỗi kernel size 1 Conv1d riêng
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                    padding=k // 2      # giữ độ dài sequence
                ),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for k in kernel_sizes
        ])

        # LSTM nhận đầu vào là concat các CNN output
        cnn_out_dim = num_filters * len(kernel_sizes)
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=hidden_dim,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=dropout if NUM_LAYERS > 1 else 0,
            bidirectional=True
        )

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids):
        # Embedding: [batch, seq_len] → [batch, seq_len, embed_dim]
        x = self.embedding(input_ids)
        x = self.emb_dropout(x)

        # CNN cần [batch, channels, seq_len]
        x_cnn = x.permute(0, 2, 1)

        # Áp dụng từng Conv1d rồi concat
        cnn_outs = []
        for conv in self.convs:
            out = conv(x_cnn)           # [batch, num_filters, seq_len]
            cnn_outs.append(out)

        # Concat theo chiều filter: [batch, num_filters*3, seq_len]
        x_cat = torch.cat(cnn_outs, dim=1)

        # Chuyển lại cho LSTM: [batch, seq_len, num_filters*3]
        x_lstm_in = x_cat.permute(0, 2, 1)

        # LSTM
        out, (hidden, _) = self.lstm(x_lstm_in)

        # Lấy hidden state của 2 chiều cuối cùng
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = self.dropout(hidden)

        return self.classifier(hidden)


model = CNNLSTMClassifier(
    vocab_size   = VOCAB_SIZE,
    embed_dim    = EMBED_DIM,
    num_filters  = NUM_FILTERS,
    kernel_sizes = KERNEL_SIZES,
    hidden_dim   = HIDDEN_DIM,
    num_classes  = NUM_CLASSES,
    dropout      = DROPOUT
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Tổng tham số: {total_params:,}")
print(f"  CNN kernel sizes : {KERNEL_SIZES} (bigram, trigram, 4-gram)")
print(f"  CNN filters/size : {NUM_FILTERS}")
print(f"  BiLSTM hidden    : {HIDDEN_DIM} × 2 (bidirectional)")

# ------------------------------------------------------------
# 5. LOSS VỚI CLASS WEIGHTS
# ------------------------------------------------------------
label_counts = [0, 0, 0]
for item in cache_train:
    label_counts[item["label"]] += 1
total = sum(label_counts)
weights = [total / (3 * c) for c in label_counts]
weight_tensor = torch.tensor(weights, dtype=torch.float).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weight_tensor)
print(f"\n  Class weights: {[round(w, 3) for w in weights]}")

# ------------------------------------------------------------
# 6. HÀM TRAIN & EVALUATE
# ------------------------------------------------------------
print("\n[4/6] Định nghĩa hàm train và evaluate...")

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        ids    = batch["input_ids"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        optimizer.zero_grad()
        logits = model(ids)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), correct / total


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0
    with torch.no_grad():
        for batch in loader:
            ids    = batch["input_ids"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits = model(ids)
            total_loss += criterion(logits, labels).item()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    f1  = f1_score(all_labels, all_preds, average="macro")
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc, f1, all_labels, all_preds

# ------------------------------------------------------------
# 7. TRAINING
# ------------------------------------------------------------
print("\n[5/6] Bắt đầu training CNN+LSTM...")
print(f"  Epochs: {NUM_EPOCHS} | LR: {LR} | Batch: {BATCH_SIZE}")
print("  " + "=" * 45)

optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

best_f1    = 0
best_path  = f"{MODEL_DIR}/CNN_LSTM_best.pt"
history    = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
    val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_f1"].append(val_f1)
    history["val_acc"].append(val_acc)

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), best_path)
        mark = " ← best"
    else:
        mark = ""

    print(f"  Epoch {epoch:2d}/{NUM_EPOCHS} | "
          f"Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f} | "
          f"Val F1: {val_f1:.4f}{mark}")

print(f"\n  Best Val F1: {best_f1:.4f} → {best_path}")

# ------------------------------------------------------------
# 8. ĐÁNH GIÁ TRÊN TEST SET
# ------------------------------------------------------------
print("\n[6/6] Đánh giá trên Test Set...")
label_names = ["Tiêu cực", "Trung tính", "Tích cực"]

model.load_state_dict(torch.load(best_path, map_location=DEVICE))
_, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader)

print(f"\n  Kết quả CNN+LSTM trên Test Set:")
print(f"  Accuracy  : {test_acc:.4f}")
print(f"  F1-macro  : {test_f1:.4f}")
print(classification_report(y_true, y_pred,
      target_names=label_names, digits=4,
      zero_division=0))

# ------------------------------------------------------------
# 9. VẼ LEARNING CURVE
# ------------------------------------------------------------
print("Vẽ biểu đồ Learning Curve...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Learning Curve — CNN+LSTM", fontsize=13)
epochs = range(1, NUM_EPOCHS + 1)

ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#378ADD")
ax1.plot(epochs, history["val_loss"],   "s--",label="Val Loss",   color="#E24B4A")
ax1.set_title("Loss theo Epoch")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.legend(); ax1.grid(alpha=0.3)
ax1.spines[["top","right"]].set_visible(False)

ax2.plot(epochs, history["val_f1"],  "^-", label="Val F1-macro", color="#1D9E75")
ax2.plot(epochs, history["val_acc"], "D--",label="Val Accuracy", color="#BA7517")
ax2.set_title("F1 & Accuracy theo Epoch")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score")
ax2.legend(); ax2.grid(alpha=0.3)
ax2.spines[["top","right"]].set_visible(False)

plt.tight_layout()
curve_path = f"{OUTPUT_DIR}/learning_curve_cnn_lstm.png"
plt.savefig(curve_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Đã lưu: {curve_path}")

# ------------------------------------------------------------
# 10. CẬP NHẬT FILE KẾT QUẢ (thêm vào results_baseline.json)
# ------------------------------------------------------------
result_path = f"{OUTPUT_DIR}/results_baseline.json"
if os.path.exists(result_path):
    with open(result_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)
else:
    all_results = {}

all_results["CNN+LSTM"] = {
    "accuracy": round(test_acc, 4),
    "f1_macro": round(test_f1, 4)
}

with open(result_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

# ------------------------------------------------------------
# TỔNG KẾT
# ------------------------------------------------------------
print("\n" + "=" * 55)
print("BƯỚC 4 HOÀN THÀNH!")
print("=" * 55)
print("Các file đã tạo:")
print(f"  models/CNN_LSTM_best.pt")
print(f"  outputs/learning_curve_cnn_lstm.png")
print(f"  outputs/results_baseline.json  (đã cập nhật)")

print(f"\nTóm tắt kết quả tất cả baseline đến nay:")
for name, r in all_results.items():
    print(f"  {name:10s} → Acc: {r['accuracy']:.4f} | F1-macro: {r['f1_macro']:.4f}")
print("\nBước tiếp theo: step5_train_phobert.py — Model chính!")