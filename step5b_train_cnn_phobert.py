# ============================================================
#  BƯỚC 5B — Train CNN + PhoBERT (Model mới)
#  Kiến trúc: PhoBERT → CNN multi-scale → Classifier
#  So sánh với PhoBERT đơn thuần (step5)
# ============================================================

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, get_linear_schedule_with_warmup
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

MODEL_NAME   = "vinai/phobert-base"
NUM_CLASSES  = 3
BATCH_SIZE   = 16
NUM_EPOCHS   = 10
LR           = 2e-5
DROPOUT      = 0.3
PATIENCE     = 3          # early stopping

# CNN config
NUM_FILTERS  = 256        # số filter mỗi kernel
KERNEL_SIZES = [2, 3, 4]  # kích thước cửa sổ CNN

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 55)
print("BƯỚC 5B — TRAIN CNN + PHOBERT")
print("=" * 55)
print(f"Model     : {MODEL_NAME} + CNN")
print(f"Device    : {DEVICE}")
print(f"CNN filters   : {NUM_FILTERS} × {KERNEL_SIZES}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs    : {NUM_EPOCHS} (early stop={PATIENCE})")
print(f"LR        : {LR}")

# ------------------------------------------------------------
# 2. LOAD CACHE
# ------------------------------------------------------------
print("\n[1/6] Load cache...")

def load_cache(name):
    with open(f"{CACHE_DIR}/{name}.pkl", "rb") as f:
        return pickle.load(f)

cache_train = load_cache("train")
cache_val   = load_cache("val")
cache_test  = load_cache("test")

print(f"  Train: {len(cache_train):,} | Val: {len(cache_val):,} | Test: {len(cache_test):,}")

# ------------------------------------------------------------
# 3. DATASET
# ------------------------------------------------------------
class PhoBERTDataset(Dataset):
    def __init__(self, cache):
        self.data = cache
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids":      torch.tensor(item["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "label":          torch.tensor(item["label"],          dtype=torch.long),
        }

train_loader = DataLoader(PhoBERTDataset(cache_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(PhoBERTDataset(cache_val),   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(PhoBERTDataset(cache_test),  batch_size=BATCH_SIZE, shuffle=False)

# ------------------------------------------------------------
# 4. KIẾN TRÚC CNN + PHOBERT
# ------------------------------------------------------------
print("\n[2/6] Định nghĩa kiến trúc CNN + PhoBERT...")

class CNNPhoBERT(nn.Module):
    """
    Kiến trúc:
      Input → PhoBERT (12 Transformer layers)
           → Lấy toàn bộ sequence hidden states [batch, seq_len, 768]
           → CNN multi-scale (kernel 2,3,4) trích đặc trưng n-gram
           → MaxPooling over time
           → Concat 3 CNN outputs
           → Dropout → Linear → 3 classes

    Ưu điểm so với PhoBERT đơn thuần:
      - PhoBERT đơn thuần chỉ dùng [CLS] token (1 vector)
      - CNN+PhoBERT dùng TOÀN BỘ sequence → phong phú hơn
      - CNN bắt được cụm từ quan trọng cục bộ
        VD: "rất hay", "không hiểu", "khó quá"
    """
    def __init__(self, num_classes, num_filters, kernel_sizes, dropout):
        super().__init__()

        # PhoBERT backbone
        self.phobert = AutoModel.from_pretrained(MODEL_NAME)

        # CNN layers — mỗi kernel size 1 Conv1d
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=768,      # PhoBERT hidden size
                    out_channels=num_filters,
                    kernel_size=k,
                    padding=k // 2
                ),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for k in kernel_sizes
        ])

        # Dropout + Classifier
        self.dropout    = nn.Dropout(dropout)
        cnn_output_dim  = num_filters * len(kernel_sizes)  # 256 * 3 = 768
        self.classifier = nn.Sequential(
            nn.Linear(cnn_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        # PhoBERT: lấy TOÀN BỘ hidden states (không chỉ [CLS])
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # sequence_output: [batch, seq_len, 768]
        sequence_output = outputs.last_hidden_state

        # CNN cần [batch, channels, seq_len]
        x = sequence_output.permute(0, 2, 1)

        # Áp dụng từng CNN kernel → MaxPool
        pooled_outputs = []
        for conv in self.convs:
            conv_out = conv(x)                          # [batch, num_filters, seq_len]
            pooled   = torch.max(conv_out, dim=2)[0]   # [batch, num_filters] — max pooling
            pooled_outputs.append(pooled)

        # Concatenate: [batch, num_filters * 3]
        x_cat = torch.cat(pooled_outputs, dim=1)
        x_cat = self.dropout(x_cat)

        return self.classifier(x_cat)


# Khởi tạo model
model = CNNPhoBERT(
    num_classes  = NUM_CLASSES,
    num_filters  = NUM_FILTERS,
    kernel_sizes = KERNEL_SIZES,
    dropout      = DROPOUT
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Tổng tham số   : {total_params:,}")
print(f"  CNN filters    : {NUM_FILTERS} × {KERNEL_SIZES}")
print(f"  CNN output dim : {NUM_FILTERS * len(KERNEL_SIZES)}")
print(f"  Kiến trúc: PhoBERT → CNN(2,3,4) → MaxPool → Concat → Linear(3)")

# ------------------------------------------------------------
# 5. LOSS VỚI CLASS WEIGHTS
# ------------------------------------------------------------
label_counts = [0, 0, 0]
for item in cache_train:
    label_counts[item["label"]] += 1
total_samples = sum(label_counts)
weights = [total_samples / (3 * c) for c in label_counts]
weight_tensor = torch.tensor(weights, dtype=torch.float).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weight_tensor)

print(f"\n  Class weights: {[round(w,3) for w in weights]}")

# ------------------------------------------------------------
# 6. OPTIMIZER & SCHEDULER
# ------------------------------------------------------------
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {"params": [p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)],
     "weight_decay": 0.01},
    {"params": [p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)],
     "weight_decay": 0.0},
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LR)

total_steps  = len(train_loader) * NUM_EPOCHS
warmup_steps = int(total_steps * 0.1)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# ------------------------------------------------------------
# 7. HÀM TRAIN & EVALUATE
# ------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for i, batch in enumerate(loader):
        ids    = batch["input_ids"].to(DEVICE)
        mask   = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(ids, mask)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

        if (i + 1) % 100 == 0:
            print(f"    Batch {i+1:4d}/{len(loader)} | "
                  f"Loss: {total_loss/(i+1):.4f} | "
                  f"Acc: {correct/total:.4f}")

    return total_loss / len(loader), correct / total


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0
    with torch.no_grad():
        for batch in loader:
            ids    = batch["input_ids"].to(DEVICE)
            mask   = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits = model(ids, mask)
            total_loss += criterion(logits, labels).item()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    f1  = f1_score(all_labels, all_preds, average="macro")
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc, f1, all_labels, all_preds

# ------------------------------------------------------------
# 8. TRAINING LOOP
# ------------------------------------------------------------
print("\n[3/6] Bắt đầu training CNN+PhoBERT...")
print("  " + "=" * 45)

best_f1    = 0
no_improve = 0
best_path  = f"{MODEL_DIR}/CNN_PhoBERT_best.pt"
history    = {"train_loss":[], "val_loss":[], "val_f1":[], "val_acc":[]}

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\n  Epoch {epoch}/{NUM_EPOCHS}")
    print(f"  {'-'*40}")

    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler)
    val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_f1"].append(val_f1)
    history["val_acc"].append(val_acc)

    if val_f1 > best_f1:
        best_f1    = val_f1
        no_improve = 0
        torch.save(model.state_dict(), best_path)
        mark = " ← best ✓"
    else:
        no_improve += 1
        mark = f" (no improve {no_improve}/{PATIENCE})"

    print(f"\n  → Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  → Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc:.4f} | Val F1: {val_f1:.4f}{mark}")

    # Early stopping
    if no_improve >= PATIENCE:
        print(f"\n  ⏹️  Early stopping tại epoch {epoch}")
        break

print(f"\n  Best Val F1: {best_f1:.4f} → {best_path}")

# ------------------------------------------------------------
# 9. ĐÁNH GIÁ TRÊN TEST SET
# ------------------------------------------------------------
print("\n[4/6] Đánh giá trên Test Set...")
label_names = ["Tiêu cực", "Trung tính", "Tích cực"]

model.load_state_dict(torch.load(best_path, map_location=DEVICE))
_, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader)

print(f"\n  {'='*45}")
print(f"  KẾT QUẢ CNN+PHOBERT TRÊN TEST SET")
print(f"  {'='*45}")
print(f"  Accuracy : {test_acc:.4f}")
print(f"  F1-macro : {test_f1:.4f}")
print(classification_report(y_true, y_pred,
      target_names=label_names, digits=4, zero_division=0))

# ------------------------------------------------------------
# 10. VẼ LEARNING CURVE
# ------------------------------------------------------------
print("\n[5/6] Vẽ Learning Curve...")

epochs_range = range(1, len(history["train_loss"]) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Learning Curve — CNN + PhoBERT", fontsize=13)

ax1.plot(epochs_range, history["train_loss"], "o-", label="Train Loss", color="#378ADD")
ax1.plot(epochs_range, history["val_loss"],   "s--",label="Val Loss",   color="#E24B4A")
ax1.set_title("Loss theo Epoch"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.legend(); ax1.grid(alpha=0.3); ax1.spines[["top","right"]].set_visible(False)

ax2.plot(epochs_range, history["val_f1"],  "^-", label="Val F1-macro", color="#1D9E75")
ax2.plot(epochs_range, history["val_acc"], "D--",label="Val Accuracy", color="#BA7517")
ax2.set_title("F1 & Accuracy theo Epoch"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score")
ax2.legend(); ax2.grid(alpha=0.3); ax2.spines[["top","right"]].set_visible(False)

plt.tight_layout()
curve_path = f"{OUTPUT_DIR}/learning_curve_cnn_phobert.png"
plt.savefig(curve_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Đã lưu: {curve_path}")

# ------------------------------------------------------------
# 11. CẬP NHẬT BẢNG KẾT QUẢ
# ------------------------------------------------------------
print("\n[6/6] Cập nhật bảng kết quả...")

result_path = f"{OUTPUT_DIR}/results_baseline.json"
if os.path.exists(result_path):
    with open(result_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)
else:
    all_results = {}

all_results["CNN+PhoBERT"] = {
    "accuracy": round(test_acc, 4),
    "f1_macro": round(test_f1, 4)
}

with open(result_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

# In bảng so sánh
print(f"\n  {'='*55}")
print(f"  BẢNG SO SÁNH TẤT CẢ MÔ HÌNH")
print(f"  {'='*55}")
print(f"  {'Mô hình':<14} {'Accuracy':>10} {'F1-macro':>10}")
print(f"  {'-'*36}")
for name, r in all_results.items():
    marker = " ← best" if r['f1_macro'] == max(v['f1_macro'] for v in all_results.values()) else ""
    print(f"  {name:<14} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f}{marker}")

print("\n" + "=" * 55)
print("BƯỚC 5B HOÀN THÀNH!")
print("=" * 55)
print("Files đã tạo:")
print(f"  models/CNN_PhoBERT_best.pt")
print(f"  outputs/learning_curve_cnn_phobert.png")
print(f"  outputs/results_baseline.json (đã cập nhật)")
print("\nBước tiếp theo: step6_demo_web.py — Demo web!")