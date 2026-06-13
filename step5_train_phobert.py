# ============================================================
#  BƯỚC 5 — Fine-tuning PhoBERT (Model Chính)
#  Đề tài: Phân tích cảm xúc phản hồi sinh viên với PhoBERT
# ============================================================

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score, accuracy_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ------------------------------------------------------------
# 1. CẤU HÌNH
# ------------------------------------------------------------
CACHE_DIR  = "cache"
MODEL_DIR  = "models"
OUTPUT_DIR = "outputs"

MODEL_NAME  = "vinai/phobert-base"
NUM_CLASSES = 3
BATCH_SIZE  = 16      # nhỏ hơn vì PhoBERT nặng hơn
NUM_EPOCHS  = 5       # PhoBERT hội tụ nhanh, không cần nhiều epoch
LR          = 2e-5    # learning rate nhỏ cho fine-tuning
DROPOUT     = 0.3
WARMUP_RATIO = 0.1    # 10% đầu dùng warmup

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 55)
print("BƯỚC 5 — FINE-TUNING PHOBERT (MODEL CHÍNH)")
print("=" * 55)
print(f"Model     : {MODEL_NAME}")
print(f"Thiết bị  : {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs    : {NUM_EPOCHS}")
print(f"LR        : {LR}")
print()
if DEVICE.type == "cpu":
    print("  ⚠️  Đang chạy trên CPU — mỗi epoch mất ~30–60 phút")
    print("  ⚠️  Tổng thời gian ước tính: 2–5 giờ")
    print("  💡 Để nhanh hơn: dùng Google Colab (GPU miễn phí)")

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

print(f"  Train: {len(cache_train):,} | Val: {len(cache_val):,} | Test: {len(cache_test):,}")

# ------------------------------------------------------------
# 3. DATASET & DATALOADER
# ------------------------------------------------------------
print("\n[2/6] Tạo Dataset và DataLoader...")

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

train_dataset = PhoBERTDataset(cache_train)
val_dataset   = PhoBERTDataset(cache_val)
test_dataset  = PhoBERTDataset(cache_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"  Train batch: {len(train_loader)} | Val batch: {len(val_loader)}")

# ------------------------------------------------------------
# 4. KIẾN TRÚC PHOBERT
# ------------------------------------------------------------
print("\n[3/6] Load PhoBERT và định nghĩa kiến trúc...")

class PhoBERTSentiment(nn.Module):
    """
    Kiến trúc fine-tuning PhoBERT:
      PhoBERT (12 transformer layers) → [CLS] token → Dropout → Linear(768→3)

    [CLS] token chứa toàn bộ thông tin ngữ nghĩa của câu
    sau khi đi qua 12 lớp self-attention của PhoBERT
    """
    def __init__(self, model_name, num_classes, dropout):
        super().__init__()
        self.phobert    = AutoModel.from_pretrained(model_name)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)  # 768 = hidden size PhoBERT-base

    def forward(self, input_ids, attention_mask):
        # outputs.last_hidden_state: [batch, seq_len, 768]
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Lấy [CLS] token — vị trí 0
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        cls_output = self.dropout(cls_output)
        return self.classifier(cls_output)               # [batch, 3]


print("  Đang tải PhoBERT weights...")
model = PhoBERTSentiment(MODEL_NAME, NUM_CLASSES, DROPOUT).to(DEVICE)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Tổng tham số    : {total_params:,}")
print(f"  Trainable params: {trainable_params:,}")
print(f"  Kiến trúc: PhoBERT-base → [CLS] → Dropout({DROPOUT}) → Linear(768→{NUM_CLASSES})")

# ------------------------------------------------------------
# 5. OPTIMIZER & SCHEDULER & LOSS
# ------------------------------------------------------------
print("\n[4/6] Cấu hình optimizer và loss...")

# Class weights xử lý mất cân bằng
label_counts = [0, 0, 0]
for item in cache_train:
    label_counts[item["label"]] += 1
total_samples = sum(label_counts)
weights = [total_samples / (3 * c) for c in label_counts]
weight_tensor = torch.tensor(weights, dtype=torch.float).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weight_tensor)

print(f"  Class weights: {[round(w, 3) for w in weights]}")

# AdamW — optimizer chuẩn cho BERT fine-tuning
# Phân biệt: weight decay KHÔNG áp dụng cho bias và LayerNorm
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters()
                   if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,
    },
    {
        "params": [p for n, p in model.named_parameters()
                   if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LR)

# Linear warmup scheduler — chuẩn cho BERT
total_steps  = len(train_loader) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
print(f"  Optimizer : AdamW (lr={LR}, weight_decay=0.01)")
print(f"  Scheduler : Linear warmup ({warmup_steps} steps) → decay")
print(f"  Total steps: {total_steps}")

# ------------------------------------------------------------
# 6. HÀM TRAIN & EVALUATE
# ------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for i, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping — tránh exploding gradient
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

        # In tiến trình mỗi 50 batch
        if (i + 1) % 50 == 0:
            print(f"    Batch {i+1:4d}/{len(loader)} | "
                  f"Loss: {total_loss/(i+1):.4f} | "
                  f"Acc: {correct/total:.4f}")

    return total_loss / len(loader), correct / total


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)
            logits         = model(input_ids, attention_mask)
            total_loss    += criterion(logits, labels).item()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1  = f1_score(all_labels, all_preds, average="macro")
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc, f1, all_labels, all_preds

# ------------------------------------------------------------
# 7. TRAINING LOOP
# ------------------------------------------------------------
print("\n[5/6] Bắt đầu Fine-tuning PhoBERT...")
print("  " + "=" * 45)

best_f1   = 0
best_path = f"{MODEL_DIR}/PhoBERT_best.pt"
history   = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}

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
        best_f1 = val_f1
        torch.save(model.state_dict(), best_path)
        mark = " ← best ✓"
    else:
        mark = ""

    print(f"\n  → Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  → Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc:.4f} | Val F1: {val_f1:.4f}{mark}")

print(f"\n  Best Val F1: {best_f1:.4f} → {best_path}")

# ------------------------------------------------------------
# 8. ĐÁNH GIÁ TRÊN TEST SET
# ------------------------------------------------------------
print("\n[6/6] Đánh giá PhoBERT trên Test Set...")
label_names = ["Tiêu cực", "Trung tính", "Tích cực"]

model.load_state_dict(torch.load(best_path, map_location=DEVICE))
_, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader)

print(f"\n  {'='*45}")
print(f"  KẾT QUẢ PHOBERT TRÊN TEST SET")
print(f"  {'='*45}")
print(f"  Accuracy  : {test_acc:.4f}")
print(f"  F1-macro  : {test_f1:.4f}")
print()
print(classification_report(y_true, y_pred,
      target_names=label_names, digits=4,
      zero_division=0))

# ------------------------------------------------------------
# 9. VẼ LEARNING CURVE
# ------------------------------------------------------------
print("Vẽ biểu đồ Learning Curve PhoBERT...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Learning Curve — PhoBERT Fine-tuning", fontsize=13)
epochs_range = range(1, NUM_EPOCHS + 1)

ax1.plot(epochs_range, history["train_loss"], "o-", label="Train Loss", color="#378ADD")
ax1.plot(epochs_range, history["val_loss"],   "s--",label="Val Loss",   color="#E24B4A")
ax1.set_title("Loss theo Epoch")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.legend(); ax1.grid(alpha=0.3)
ax1.spines[["top","right"]].set_visible(False)

ax2.plot(epochs_range, history["val_f1"],  "^-", label="Val F1-macro", color="#1D9E75")
ax2.plot(epochs_range, history["val_acc"], "D--",label="Val Accuracy", color="#BA7517")
ax2.set_title("F1 & Accuracy theo Epoch")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score")
ax2.legend(); ax2.grid(alpha=0.3)
ax2.spines[["top","right"]].set_visible(False)

plt.tight_layout()
curve_path = f"{OUTPUT_DIR}/learning_curve_phobert.png"
plt.savefig(curve_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Đã lưu: {curve_path}")

# ------------------------------------------------------------
# 10. CẬP NHẬT VÀ IN BẢNG SO SÁNH ĐẦY ĐỦ
# ------------------------------------------------------------
result_path = f"{OUTPUT_DIR}/results_baseline.json"
if os.path.exists(result_path):
    with open(result_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)
else:
    all_results = {}

all_results["PhoBERT"] = {
    "accuracy": round(test_acc, 4),
    "f1_macro": round(test_f1, 4)
}

with open(result_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

# ------------------------------------------------------------
# TỔNG KẾT
# ------------------------------------------------------------
print("\n" + "=" * 55)
print("BƯỚC 5 HOÀN THÀNH!")
print("=" * 55)
print("Các file đã tạo:")
print(f"  models/PhoBERT_best.pt")
print(f"  outputs/learning_curve_phobert.png")
print(f"  outputs/results_baseline.json  (đã cập nhật)")

print(f"\n{'='*55}")
print(f"BẢNG SO SÁNH TẤT CẢ MÔ HÌNH")
print(f"{'='*55}")
print(f"  {'Mô hình':<12} {'Accuracy':>10} {'F1-macro':>10}")
print(f"  {'-'*34}")
for name, r in all_results.items():
    marker = " ← best" if name == "PhoBERT" else ""
    print(f"  {name:<12} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f}{marker}")

print(f"\nBước tiếp theo: step6_compare_results.py — Vẽ biểu đồ so sánh!")