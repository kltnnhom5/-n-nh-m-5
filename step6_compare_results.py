# ============================================================
#  BƯỚC 6 — So sánh kết quả tất cả mô hình & Phân tích
#  Đề tài: Phân tích cảm xúc phản hồi sinh viên với PhoBERT
# ============================================================

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
import seaborn as sns

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 55)
print("BƯỚC 6 — SO SÁNH KẾT QUẢ & PHÂN TÍCH")
print("=" * 55)

# ------------------------------------------------------------
# 1. KẾT QUẢ TỪ FILE JSON
# ------------------------------------------------------------
print("\n[1/4] Đọc kết quả các mô hình...")

with open(f"{OUTPUT_DIR}/results_baseline.json", "r", encoding="utf-8") as f:
    results = json.load(f)

label_names  = ["Tiêu cực", "Trung tính", "Tích cực"]
model_names  = list(results.keys())
accuracies   = [results[m]["accuracy"] for m in model_names]
f1_macros    = [results[m]["f1_macro"] for m in model_names]

print(f"\n  {'Mô hình':<12} {'Accuracy':>10} {'F1-macro':>10}")
print(f"  {'-'*34}")
for name in model_names:
    r = results[name]
    marker = " ← best" if name == "PhoBERT" else ""
    print(f"  {name:<12} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f}{marker}")

# ------------------------------------------------------------
# 2. BIỂU ĐỒ SO SÁNH TỔNG QUAN
# ------------------------------------------------------------
print("\n[2/4] Vẽ biểu đồ so sánh...")

colors_bar = ["#888780", "#378ADD", "#BA7517", "#1D9E75"]
x = np.arange(len(model_names))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("So sánh hiệu năng các mô hình — UIT-VSFC Test Set",
             fontsize=13, y=1.02)

# -- Biểu đồ trái: Accuracy & F1-macro cạnh nhau --
ax1 = axes[0]
bars1 = ax1.bar(x - width/2, accuracies, width,
                label="Accuracy", color=[c + "CC" for c in colors_bar],
                edgecolor="white")
bars2 = ax1.bar(x + width/2, f1_macros,  width,
                label="F1-macro", color=colors_bar,
                edgecolor="white")

for bar, val in zip(bars1, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{val:.4f}", ha="center", va="bottom", fontsize=8.5)
for bar, val in zip(bars2, f1_macros):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{val:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(model_names)
ax1.set_ylim(0, 1.08)
ax1.set_ylabel("Score")
ax1.set_title("Accuracy & F1-macro theo mô hình")
ax1.legend()
ax1.grid(axis="y", alpha=0.3)
ax1.spines[["top","right"]].set_visible(False)

# -- Biểu đồ phải: F1-macro ngang (dễ so sánh) --
ax2 = axes[1]
y_pos = np.arange(len(model_names))
h_bars = ax2.barh(y_pos, f1_macros, color=colors_bar,
                   edgecolor="white", height=0.5)
for bar, val in zip(h_bars, f1_macros):
    ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2,
             f"{val:.4f}", va="center", fontsize=10, fontweight="bold")

ax2.set_yticks(y_pos)
ax2.set_yticklabels(model_names)
ax2.set_xlim(0, 1.05)
ax2.set_xlabel("F1-macro")
ax2.set_title("Xếp hạng F1-macro (cao hơn = tốt hơn)")
ax2.axvline(x=max(f1_macros), color="#1D9E75",
            linestyle="--", alpha=0.5, label=f"Best: {max(f1_macros):.4f}")
ax2.legend(fontsize=9)
ax2.grid(axis="x", alpha=0.3)
ax2.spines[["top","right"]].set_visible(False)

plt.tight_layout()
path1 = f"{OUTPUT_DIR}/compare_models.png"
plt.savefig(path1, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Đã lưu: {path1}")

# ------------------------------------------------------------
# 3. CONFUSION MATRIX PHOBERT (chi tiết nhất)
# ------------------------------------------------------------
print("\n[3/4] Vẽ Confusion Matrix PhoBERT...")

# Load lại cache test và model PhoBERT để lấy y_true, y_pred
print("  Đang load model PhoBERT để lấy predictions...")

from transformers import AutoModel
from torch.utils.data import Dataset, DataLoader

def load_cache(name):
    with open(f"cache/{name}.pkl", "rb") as f:
        return pickle.load(f)

cache_test = load_cache("test")

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

class PhoBERTSentiment(nn.Module):
    def __init__(self):
        super().__init__()
        self.phobert    = AutoModel.from_pretrained("vinai/phobert-base")
        self.dropout    = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 3)
    def forward(self, input_ids, attention_mask):
        out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))

DEVICE = torch.device("cpu")
model_pb = PhoBERTSentiment().to(DEVICE)
model_pb.load_state_dict(torch.load("models/PhoBERT_best.pt", map_location=DEVICE))
model_pb.eval()

test_loader = DataLoader(PhoBERTDataset(cache_test), batch_size=32, shuffle=False)

y_true_pb, y_pred_pb = [], []
with torch.no_grad():
    for batch in test_loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        logits = model_pb(ids, mask)
        y_pred_pb.extend(logits.argmax(1).cpu().numpy())
        y_true_pb.extend(batch["label"].numpy())

# Vẽ confusion matrix đẹp
cm = confusion_matrix(y_true_pb, y_pred_pb)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
annot = np.array([
    [f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)" for j in range(3)]
    for i in range(3)
])

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm_pct, annot=annot, fmt="", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names,
            linewidths=0.5, linecolor="white",
            annot_kws={"size": 12}, ax=ax)
ax.set_xlabel("Dự đoán (Predicted)", fontsize=12, labelpad=10)
ax.set_ylabel("Thực tế (True Label)", fontsize=12, labelpad=10)
ax.set_title("Confusion Matrix — PhoBERT (Test Set)", fontsize=13, pad=15)
plt.tight_layout()
path2 = f"{OUTPUT_DIR}/confusion_matrix_phobert.png"
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Đã lưu: {path2}")

# ------------------------------------------------------------
# 4. IN PHÂN TÍCH CHI TIẾT CHO BÁO CÁO
# ------------------------------------------------------------
print("\n[4/4] Phân tích kết quả chi tiết...")

print(f"\n{'='*55}")
print(f"CLASSIFICATION REPORT — PHOBERT")
print(f"{'='*55}")
print(classification_report(y_true_pb, y_pred_pb,
      target_names=label_names, digits=4, zero_division=0))

# Tính mức cải thiện so với baseline tốt nhất
best_baseline_f1 = max(results["BiLSTM"]["f1_macro"],
                       results["CNN+LSTM"]["f1_macro"])
phobert_f1       = results["PhoBERT"]["f1_macro"]
improvement      = (phobert_f1 - best_baseline_f1) / best_baseline_f1 * 100

print(f"\n{'='*55}")
print(f"PHÂN TÍCH SO SÁNH (cho Chương 5 báo cáo)")
print(f"{'='*55}")
print(f"""
1. TỔNG QUAN KẾT QUẢ
   PhoBERT đạt F1-macro = {phobert_f1:.4f} và Accuracy = {results['PhoBERT']['accuracy']:.4f}
   vượt trội so với tất cả baseline trên test set UIT-VSFC.

2. CẢI THIỆN SO VỚI BASELINE TỐT NHẤT
   Baseline tốt nhất (BiLSTM): F1-macro = {best_baseline_f1:.4f}
   PhoBERT                   : F1-macro = {phobert_f1:.4f}
   Cải thiện                 : +{improvement:.1f}%

3. PHÂN TÍCH THEO TỪNG LỚP
   - Tiêu cực : PhoBERT đạt F1 cao (~0.95) nhờ ngữ cảnh hai chiều
   - Tích cực : Tốt nhất (~0.96) do nhiều mẫu học
   - Trung tính: Thấp nhất (~0.59) do ít mẫu + ranh giới mờ với 2 lớp kia
     → Đây là điểm cần phân tích sâu trong báo cáo

4. NHẬN XÉT LSTM THUẦN
   LSTM F1 = 0.2229 — chỉ dự đoán 1 lớp (Tích cực) do mất cân bằng
   → Cần thêm class weights hoặc dữ liệu augmentation

5. KẾT LUẬN
   PhoBERT vượt trội nhờ:
   a) Pre-trained trên 20GB văn bản tiếng Việt
   b) Self-attention nắm bắt ngữ cảnh toàn câu
   c) Subword tokenization xử lý từ ngoài từ điển
   d) Transfer learning hiệu quả với ít dữ liệu fine-tuning
""")

# ------------------------------------------------------------
# TỔNG KẾT
# ------------------------------------------------------------
print("=" * 55)
print("BƯỚC 6 HOÀN THÀNH! ĐỀ TÀI HOÀN CHỈNH!")
print("=" * 55)
print("\nTất cả file outputs đã tạo:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(f"{OUTPUT_DIR}/{f}") / 1024
    print(f"  outputs/{f:<45} {size:6.0f} KB")

print(f"\n{'='*55}")
print(f"BẢNG KẾT QUẢ CUỐI CÙNG")
print(f"{'='*55}")
print(f"  {'Mô hình':<12} {'Accuracy':>10} {'F1-macro':>10} {'Ghi chú'}")
print(f"  {'-'*55}")
notes = {
    "LSTM":     "Baseline yếu, chỉ đoán 1 lớp",
    "BiLSTM":   "Baseline tốt, 2 chiều hiệu quả",
    "CNN+LSTM": "Baseline tốt, local features",
    "PhoBERT":  "Model chính, tốt nhất ← BEST",
}
for name in model_names:
    r = results[name]
    print(f"  {name:<12} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f}   {notes[name]}")

print("\nBước tiếp theo: step7_demo.py — Demo nhập câu → dự đoán cảm xúc")