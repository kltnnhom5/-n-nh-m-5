# ============================================================
#  BƯỚC 1 — Khám phá Dataset UIT-VSFC
#  Đề tài: Phân tích cảm xúc phản hồi sinh viên với PhoBERT
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from datasets import load_dataset

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ------------------------------------------------------------
# 1. TẠO THƯ MỤC ĐẦU RA
# ------------------------------------------------------------
os.makedirs("data",    exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ------------------------------------------------------------
# 2. LOAD DATASET TỪ HUGGINGFACE
# ------------------------------------------------------------
print("Đang tải dataset UIT-VSFC từ HuggingFace...")
ds = load_dataset("uitnlp/vietnamese_students_feedback")
print("Tải xong!\n")

df_train = ds["train"].to_pandas()
df_val   = ds["validation"].to_pandas()
df_test  = ds["test"].to_pandas()

print(f"Train: {len(df_train)} | Validation: {len(df_val)} | Test: {len(df_test)}")

# ------------------------------------------------------------
# 3. TIỀN XỬ LÝ
# ------------------------------------------------------------
label_map = {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"}

for df in [df_train, df_val, df_test]:
    df["label_name"] = df["sentiment"].map(label_map)
    df["word_count"] = df["sentence"].apply(lambda x: len(str(x).split()))

# ------------------------------------------------------------
# 4. THÔNG TIN CƠ BẢN
# ------------------------------------------------------------
print("\n" + "="*55)
print("THÔNG TIN CƠ BẢN")
print("="*55)
print(f"Train      : {len(df_train):,} câu")
print(f"Validation : {len(df_val):,} câu")
print(f"Test       : {len(df_test):,} câu")
print(f"Tổng cộng  : {len(df_train)+len(df_val)+len(df_test):,} câu")

print("\n5 dòng đầu tiên (train):")
print(df_train[["sentence","sentiment","label_name","word_count"]].head(5).to_string(index=False))

# ------------------------------------------------------------
# 5. PHÂN PHỐI NHÃN
# ------------------------------------------------------------
print("\n" + "="*55)
print("PHÂN PHỐI NHÃN (TRAIN SET)")
print("="*55)
counts = df_train["label_name"].value_counts()
total  = len(df_train)
for name, count in counts.items():
    bar = "█" * int(count / total * 40)
    print(f"  {name:12s} {count:5,}  ({count/total*100:.1f}%)  {bar}")

# ------------------------------------------------------------
# 6. THỐNG KÊ ĐỘ DÀI CÂU
# ------------------------------------------------------------
print("\n" + "="*55)
print("THỐNG KÊ ĐỘ DÀI CÂU (số từ, train set)")
print("="*55)
wc = df_train["word_count"]
print(f"  Trung bình  : {wc.mean():.1f} từ")
print(f"  Trung vị    : {wc.median():.0f} từ")
print(f"  Ngắn nhất   : {wc.min()} từ")
print(f"  Dài nhất    : {wc.max()} từ")
print(f"  Phần vị 95% : {int(wc.quantile(0.95))} từ  (max_length=256 là đủ)")

print("\nĐộ dài trung bình theo nhãn:")
for name in ["Tiêu cực", "Trung tính", "Tích cực"]:
    avg = df_train[df_train["label_name"]==name]["word_count"].mean()
    print(f"  {name:12s}: {avg:.1f} từ")

# ------------------------------------------------------------
# 7. KIỂM TRA CHẤT LƯỢNG DỮ LIỆU
# ------------------------------------------------------------
print("\n" + "="*55)
print("KIỂM TRA CHẤT LƯỢNG DỮ LIỆU")
print("="*55)
nulls  = df_train[["sentence","sentiment"]].isnull().sum()
dupes  = df_train.duplicated(subset=["sentence"]).sum()
shorts = (df_train["word_count"] < 3).sum()
longs  = (df_train["word_count"] > 100).sum()
print(f"  Giá trị null    : {nulls.sum()}")
print(f"  Câu trùng lặp   : {dupes}")
print(f"  Câu quá ngắn (<3 từ)  : {shorts}")
print(f"  Câu rất dài (>100 từ) : {longs}")

# ------------------------------------------------------------
# 8. VÍ DỤ CÂU TỪNG NHÃN
# ------------------------------------------------------------
print("\n" + "="*55)
print("VÍ DỤ CÂU TỪNG NHÃN")
print("="*55)
for label_id in [2, 1, 0]:
    name = label_map[label_id]
    samples = df_train[df_train["sentiment"]==label_id]["sentence"].sample(3, random_state=42)
    print(f"\n[{name}]")
    for s in samples:
        print(f"  · {s}")

# ------------------------------------------------------------
# 9. LƯU FILE CSV
# ------------------------------------------------------------
print("\n" + "="*55)
print("LƯU FILE CSV")
print("="*55)
df_train.to_csv("data/train.csv", index=False, encoding="utf-8-sig")
df_val.to_csv("data/val.csv",     index=False, encoding="utf-8-sig")
df_test.to_csv("data/test.csv",   index=False, encoding="utf-8-sig")

for split, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
    size_kb = os.path.getsize(f"data/{split}.csv") / 1024
    print(f"  data/{split}.csv — {len(df):,} dòng — {size_kb:.0f} KB")

# ------------------------------------------------------------
# 10. VẼ BIỂU ĐỒ PHÂN TÍCH
# ------------------------------------------------------------
print("\nĐang vẽ biểu đồ...")

colors = ["#E24B4A", "#378ADD", "#1D9E75"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Phân tích Dataset UIT-VSFC — Vietnamese Students Feedback Corpus",
             fontsize=13, y=1.02)

# Biểu đồ 1: Phân phối nhãn
ax1 = axes[0]
order_names  = ["Tiêu cực", "Trung tính", "Tích cực"]
order_colors = colors
order_counts = [df_train[df_train["label_name"]==n].shape[0] for n in order_names]

bars1 = ax1.bar(order_names, order_counts, color=order_colors,
                width=0.5, edgecolor="white")
for bar, val in zip(bars1, order_counts):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 60,
             f"{val:,}\n({val/total*100:.1f}%)",
             ha="center", va="bottom", fontsize=9.5)
ax1.set_title("Phân phối nhãn cảm xúc (train)")
ax1.set_ylabel("Số câu")
ax1.set_ylim(0, 8500)
ax1.grid(axis="y", alpha=0.25)
ax1.spines[["top","right"]].set_visible(False)

# Biểu đồ 2: Số mẫu theo tập
ax2 = axes[1]
split_names  = ["Train", "Validation", "Test"]
split_sizes  = [len(df_train), len(df_val), len(df_test)]
split_colors = ["#534AB7", "#BA7517", "#993556"]

bars2 = ax2.bar(split_names, split_sizes, color=split_colors,
                width=0.45, edgecolor="white")
for bar, val in zip(bars2, split_sizes):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 100,
             f"{val:,}", ha="center", va="bottom", fontsize=10)
ax2.set_title("Số mẫu theo tập dữ liệu")
ax2.set_ylabel("Số câu")
ax2.set_ylim(0, 14500)
ax2.grid(axis="y", alpha=0.25)
ax2.spines[["top","right"]].set_visible(False)

# -- Biểu đồ 3: Phân phối độ dài câu --
ax3 = axes[2]
for label_id, label_name, color in zip([0, 1, 2], order_names, colors):
    subset = df_train[df_train["sentiment"]==label_id]["word_count"]
    ax3.hist(subset, bins=30, alpha=0.6,
             label=label_name, color=color, edgecolor="white")
ax3.set_title("Phân phối độ dài câu theo nhãn")
ax3.set_xlabel("Số từ")
ax3.set_ylabel("Số câu")
ax3.legend(fontsize=9)
ax3.grid(axis="y", alpha=0.25)
ax3.spines[["top","right"]].set_visible(False)

plt.tight_layout()
save_path = "outputs/vsfc_analysis.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Biểu đồ đã lưu: {save_path}")

# ------------------------------------------------------------
# 11. BẢNG TỔNG KẾT (đưa vào báo cáo)
# ------------------------------------------------------------
print("\n" + "="*55)
print("BẢNG TỔNG KẾT (đưa vào báo cáo Chương 3)")
print("="*55)

rows = []
for split_name, df in [("Train", df_train), ("Validation", df_val), ("Test", df_test)]:
    c = df["label_name"].value_counts()
    n = len(df)
    rows.append({
        "Tập"         : split_name,
        "Tổng câu"    : n,
        "Tích cực"    : f'{c.get("Tích cực",0):,} ({c.get("Tích cực",0)/n*100:.1f}%)',
        "Trung tính"  : f'{c.get("Trung tính",0):,} ({c.get("Trung tính",0)/n*100:.1f}%)',
        "Tiêu cực"    : f'{c.get("Tiêu cực",0):,} ({c.get("Tiêu cực",0)/n*100:.1f}%)',
        "Dài TB (từ)" : f'{df["word_count"].mean():.1f}',
    })

summary_df = pd.DataFrame(rows)
print(summary_df.to_string(index=False))
summary_df.to_csv("outputs/bang_thong_ke.csv", index=False, encoding="utf-8-sig")
print("\n  outputs/bang_thong_ke.csv đã lưu")

print("\n" + "="*55)
print("BƯỚC 1 HOÀN THÀNH!")
print("="*55)
print("Các file đã tạo:")
print("  data/train.csv")
print("  data/val.csv")
print("  data/test.csv")
print("  outputs/vsfc_analysis.png")
print("  outputs/bang_thong_ke.csv")

