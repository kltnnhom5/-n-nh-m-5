# ============================================================
#  BƯỚC 2B — Thêm dữ liệu mới vào dataset và tạo cache mới
#  Chạy SAU step2_preprocess.py
#  File cần có: data/train.csv, cache/train.pkl
#               Dataset_TuViet_GopChung.xlsx (557 câu mới)
# ============================================================

import os
import re
import time
import pickle
import unicodedata

import pandas as pd
from tqdm import tqdm
from pyvi import ViTokenizer
from transformers import AutoTokenizer

# ------------------------------------------------------------
# 1. CẤU HÌNH
# ------------------------------------------------------------
MAX_LENGTH     = 256
MODEL_NAME     = "vinai/phobert-base"
DATA_DIR       = "data"
CACHE_DIR      = "cache"
NEW_DATA_FILE  = "Dataset_TuViet_GopChung.xlsx"   # file 557 câu

os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

print("=" * 55)
print("BƯỚC 2B — THÊM DỮ LIỆU MỚI VÀO DATASET")
print("=" * 55)

# ------------------------------------------------------------
# 2. ĐỌC FILE EXCEL MỚI
# ------------------------------------------------------------
print("\n[1/6] Đọc file dữ liệu mới...")

df_new = pd.read_excel(
    NEW_DATA_FILE,
    sheet_name=0,
    skiprows=3,          # bỏ 3 dòng header
    usecols=[1, 2],      # cột B=sentence, C=sentiment
    names=["sentence", "sentiment"]
)

# Làm sạch
df_new = df_new.dropna(subset=["sentence", "sentiment"])
df_new = df_new[df_new["sentence"].astype(str).str.strip() != ""]
df_new["sentiment"] = df_new["sentiment"].astype(int)
df_new["sentence"]  = df_new["sentence"].astype(str).str.strip()

print(f"  Tổng câu mới đọc được: {len(df_new)}")
print(f"  Tích cực(2): {(df_new['sentiment']==2).sum()}")
print(f"  Tiêu cực(0): {(df_new['sentiment']==0).sum()}")
print(f"  Trung tính(1): {(df_new['sentiment']==1).sum()}")

# ------------------------------------------------------------
# 3. ĐỌC DATASET GỐC VÀ GỘP
# ------------------------------------------------------------
print("\n[2/6] Đọc dataset gốc và gộp...")

df_train_old = pd.read_csv(
    f"{DATA_DIR}/train.csv", encoding="utf-8-sig"
)
print(f"  Train gốc  : {len(df_train_old):,} câu")

# Gộp
df_train_new = pd.concat(
    [df_train_old[["sentence","sentiment"]], df_new[["sentence","sentiment"]]],
    ignore_index=True
)

# Xóa câu trùng
before = len(df_train_new)
df_train_new = df_train_new.drop_duplicates(subset=["sentence"])
after  = len(df_train_new)
print(f"  Sau gộp    : {before:,} câu")
print(f"  Trùng lặp  : {before - after} câu (đã xóa)")
print(f"  Train mới  : {after:,} câu ← sẽ dùng để train")

print(f"\n  Phân phối nhãn (train mới):")
label_map = {0:"Tiêu cực", 1:"Trung tính", 2:"Tích cực"}
for lbl, name in [(2,"Tích cực"),(0,"Tiêu cực"),(1,"Trung tính")]:
    cnt = (df_train_new["sentiment"]==lbl).sum()
    pct = cnt / len(df_train_new) * 100
    bar = "█" * int(pct / 2.5)
    print(f"  {name:12s} {cnt:5,}  ({pct:.1f}%)  {bar}")

# Lưu file CSV mới
df_train_new.to_csv(
    f"{DATA_DIR}/train_extended.csv",
    index=False, encoding="utf-8-sig"
)
print(f"\n  Đã lưu: data/train_extended.csv")

# ------------------------------------------------------------
# 4. HÀM TIỀN XỬ LÝ
# ------------------------------------------------------------
print("\n[3/6] Khởi tạo tokenizer và hàm tiền xử lý...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(
        r"[^a-zA-Z0-9\s\.,!?;:\-"
        r"áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệ"
        r"íìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữự"
        r"ýỳỷỹỵđÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆ"
        r"ÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰ"
        r"ÝỲỶỸỴĐ]",
        " ", text
    )
    return re.sub(r"\s+", " ", text).strip().lower()

def preprocess_text(text: str) -> str:
    text = clean_text(text)
    text = ViTokenizer.tokenize(text)
    return text

print("  OK — tokenizer và hàm tiền xử lý đã sẵn sàng")

# ------------------------------------------------------------
# 5. TẠO CACHE MỚI CHỈ CHO PHẦN DỮ LIỆU MỚI
# ------------------------------------------------------------
print("\n[4/6] Tạo cache cho dữ liệu mới...")
print(f"  Xử lý {len(df_new)} câu mới — mất vài phút...")

cache_new = []
errors = 0
start  = time.time()

for idx, row in tqdm(df_new.iterrows(), total=len(df_new),
                     desc="  [new_data]", ncols=70):
    try:
        text = preprocess_text(str(row["sentence"]))
        enc  = tokenizer(
            text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
        )
        cache_new.append({
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label":          int(row["sentiment"]),
            "text_original":  str(row["sentence"]),
            "text_processed": text,
        })
    except Exception as e:
        errors += 1
        print(f"\n  Lỗi dòng {idx}: {e}")

elapsed = time.time() - start
print(f"  Xong: {len(cache_new)} mẫu | Lỗi: {errors} | {elapsed:.0f}s")

# ------------------------------------------------------------
# 6. GỘP CACHE CŨ + MỚI
# ------------------------------------------------------------
print("\n[5/6] Gộp cache cũ và cache mới...")

# Load cache train cũ
cache_path_old = f"{CACHE_DIR}/train.pkl"
if os.path.exists(cache_path_old):
    with open(cache_path_old, "rb") as f:
        cache_old = pickle.load(f)
    print(f"  Cache cũ : {len(cache_old):,} mẫu")
else:
    cache_old = []
    print("  Không tìm thấy cache cũ — dùng cache mới hoàn toàn")

# Gộp
cache_train_extended = cache_old + cache_new
print(f"  Cache mới: {len(cache_new):,} mẫu")
print(f"  Tổng gộp : {len(cache_train_extended):,} mẫu")

# Lưu cache train mới (ghi đè train.pkl)
cache_path_new = f"{CACHE_DIR}/train.pkl"
with open(cache_path_new, "wb") as f:
    pickle.dump(cache_train_extended, f)

size_mb = os.path.getsize(cache_path_new) / (1024 * 1024)
print(f"  Đã lưu: {cache_path_new} ({size_mb:.1f} MB)")

# Lưu thêm bản backup
cache_path_bak = f"{CACHE_DIR}/train_extended.pkl"
with open(cache_path_bak, "wb") as f:
    pickle.dump(cache_train_extended, f)
print(f"  Backup : {cache_path_bak}")

# ------------------------------------------------------------
# 7. KIỂM TRA KẾT QUẢ
# ------------------------------------------------------------
print("\n[6/6] Kiểm tra kết quả...")

label_counts = {0: 0, 1: 0, 2: 0}
for item in cache_train_extended:
    label_counts[item["label"]] += 1

total = len(cache_train_extended)
print(f"\n  Phân phối cache train mới:")
for lbl, name in [(2,"Tích cực"),(0,"Tiêu cực"),(1,"Trung tính")]:
    cnt = label_counts[lbl]
    pct = cnt / total * 100
    bar = "█" * int(pct / 2.5)
    print(f"  {name:12s} {cnt:5,}  ({pct:.1f}%)  {bar}")

# Tính class weights mới
print(f"\n  Class weights mới (cho CrossEntropyLoss):")
for lbl, name in [(0,"Tiêu cực"),(1,"Trung tính"),(2,"Tích cực")]:
    w = total / (3 * label_counts[lbl])
    print(f"  w_{name:10s} = {w:.4f}")

# Xem 2 ví dụ câu mới
print(f"\n  Ví dụ 2 câu mới được thêm vào:")
for item in cache_new[:2]:
    print(f"  Gốc   : {item['text_original']}")
    print(f"  Xử lý : {item['text_processed'][:60]}")
    print(f"  Nhãn  : {item['label']} ({label_map[item['label']]})")
    print()

# ------------------------------------------------------------
# TỔNG KẾT
# ------------------------------------------------------------
print("=" * 55)
print("BƯỚC 2B HOÀN THÀNH!")
print("=" * 55)
print(f"  data/train_extended.csv  — {after:,} câu (CSV)")
print(f"  cache/train.pkl          — {total:,} mẫu (cache đã gộp)")
print(f"  cache/train_extended.pkl — {total:,} mẫu (backup)")
print(f"\n  Dataset tăng từ 11.426 → {total:,} câu")
print(f"  (+{total - len(cache_old):,} câu mới, +{(total-len(cache_old))/len(cache_old)*100:.1f}%)")
print(f"\nBước tiếp theo: Chạy lại step3, step4, step5 với cache mới!")
print(f"  Cache train.pkl đã được cập nhật — không cần sửa code step3/4/5")