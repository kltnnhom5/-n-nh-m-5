# ============================================================
#  BƯỚC 2 — Tiền xử lý văn bản tiếng Việt cho PhoBERT
#  Đề tài: Phân tích cảm xúc phản hồi sinh viên với PhoBERT
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
MAX_LENGTH  = 256          # độ dài tối đa token
MODEL_NAME  = "vinai/phobert-base"
DATA_DIR    = "data"
CACHE_DIR   = "cache"
OUTPUT_DIR  = "outputs"

os.makedirs(CACHE_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 55)
print("BƯỚC 2 — TIỀN XỬ LÝ DỮ LIỆU")
print("=" * 55)

# ------------------------------------------------------------
# 2. ĐỌC FILE CSV (từ bước 1)
# ------------------------------------------------------------
print("\n[1/5] Đọc dữ liệu từ Bước 1...")

df_train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
df_val   = pd.read_csv(f"{DATA_DIR}/val.csv",   encoding="utf-8-sig")
df_test  = pd.read_csv(f"{DATA_DIR}/test.csv",  encoding="utf-8-sig")

print(f"  Train : {len(df_train):,} câu")
print(f"  Val   : {len(df_val):,} câu")
print(f"  Test  : {len(df_test):,} câu")

# ------------------------------------------------------------
# 3. HÀM LÀM SẠCH VĂN BẢN
# ------------------------------------------------------------
print("\n[2/5] Định nghĩa hàm làm sạch và tách từ...")

def clean_text(text: str) -> str:
    """
    Làm sạch văn bản tiếng Việt:
    - Chuẩn hóa Unicode (quan trọng với tiếng Việt)
    - Xóa URL, HTML, ký tự đặc biệt
    - Giữ lại chữ cái, số, dấu câu cơ bản
    - Lowercase
    """
    if not isinstance(text, str):
        text = str(text)

    # Chuẩn hóa Unicode NFC — bắt buộc với tiếng Việt
    text = unicodedata.normalize("NFC", text)

    # Xóa URL
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Xóa HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Xóa email
    text = re.sub(r"\S+@\S+", " ", text)

    # Giữ lại chữ tiếng Việt, số, dấu câu cơ bản
    text = re.sub(
        r"[^a-zA-Z0-9\s\.,!?;:\-"
        r"áàảãạăắằẳẵặâấầẩẫậ"
        r"éèẻẽẹêếềểễệ"
        r"íìỉĩị"
        r"óòỏõọôốồổỗộơớờởỡợ"
        r"úùủũụưứừửữự"
        r"ýỳỷỹỵđ"
        r"ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ"
        r"ÉÈẺẼẸÊẾỀỂỄỆ"
        r"ÍÌỈĨỊ"
        r"ÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ"
        r"ÚÙỦŨỤƯỨỪỬỮỰ"
        r"ÝỲỶỸỴĐ]",
        " ", text
    )

    # Xóa khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()

    # Lowercase
    text = text.lower()

    return text

def word_segment(text: str) -> str:
    segmented = ViTokenizer.tokenize(text)
    return segmented



def preprocess_text(text: str) -> str:
    """Pipeline đầy đủ: clean → tách từ"""
    text = clean_text(text)
    text = word_segment(text)
    return text


# --- Test thử hàm ---
sample_texts = [
    "Thầy dạy RẤT hay!! 😊  http://fb.com  môn học tốt lắm",
    "Môn học khó, tài liệu không đủ, thi rớt nhiều quá.",
    "Bình thường, không có gì đặc biệt để khen hay chê.",
]

print("\n  Kết quả test hàm tiền xử lý:")
print(f"  {'Input':<50} {'Output'}")
print("  " + "-"*80)
for t in sample_texts:
    result = preprocess_text(t)
    print(f"  {t[:48]:<50} → {result[:50]}")

# ------------------------------------------------------------
# 4. LOAD TOKENIZER PHOBERT
# ------------------------------------------------------------
print(f"\n[3/5] Tải PhoBERT tokenizer ({MODEL_NAME})...")
print("  (Lần đầu sẽ tải ~500MB, chờ chút...)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"  Vocab size : {tokenizer.vocab_size:,} từ")
print(f"  Max length : {MAX_LENGTH} token")
print(f"  Token đặc biệt: CLS={tokenizer.cls_token}, "
      f"SEP={tokenizer.sep_token}, PAD={tokenizer.pad_token}")

# --- Test tokenizer ---
sample_seg = preprocess_text("thầy dạy rất hay môn học tốt lắm")
enc = tokenizer(
    sample_seg,
    max_length=MAX_LENGTH,
    truncation=True,
    padding="max_length",
    return_tensors="pt"
)
tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
real_tokens = [t for t in tokens if t != tokenizer.pad_token]

print(f"\n  Ví dụ tokenize:")
print(f"  Input     : '{sample_seg}'")
print(f"  Tokens    : {real_tokens}")
print(f"  input_ids shape : {enc['input_ids'].shape}")

# ------------------------------------------------------------
# 5. TIỀN XỬ LÝ VÀ LƯU CACHE
# ------------------------------------------------------------
print(f"\n[4/5] Tiền xử lý toàn bộ dataset và lưu cache...")
print("  (Bước này mất 5–15 phút tùy máy)")


def process_and_cache(df: pd.DataFrame, split_name: str) -> list:
    """
    Xử lý toàn bộ một tập dữ liệu:
    1. Clean text
    2. Tách từ
    3. Tokenize
    4. Lưu cache dạng pickle
    """
    cache_path = f"{CACHE_DIR}/{split_name}.pkl"

    # Nếu đã có cache thì load lại, không cần xử lý lại
    if os.path.exists(cache_path):
        print(f"  [{split_name}] Đã có cache, load lại...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    cache = []
    errors = 0
    start  = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  [{split_name}]", ncols=70):
        try:
            # Tiền xử lý văn bản
            text = preprocess_text(str(row["sentence"]))

            # Tokenize
            enc = tokenizer(
                text,
                max_length=MAX_LENGTH,
                truncation=True,
                padding="max_length",
            )

            cache.append({
                "input_ids":      enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "label":          int(row["sentiment"]),
                "text_original":  str(row["sentence"]),   # giữ lại để debug
                "text_processed": text,
            })

        except Exception as e:
            errors += 1
            print(f"\n  Lỗi dòng {_}: {e}")

    elapsed = time.time() - start
    print(f"  [{split_name}] Xong: {len(cache):,} mẫu | "
          f"Lỗi: {errors} | Thời gian: {elapsed:.0f}s")

    # Lưu cache
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    size_mb = os.path.getsize(cache_path) / (1024*1024)
    print(f"  [{split_name}] Đã lưu: {cache_path} ({size_mb:.1f} MB)")

    return cache


# Chạy lần lượt 3 tập
cache_train = process_and_cache(df_train, "train")
cache_val   = process_and_cache(df_val,   "val")
cache_test  = process_and_cache(df_test,  "test")

# ------------------------------------------------------------
# 6. KIỂM TRA KẾT QUẢ
# ------------------------------------------------------------
print(f"\n[5/5] Kiểm tra kết quả...")

print(f"\n  Số mẫu sau xử lý:")
print(f"  Train : {len(cache_train):,}")
print(f"  Val   : {len(cache_val):,}")
print(f"  Test  : {len(cache_test):,}")

# Xem 3 ví dụ cụ thể
print(f"\n  Ví dụ 3 mẫu đầu tiên trong train cache:")
print("  " + "-"*70)
for i, item in enumerate(cache_train[:3]):
    print(f"\n  [{i+1}] Gốc     : {item['text_original'][:60]}")
    print(f"       Xử lý  : {item['text_processed'][:60]}")
    print(f"       Nhãn   : {item['label']} "
          f"({'Tiêu cực' if item['label']==0 else 'Trung tính' if item['label']==1 else 'Tích cực'})")
    print(f"       input_ids  (10 đầu): {item['input_ids'][:10]}")
    print(f"       attn_mask  (10 đầu): {item['attention_mask'][:10]}")
    print(f"       Độ dài thực (không pad): "
          f"{sum(item['attention_mask'])} token")

# Thống kê độ dài token thực tế
real_lengths = [sum(item["attention_mask"]) for item in cache_train]
avg_len = sum(real_lengths) / len(real_lengths)
max_len = max(real_lengths)
over_256 = sum(1 for l in real_lengths if l >= MAX_LENGTH)

print(f"\n  Thống kê độ dài token (train):")
print(f"  Trung bình  : {avg_len:.1f} token")
print(f"  Dài nhất    : {max_len} token")
print(f"  Bị cắt (>={MAX_LENGTH}): {over_256} câu "
      f"({over_256/len(real_lengths)*100:.1f}%) — chấp nhận được")

# ------------------------------------------------------------
# TỔNG KẾT
# ------------------------------------------------------------
print("\n" + "="*55)
print("BƯỚC 2 HOÀN THÀNH!")
print("="*55)
print("Các file đã tạo:")
print(f"  cache/train.pkl  — {len(cache_train):,} mẫu")
print(f"  cache/val.pkl    — {len(cache_val):,} mẫu")
print(f"  cache/test.pkl   — {len(cache_test):,} mẫu")
print("\nBước tiếp theo: Bước 3 — Xây dựng và huấn luyện mô hình PhoBERT")