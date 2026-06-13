# ============================================================
#  BƯỚC 7 — Demo Phân tích Cảm xúc Phản hồi Sinh viên
#  Đề tài: Phân tích cảm xúc phản hồi sinh viên với PhoBERT
#  Chạy: python step7_demo.py
# ============================================================

import torch
import torch.nn as nn
import unicodedata
import re
from transformers import AutoTokenizer, AutoModel


# ------------------------------------------------------------
# 1. CẤU HÌNH
# ------------------------------------------------------------
MODEL_PATH = "models/PhoBERT_best.pt"
MODEL_NAME = "vinai/phobert-base"
MAX_LENGTH = 256
DEVICE     = torch.device("cpu")

LABEL_MAP  = {0: "Tiêu cực 😞", 1: "Trung tính 😐", 2: "Tích cực 😊"}
LABEL_VI   = {0: "TIÊU CỰC", 1: "TRUNG TÍNH", 2: "TÍCH CỰC"}
COLORS     = {0: "\033[91m", 1: "\033[93m", 2: "\033[92m"}  # đỏ / vàng / xanh
RESET      = "\033[0m"
BOLD       = "\033[1m"

# ------------------------------------------------------------
# 2. LOAD MODEL
# ------------------------------------------------------------
class PhoBERTSentiment(nn.Module):
    def __init__(self):
        super().__init__()
        self.phobert    = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout    = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask):
        out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))


def load_model():
    print("  Đang tải mô hình PhoBERT...", end=" ", flush=True)
    model = PhoBERTSentiment().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("OK!")
    return model


tokenizer = None
def load_tokenizer():
    global tokenizer
    if tokenizer is None:
        print("  Đang tải tokenizer...", end=" ", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print("OK!")
    return tokenizer

# ------------------------------------------------------------
# 3. HÀM TIỀN XỬ LÝ
# ------------------------------------------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(
        r"[^a-zA-Z0-9\s\.,!?;:\-"
        r"áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệ"
        r"íìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ"
        r"ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆ"
        r"ÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ]",
        " ", text
    )
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def preprocess(text: str) -> str:
    text = clean_text(text)
    return text   # tokenizer PhoBERT tự xử lý, không cần tách từ thêm

# ------------------------------------------------------------
# 4. HÀM DỰ ĐOÁN
# ------------------------------------------------------------
def predict(text: str, model, tok) -> dict:
    """Dự đoán cảm xúc và trả về kết quả chi tiết"""
    processed = preprocess(text)
    enc = tok(
        processed,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(
            enc["input_ids"].to(DEVICE),
            enc["attention_mask"].to(DEVICE)
        )

    probs     = torch.softmax(logits, dim=1)[0].cpu().numpy()
    label_id  = int(probs.argmax())

    return {
        "label_id":    label_id,
        "label":       LABEL_MAP[label_id],
        "label_vi":    LABEL_VI[label_id],
        "confidence":  float(probs[label_id]) * 100,
        "probs": {
            "Tiêu cực":  float(probs[0]) * 100,
            "Trung tính": float(probs[1]) * 100,
            "Tích cực":  float(probs[2]) * 100,
        },
        "processed_text": processed,
    }


def print_result(text: str, result: dict):
    """In kết quả đẹp ra terminal"""
    label_id = result["label_id"]
    color    = COLORS[label_id]

    print(f"\n  {'─'*50}")
    print(f"  Input    : {text}")
    print(f"  Xử lý   : {result['processed_text'][:70]}...")
    print(f"  {'─'*50}")
    print(f"  {BOLD}Kết quả  : {color}{result['label_vi']}{RESET}"
          f"  (confidence: {result['confidence']:.1f}%)")
    print(f"  {'─'*50}")

    # Thanh xác suất
    for name, prob in result["probs"].items():
        bar_len = int(prob / 2.5)
        bar     = "█" * bar_len + "░" * (40 - bar_len)
        if name == "Tiêu cực":
            c = COLORS[0]
        elif name == "Trung tính":
            c = COLORS[1]
        else:
            c = COLORS[2]
        print(f"  {name:10s} {c}{bar}{RESET} {prob:5.1f}%")


def batch_predict(sentences: list, model, tok):
    """Dự đoán hàng loạt và in bảng tổng hợp"""
    print(f"\n  {'─'*65}")
    print(f"  {'#':>3}  {'Câu':<40} {'Kết quả':<12} {'Conf':>6}")
    print(f"  {'─'*65}")
    for i, sent in enumerate(sentences, 1):
        r = predict(sent, model, tok)
        c = COLORS[r["label_id"]]
        short = sent[:38] + ".." if len(sent) > 40 else sent
        print(f"  {i:>3}  {short:<40} "
              f"{c}{r['label_vi']:<12}{RESET} {r['confidence']:>5.1f}%")
    print(f"  {'─'*65}")

# ------------------------------------------------------------
# 5. CHẠY DEMO
# ------------------------------------------------------------
def main():
    print("\n" + "=" * 55)
    print("  DEMO — PHÂN TÍCH CẢM XÚC PHẢN HỒI SINH VIÊN")
    print("  Mô hình: PhoBERT fine-tuned trên UIT-VSFC")
    print("=" * 55)

    # Load model
    print("\nĐang khởi tạo hệ thống...")
    model = load_model()
    tok   = load_tokenizer()
    print("  Hệ thống sẵn sàng!\n")

    # --- Demo tự động với ví dụ mẫu ---
    print("=" * 55)
    print("  THỬ NGHIỆM VỚI CÂU MẪU")
    print("=" * 55)

    examples = [
        "Thầy dạy rất nhiệt tình, giải thích rõ ràng và dễ hiểu.",
        "Môn học quá khó, tài liệu không đủ, thi trượt nhiều.",
        "Môn học bình thường, không có gì đặc biệt để khen hay chê.",
        "Giảng viên tâm huyết, nội dung bài giảng hay và thực tế.",
        "Bài tập quá nhiều, thời gian học không đủ để hoàn thành.",
        "Phòng học nóng bức, ảnh hưởng nhiều đến việc tiếp thu bài.",
        "Môn học thú vị, em học được nhiều kiến thức bổ ích.",
        "Giáo viên giảng nhanh quá, sinh viên khó theo kịp bài.",
    ]

    for text in examples:
        result = predict(text, model, tok)
        print_result(text, result)

    # --- Bảng tổng hợp ---
    print(f"\n{'='*55}")
    print(f"  BẢNG TỔNG HỢP DỰ ĐOÁN")
    print(f"{'='*55}")
    batch_predict(examples, model, tok)

    # --- Chế độ nhập tay ---
    print(f"\n{'='*55}")
    print(f"  CHẾ ĐỘ NHẬP TAY (gõ 'q' để thoát)")
    print(f"{'='*55}")
    print("  Nhập câu phản hồi tiếng Việt để phân tích cảm xúc:")

    while True:
        print()
        user_input = input("  >>> Nhập câu: ").strip()

        if user_input.lower() in ["q", "quit", "exit", "thoát"]:
            print("\n  Cảm ơn! Hẹn gặp lại.\n")
            break

        if not user_input:
            print("  ⚠️  Vui lòng nhập câu không rỗng!")
            continue

        if len(user_input.split()) < 2:
            print("  ⚠️  Câu quá ngắn, vui lòng nhập ít nhất 2 từ!")
            continue

        result = predict(user_input, model, tok)
        print_result(user_input, result)


if __name__ == "__main__":
    main()