# ============================================================
#  BƯỚC 6 — Demo Web Phân tích Cảm xúc Phản hồi Sinh viên
#  Sử dụng Flask + Bootstrap
#  Chạy: python step6_demo_web.py
#  Mở trình duyệt: http://localhost:5000
# ============================================================

import os
import re
import json
import unicodedata

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from pyvi import ViTokenizer
from flask import Flask, request, jsonify, render_template_string

# ============================================================
#  CẤU HÌNH
# ============================================================
MODEL_DIR  = "models"
MODEL_NAME = "vinai/phobert-base"
DEVICE     = torch.device("cpu")
PORT       = 5000

LABEL_MAP = {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"}
LABEL_EN  = {0: "negative", 1: "neutral",   2: "positive"}
COLORS    = {0: "#E24B4A",  1: "#F59E0B",   2: "#1D9E75"}
ICONS     = {0: "😞",       1: "😐",         2: "😊"}

# ============================================================
#  ĐỊNH NGHĨA 2 MODEL
# ============================================================
class PhoBERTSentiment(nn.Module):
    """PhoBERT đơn thuần — dùng [CLS] token"""
    def __init__(self):
        super().__init__()
        self.phobert    = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout    = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 3)
    def forward(self, input_ids, attention_mask):
        out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))


class CNNPhoBERT(nn.Module):
    """CNN + PhoBERT — dùng toàn bộ sequence"""
    def __init__(self, num_filters=256, kernel_sizes=[2,3,4], dropout=0.3):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(MODEL_NAME)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(768, num_filters, kernel_size=k, padding=k//2),
                nn.ReLU(), nn.Dropout(dropout)
            ) for k in kernel_sizes
        ])
        self.dropout    = nn.Dropout(dropout)
        cnn_out_dim     = num_filters * len(kernel_sizes)
        self.classifier = nn.Sequential(
            nn.Linear(cnn_out_dim, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, 3)
        )
    def forward(self, input_ids, attention_mask):
        out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        x   = out.last_hidden_state.permute(0, 2, 1)
        pooled = [torch.max(conv(x), dim=2)[0] for conv in self.convs]
        x_cat  = torch.cat(pooled, dim=1)
        return self.classifier(self.dropout(x_cat))


# ============================================================
#  LOAD MODELS
# ============================================================
print("Đang khởi động hệ thống...")
print("Tải PhoBERT tokenizer...", end=" ", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("OK")

models_loaded = {}

# Load PhoBERT
phobert_path = f"{MODEL_DIR}/PhoBERT_best.pt"
if os.path.exists(phobert_path):
    print("Tải PhoBERT model...", end=" ", flush=True)
    m1 = PhoBERTSentiment().to(DEVICE)
    m1.load_state_dict(torch.load(phobert_path, map_location=DEVICE))
    m1.eval()
    models_loaded["PhoBERT"] = m1
    print("OK")
else:
    print(f"⚠️  Không tìm thấy {phobert_path}")

# Load CNN+PhoBERT
cnn_phobert_path = f"{MODEL_DIR}/CNN_PhoBERT_best.pt"
if os.path.exists(cnn_phobert_path):
    print("Tải CNN+PhoBERT model...", end=" ", flush=True)
    m2 = CNNPhoBERT().to(DEVICE)
    m2.load_state_dict(torch.load(cnn_phobert_path, map_location=DEVICE))
    m2.eval()
    models_loaded["CNN+PhoBERT"] = m2
    print("OK")
else:
    print(f"⚠️  Không tìm thấy {cnn_phobert_path}")

print(f"Đã tải {len(models_loaded)} model: {list(models_loaded.keys())}")

# ============================================================
#  HÀM TIỀN XỬ LÝ & DỰ ĐOÁN
# ============================================================
def clean_text(text):
    if not isinstance(text, str): text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(
        r"[^a-zA-Z0-9\s\.,!?;:\-"
        r"áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩị"
        r"óòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ"
        r"ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊ"
        r"ÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ]",
        " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()

def preprocess(text):
    text = clean_text(text)
    text = ViTokenizer.tokenize(text)
    return text

def predict_one(text, model):
    processed = preprocess(text)
    enc = tokenizer(
        processed, max_length=256,
        truncation=True, padding="max_length",
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(
            enc["input_ids"].to(DEVICE),
            enc["attention_mask"].to(DEVICE)
        )
    probs    = torch.softmax(logits, dim=1)[0].cpu().numpy()
    label_id = int(probs.argmax())
    return {
        "label_id":   label_id,
        "label":      LABEL_MAP[label_id],
        "icon":       ICONS[label_id],
        "color":      COLORS[label_id],
        "confidence": round(float(probs[label_id]) * 100, 1),
        "probs": {
            "Tiêu cực":  round(float(probs[0]) * 100, 1),
            "Trung tính": round(float(probs[1]) * 100, 1),
            "Tích cực":  round(float(probs[2]) * 100, 1),
        },
        "processed": processed
    }

# Load kết quả so sánh nếu có
results_path = "outputs/results_baseline.json"
model_results = {}
if os.path.exists(results_path):
    with open(results_path, "r", encoding="utf-8") as f:
        model_results = json.load(f)

# ============================================================
#  HTML TEMPLATE
# ============================================================
HTML = """
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Phân tích Cảm xúc Phản hồi Sinh viên</title>
<link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
<style>
  body { background:#F0F4F8; font-family:'Segoe UI',sans-serif; }
  .navbar { background:linear-gradient(135deg,#1F4E79,#2E75B6); }
  .card  { border:none; border-radius:16px; box-shadow:0 4px 20px rgba(0,0,0,.08); }
  .btn-primary { background:linear-gradient(135deg,#1F4E79,#2E75B6); border:none; border-radius:10px; padding:12px 32px; font-size:1.05rem; }
  .btn-primary:hover { opacity:.9; transform:translateY(-1px); transition:.2s; }
  .result-card { border-radius:16px; padding:1.5rem; color:#fff; transition:all .3s; }
  .result-positive { background:linear-gradient(135deg,#1D9E75,#16a085); }
  .result-neutral  { background:linear-gradient(135deg,#F59E0B,#d68910); }
  .result-negative { background:linear-gradient(135deg,#E24B4A,#c0392b); }
  .prob-bar { height:14px; border-radius:8px; transition:width .6s ease; }
  .model-badge { border-radius:20px; padding:6px 16px; font-size:.85rem; font-weight:600; cursor:pointer; transition:.2s; }
  .model-badge.active { box-shadow:0 0 0 3px rgba(30,78,121,.4); transform:scale(1.05); }
  .example-btn { border-radius:20px; font-size:.82rem; padding:5px 14px; margin:3px; }
  .history-item { border-left:4px solid; border-radius:0 8px 8px 0; padding:10px 14px; margin-bottom:8px; background:#fff; }
  .stat-card { border-radius:12px; padding:1rem; text-align:center; }
  .confidence-ring { font-size:2.2rem; font-weight:700; }
  textarea { border-radius:12px; resize:vertical; min-height:120px; font-size:1rem; }
  .spinner-border { width:1.2rem; height:1.2rem; }
  #compareSection table td, #compareSection table th { padding:.6rem 1rem; }
  .fade-in { animation:fadeIn .4s ease; }
  @keyframes fadeIn { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
</style>
</head>
<body>

<!-- NAVBAR -->
<nav class="navbar navbar-dark px-4 py-3 mb-4">
  <div class="container-fluid">
    <span class="navbar-brand fw-bold fs-5">🎓 Phân tích Cảm xúc Phản hồi Sinh viên</span>
    <span class="text-white-50 small">PhoBERT &amp; CNN+PhoBERT · UIT-VSFC</span>
  </div>
</nav>

<div class="container" style="max-width:960px">

  <!-- INPUT SECTION -->
  <div class="card mb-4">
    <div class="card-body p-4">
      <h5 class="fw-bold mb-3">📝 Nhập câu phản hồi</h5>

      <!-- Chọn model -->
      <div class="mb-3">
        <label class="form-label text-muted small fw-semibold">CHỌN MÔ HÌNH</label>
        <div id="modelSelector" class="d-flex flex-wrap gap-2">
          {% for name in models %}
          <span class="model-badge border border-primary text-primary {% if loop.first %}active{% endif %}"
                onclick="selectModel(this, '{{ name }}')" data-model="{{ name }}">
            {{ name }}
          </span>
          {% endfor %}
          {% if models|length > 1 %}
          <span class="model-badge border border-secondary text-secondary"
                onclick="selectModel(this, 'so_sanh')" data-model="so_sanh">
            ⚖️ So sánh 2 model
          </span>
          {% endif %}
        </div>
      </div>

      <!-- Textarea -->
      <textarea id="inputText" class="form-control mb-3"
        placeholder="Nhập câu phản hồi tiếng Việt về môn học, giảng viên, bài tập..."></textarea>

      <!-- Câu ví dụ -->
      <div class="mb-3">
        <span class="text-muted small">Ví dụ nhanh:</span><br>
        {% for ex in examples %}
        <button class="btn btn-outline-secondary example-btn mt-1"
                onclick="setExample('{{ ex[0] }}')">{{ ex[0][:45] }}...</button>
        {% endfor %}
      </div>

      <button class="btn btn-primary w-100" onclick="analyze()">
        <span id="btnText">🔍 Phân tích cảm xúc</span>
        <span id="btnSpinner" class="spinner-border ms-2 d-none" role="status"></span>
      </button>
    </div>
  </div>

  <!-- KẾT QUẢ -->
  <div id="resultSection" class="d-none fade-in mb-4">
    <div id="resultContainer"></div>
  </div>

  <!-- THỐNG KÊ PHIÊN -->
  <div class="row mb-4 g-3" id="statsRow" style="display:none!important">
    <div class="col-4">
      <div class="stat-card" style="background:#E1F5EE">
        <div class="fs-3 fw-bold text-success" id="statPos">0</div>
        <div class="small text-muted">😊 Tích cực</div>
      </div>
    </div>
    <div class="col-4">
      <div class="stat-card" style="background:#FFF3CD">
        <div class="fs-3 fw-bold text-warning" id="statNeu">0</div>
        <div class="small text-muted">😐 Trung tính</div>
      </div>
    </div>
    <div class="col-4">
      <div class="stat-card" style="background:#FAECE7">
        <div class="fs-3 fw-bold text-danger" id="statNeg">0</div>
        <div class="small text-muted">😞 Tiêu cực</div>
      </div>
    </div>
  </div>

  <!-- SO SÁNH MODEL -->
  {% if model_results %}
  <div class="card mb-4" id="compareSection">
    <div class="card-body p-4">
      <h5 class="fw-bold mb-3">📊 Kết quả thực nghiệm trên Test Set</h5>
      <div class="table-responsive">
        <table class="table table-hover align-middle">
          <thead class="table-dark">
            <tr>
              <th>Mô hình</th>
              <th class="text-center">Accuracy</th>
              <th class="text-center">F1-macro</th>
              <th>F1-macro (bar)</th>
            </tr>
          </thead>
          <tbody>
            {% for name, r in model_results.items() %}
            <tr {% if r.f1_macro == best_f1 %}class="table-success fw-bold"{% endif %}>
              <td>{{ name }} {% if r.f1_macro == best_f1 %}<span class="badge bg-success">BEST</span>{% endif %}</td>
              <td class="text-center">{{ "%.2f"|format(r.accuracy*100) }}%</td>
              <td class="text-center">{{ "%.4f"|format(r.f1_macro) }}</td>
              <td>
                <div class="progress" style="height:14px;border-radius:8px">
                  <div class="progress-bar {% if r.f1_macro == best_f1 %}bg-success{% else %}bg-primary{% endif %}"
                       style="width:{{ (r.f1_macro*100)|round }}%">
                  </div>
                </div>
              </td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
  </div>
  {% endif %}

  <!-- LỊCH SỬ -->
  <div class="card mb-4" id="historyCard" style="display:none">
    <div class="card-body p-4">
      <div class="d-flex justify-content-between align-items-center mb-3">
        <h5 class="fw-bold mb-0">🕐 Lịch sử phân tích</h5>
        <button class="btn btn-sm btn-outline-danger" onclick="clearHistory()">Xóa</button>
      </div>
      <div id="historyList"></div>
    </div>
  </div>

</div><!-- /container -->

<footer class="text-center py-4 text-muted small mt-4">
  Đồ án tốt nghiệp · Phân tích cảm xúc phản hồi sinh viên · PhoBERT &amp; Học sâu · 2025
</footer>

<script>
let selectedModel = '{{ models[0] if models else "PhoBERT" }}';
let history = [];
let stats = {pos:0, neu:0, neg:0};

function selectModel(el, name) {
  document.querySelectorAll('.model-badge').forEach(b => b.classList.remove('active'));
  el.classList.add('active');
  selectedModel = name;
}

function setExample(text) {
  document.getElementById('inputText').value = text;
}

function analyze() {
  const text = document.getElementById('inputText').value.trim();
  if (!text) { alert('Vui lòng nhập câu phản hồi!'); return; }
  if (text.split(' ').length < 2) { alert('Câu quá ngắn, vui lòng nhập ít nhất 3 từ!'); return; }

  document.getElementById('btnText').textContent = 'Đang phân tích...';
  document.getElementById('btnSpinner').classList.remove('d-none');

  fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ text: text, model: selectedModel })
  })
  .then(r => r.json())
  .then(data => {
    document.getElementById('btnText').textContent = '🔍 Phân tích cảm xúc';
    document.getElementById('btnSpinner').classList.add('d-none');

    if (data.error) { alert('Lỗi: ' + data.error); return; }

    showResult(data, text);
    addHistory(text, data);
    updateStats(data);
  })
  .catch(e => {
    document.getElementById('btnText').textContent = '🔍 Phân tích cảm xúc';
    document.getElementById('btnSpinner').classList.add('d-none');
    alert('Lỗi kết nối server: ' + e.message);
  });
}

function showResult(data, text) {
  const section = document.getElementById('resultSection');
  const container = document.getElementById('resultContainer');
  section.classList.remove('d-none');

  // So sánh 2 model
  if (data.compare) {
    let html = '<div class="row g-3">';
    for (const [modelName, res] of Object.entries(data.compare)) {
      const cls = res.label_id===2 ? 'positive' : res.label_id===0 ? 'negative' : 'neutral';
      html += `
        <div class="col-md-6">
          <div class="result-card result-${cls}">
            <div class="d-flex justify-content-between align-items-center mb-2">
              <span class="fw-bold">${modelName}</span>
              <span class="badge bg-white text-dark">${res.confidence}% confidence</span>
            </div>
            <div class="text-center my-2">
              <div style="font-size:2.5rem">${res.icon}</div>
              <div class="fw-bold fs-5">${res.label}</div>
            </div>
            ${probBars(res.probs)}
          </div>
        </div>`;
    }
    html += '</div>';
    container.innerHTML = html;
  } else {
    const res = data.result;
    const cls = res.label_id===2 ? 'positive' : res.label_id===0 ? 'negative' : 'neutral';
    container.innerHTML = `
      <div class="result-card result-${cls}">
        <div class="d-flex justify-content-between align-items-start mb-3">
          <div>
            <div class="text-white-50 small mb-1">Mô hình: ${data.model}</div>
            <div class="fw-bold fs-6">"${text.length>80 ? text.substring(0,80)+'...' : text}"</div>
          </div>
          <div class="text-end">
            <div style="font-size:2.8rem;line-height:1">${res.icon}</div>
          </div>
        </div>
        <div class="row align-items-center">
          <div class="col-auto">
            <div class="confidence-ring">${res.label}</div>
            <div class="text-white-50 small">${res.confidence}% confidence</div>
          </div>
          <div class="col">
            ${probBars(res.probs)}
          </div>
        </div>
        <div class="mt-3 pt-3 border-top border-white border-opacity-25">
          <span class="text-white-50 small">Văn bản đã xử lý: </span>
          <span class="small">${res.processed.length>80 ? res.processed.substring(0,80)+'...' : res.processed}</span>
        </div>
      </div>`;
  }

  section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function probBars(probs) {
  const colors = {'Tích cực':'#1D9E75','Trung tính':'#F59E0B','Tiêu cực':'#E24B4A'};
  let html = '';
  for (const [name, val] of Object.entries(probs)) {
    html += `
      <div class="d-flex align-items-center gap-2 mb-1">
        <span class="small text-white-50" style="width:82px">${name}</span>
        <div class="flex-grow-1 bg-white bg-opacity-25 rounded" style="height:10px">
          <div class="prob-bar" style="width:${val}%;background:rgba(255,255,255,0.8)"></div>
        </div>
        <span class="small fw-bold" style="width:46px;text-align:right">${val}%</span>
      </div>`;
  }
  return html;
}

function addHistory(text, data) {
  const res = data.compare ? Object.values(data.compare)[0] : data.result;
  const color = res.label_id===2 ? '#1D9E75' : res.label_id===0 ? '#E24B4A' : '#F59E0B';
  const label = LABEL_MAP_JS[res.label_id];
  history.unshift({ text, label, icon: res.icon, color, model: data.model || selectedModel });
  if (history.length > 10) history.pop();

  const card = document.getElementById('historyCard');
  const list = document.getElementById('historyList');
  card.style.display = 'block';

  list.innerHTML = history.map((h,i) => `
    <div class="history-item" style="border-left-color:${h.color}">
      <div class="d-flex justify-content-between">
        <span class="small fw-semibold">${h.icon} ${h.label}</span>
        <span class="badge" style="background:${h.color}">${h.model}</span>
      </div>
      <div class="text-muted small mt-1">${h.text.length>90 ? h.text.substring(0,90)+'...' : h.text}</div>
    </div>`).join('');
}

function updateStats(data) {
  const res = data.compare ? Object.values(data.compare)[0] : data.result;
  if (res.label_id === 2) stats.pos++;
  else if (res.label_id === 1) stats.neu++;
  else stats.neg++;
  document.getElementById('statPos').textContent = stats.pos;
  document.getElementById('statNeu').textContent = stats.neu;
  document.getElementById('statNeg').textContent = stats.neg;
  document.getElementById('statsRow').style.display = '';
  document.getElementById('statsRow').style.removeProperty('display');
  document.getElementById('statsRow').classList.remove('d-none');
  document.getElementById('statsRow').style.display = 'flex';
}

function clearHistory() {
  history = [];
  document.getElementById('historyList').innerHTML = '';
  document.getElementById('historyCard').style.display = 'none';
}

// Keyboard shortcut: Ctrl+Enter
document.addEventListener('keydown', e => {
  if (e.ctrlKey && e.key === 'Enter') analyze();
});

const LABEL_MAP_JS = {0:'Tiêu cực', 1:'Trung tính', 2:'Tích cực'};
</script>
</body>
</html>
"""

# ============================================================
#  FLASK APP
# ============================================================
app = Flask(__name__)

EXAMPLES = [
    ("Thầy dạy rất nhiệt tình, giải thích rõ ràng và dễ hiểu.", 2),
    ("Môn học quá khó, tài liệu không đủ, thi cử nặng quá.", 0),
    ("Môn học bình thường, không có gì đặc biệt để khen hay chê.", 1),
    ("Cô dạy hay, kiến thức thực tế, em học được nhiều thứ bổ ích.", 2),
    ("Giáo viên giảng nhanh quá, sinh viên không theo kịp bài.", 0),
    ("Bài tập vừa đủ, nội dung ổn theo chương trình.", 1),
]

@app.route("/")
def index():
    best_f1 = max((v["f1_macro"] for v in model_results.values()), default=0) if model_results else 0
    return render_template_string(
        HTML,
        models=list(models_loaded.keys()),
        examples=EXAMPLES,
        model_results=model_results,
        best_f1=best_f1
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data       = request.get_json()
        text       = data.get("text", "").strip()
        model_name = data.get("model", "PhoBERT")

        if not text:
            return jsonify({"error": "Vui lòng nhập câu phản hồi!"})
        if len(text.split()) < 2:
            return jsonify({"error": "Câu quá ngắn!"})

        # So sánh 2 model
        if model_name == "so_sanh":
            compare = {}
            for name, model in models_loaded.items():
                compare[name] = predict_one(text, model)
            return jsonify({"compare": compare, "model": "So sánh"})

        # Dự đoán 1 model
        if model_name not in models_loaded:
            model_name = list(models_loaded.keys())[0]
        result = predict_one(text, models_loaded[model_name])
        return jsonify({"result": result, "model": model_name})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/models")
def get_models():
    return jsonify({"models": list(models_loaded.keys())})

# ============================================================
#  CHẠY APP
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("DEMO WEB SẴN SÀNG!")
    print("=" * 55)
    print(f"  Models loaded: {list(models_loaded.keys())}")
    print(f"\n  Mở trình duyệt và vào địa chỉ:")
    print(f"  ➜  http://localhost:{PORT}")
    print(f"\n  Nhấn Ctrl+C để dừng server")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)