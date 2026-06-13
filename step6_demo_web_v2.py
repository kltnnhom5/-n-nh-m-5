# ============================================================
#  BƯỚC 6 V2 — Demo Web Nâng cao
#  Tính năng:
#    1. Nhập câu đơn → phân tích cảm xúc
#    2. Upload file Excel/CSV → phân tích hàng loạt
#    3. Vẽ biểu đồ thống kê % cảm xúc
#    4. Xuất kết quả ra file Excel
#  Chạy: python step6_demo_web_v2.py
#  Mở: http://localhost:5000
# ============================================================

import os, re, io, json, unicodedata, base64
import torch, torch.nn as nn
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from transformers import AutoModel, AutoTokenizer
from pyvi import ViTokenizer
from flask import Flask, request, jsonify, render_template_string, send_file

# ── Cấu hình ─────────────────────────────────────────────────
MODEL_DIR  = "models"
MODEL_NAME = "vinai/phobert-base"
DEVICE     = torch.device("cpu")
PORT       = 5000
COLORS     = {0:"#E24B4A", 1:"#F59E0B", 2:"#1D9E75"}
LABEL_MAP  = {0:"Tiêu cực", 1:"Trung tính", 2:"Tích cực"}
ICONS      = {0:"😞", 1:"😐", 2:"😊"}

# ── Model definitions ─────────────────────────────────────────
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

class CNNPhoBERT(nn.Module):
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
        self.classifier = nn.Sequential(
            nn.Linear(num_filters*len(kernel_sizes), 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, 3)
        )
    def forward(self, input_ids, attention_mask):
        out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        x   = out.last_hidden_state.permute(0, 2, 1)
        pooled = [torch.max(conv(x), dim=2)[0] for conv in self.convs]
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))

# ── Load models ───────────────────────────────────────────────
print("Đang khởi động hệ thống...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Tokenizer OK")

models_loaded = {}
for name, cls, path in [
    ("PhoBERT",     PhoBERTSentiment, f"{MODEL_DIR}/PhoBERT_best.pt"),
    ("CNN+PhoBERT", CNNPhoBERT,       f"{MODEL_DIR}/CNN_PhoBERT_best.pt"),
]:
    if os.path.exists(path):
        print(f"Tải {name}...", end=" ", flush=True)
        m = cls().to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        models_loaded[name] = m
        print("OK")
    else:
        print(f"⚠️  Không tìm thấy {path}")

print(f"Đã tải {len(models_loaded)} model: {list(models_loaded.keys())}")

# ── Tiền xử lý ───────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str): text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(
        r"[^a-zA-Z0-9\s\.,!?;:\-áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệ"
        r"íìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ"
        r"ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊ"
        r"ÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ]",
        " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()

def preprocess(text):
    return ViTokenizer.tokenize(clean_text(text))

def predict_one(text, model):
    processed = preprocess(text)
    enc = tokenizer(processed, max_length=256, truncation=True,
                    padding="max_length", return_tensors="pt")
    with torch.no_grad():
        logits = model(enc["input_ids"].to(DEVICE),
                       enc["attention_mask"].to(DEVICE))
    probs    = torch.softmax(logits, dim=1)[0].cpu().numpy()
    label_id = int(probs.argmax())
    return {
        "label_id":   label_id,
        "label":      LABEL_MAP[label_id],
        "icon":       ICONS[label_id],
        "color":      COLORS[label_id],
        "confidence": round(float(probs[label_id])*100, 1),
        "probs": {
            "Tiêu cực":  round(float(probs[0])*100, 1),
            "Trung tính": round(float(probs[1])*100, 1),
            "Tích cực":  round(float(probs[2])*100, 1),
        },
        "processed": processed
    }

def predict_batch(texts, model):
    """Dự đoán nhanh theo batch"""
    results = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        processed   = [preprocess(t) for t in batch_texts]
        encodings   = tokenizer(
            processed, max_length=256, truncation=True,
            padding=True, return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(
                encodings["input_ids"].to(DEVICE),
                encodings["attention_mask"].to(DEVICE)
            )
        probs_batch = torch.softmax(logits, dim=1).cpu().numpy()
        for j, probs in enumerate(probs_batch):
            label_id = int(probs.argmax())
            results.append({
                "label_id":   label_id,
                "label":      LABEL_MAP[label_id],
                "confidence": round(float(probs[label_id])*100, 1),
                "prob_neg":   round(float(probs[0])*100, 1),
                "prob_neu":   round(float(probs[1])*100, 1),
                "prob_pos":   round(float(probs[2])*100, 1),
            })
    return results

# ── Vẽ biểu đồ ───────────────────────────────────────────────
def make_charts(results_df):
    """Vẽ 3 biểu đồ: pie, bar, histogram confidence — trả về base64"""
    counts  = results_df["label"].value_counts()
    labels  = [LABEL_MAP[i] for i in [2,1,0] if LABEL_MAP[i] in counts.index]
    sizes   = [counts.get(LABEL_MAP[i], 0) for i in [2,1,0]]
    colors  = [COLORS[2], COLORS[1], COLORS[0]]
    total   = len(results_df)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor("#F8FAFC")
    fig.suptitle(f"Phân tích cảm xúc — {total:,} câu phản hồi sinh viên",
                 fontsize=13, fontweight="bold", color="#1F4E79", y=1.02)

    # ── Biểu đồ 1: Pie chart ─────────────────────────────────
    ax1 = axes[0]
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors,
        autopct=lambda p: f"{p:.1f}%\n({int(p*total/100):,})",
        startangle=90, wedgeprops={"edgecolor":"white","linewidth":2},
        textprops={"fontsize":10}, pctdistance=0.75
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
        at.set_color("white")
    ax1.set_title("Phân phối cảm xúc", fontsize=11, fontweight="bold", color="#1F4E79")

    # ── Biểu đồ 2: Bar chart ngang ────────────────────────────
    ax2 = axes[1]
    bar_labels = [LABEL_MAP[2], LABEL_MAP[1], LABEL_MAP[0]]
    bar_values = [counts.get(LABEL_MAP[2],0), counts.get(LABEL_MAP[1],0), counts.get(LABEL_MAP[0],0)]
    bar_colors = [COLORS[2], COLORS[1], COLORS[0]]
    bars = ax2.barh(bar_labels, bar_values, color=bar_colors, edgecolor="white",
                    linewidth=1.5, height=0.55)
    for bar, val in zip(bars, bar_values):
        pct = val/total*100
        ax2.text(bar.get_width()+max(bar_values)*0.01, bar.get_y()+bar.get_height()/2,
                 f"{val:,}  ({pct:.1f}%)", va="center", fontsize=10, fontweight="bold")
    ax2.set_xlim(0, max(bar_values)*1.25)
    ax2.set_title("Số lượng theo nhãn", fontsize=11, fontweight="bold", color="#1F4E79")
    ax2.set_xlabel("Số câu", fontsize=10)
    ax2.spines[["top","right"]].set_visible(False)
    ax2.grid(axis="x", alpha=0.3)

    # ── Biểu đồ 3: Histogram confidence ──────────────────────
    ax3 = axes[2]
    for label_id, color in [(2,COLORS[2]),(1,COLORS[1]),(0,COLORS[0])]:
        subset = results_df[results_df["label_id"]==label_id]["confidence"]
        if len(subset) > 0:
            ax3.hist(subset, bins=20, alpha=0.65, color=color,
                     label=LABEL_MAP[label_id], edgecolor="white")
    ax3.set_title("Phân phối Confidence (%)", fontsize=11, fontweight="bold", color="#1F4E79")
    ax3.set_xlabel("Confidence (%)", fontsize=10)
    ax3.set_ylabel("Số câu", fontsize=10)
    ax3.legend(fontsize=9)
    ax3.spines[["top","right"]].set_visible(False)
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="#F8FAFC")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ── Load kết quả nếu có ──────────────────────────────────────
model_results = {}
rp = "outputs/results_baseline.json"
if os.path.exists(rp):
    with open(rp,"r",encoding="utf-8") as f:
        model_results = json.load(f)

# ══════════════════════════════════════════════════════════════
#  HTML TEMPLATE
# ══════════════════════════════════════════════════════════════
HTML = """
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phân tích Cảm xúc Phản hồi Sinh viên</title>
<link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
<style>
  body{background:#F0F4F8;font-family:'Segoe UI',sans-serif}
  .navbar{background:linear-gradient(135deg,#1F4E79,#2E75B6)}
  .card{border:none;border-radius:16px;box-shadow:0 4px 20px rgba(0,0,0,.08)}
  .btn-primary{background:linear-gradient(135deg,#1F4E79,#2E75B6);border:none;border-radius:10px;padding:10px 28px}
  .btn-primary:hover{opacity:.9;transform:translateY(-1px);transition:.2s}
  .result-card{border-radius:14px;padding:1.4rem;color:#fff}
  .result-positive{background:linear-gradient(135deg,#1D9E75,#16a085)}
  .result-neutral{background:linear-gradient(135deg,#F59E0B,#d68910)}
  .result-negative{background:linear-gradient(135deg,#E24B4A,#c0392b)}
  .prob-bar{height:12px;border-radius:8px;transition:width .6s}
  .tab-btn{border-radius:10px 10px 0 0;border:none;padding:10px 24px;background:#e2e8f0;color:#64748b;font-weight:500;cursor:pointer}
  .tab-btn.active{background:#fff;color:#1F4E79;box-shadow:0 -2px 8px rgba(0,0,0,.06)}
  .tab-pane{display:none}.tab-pane.active{display:block}
  .drop-zone{border:2.5px dashed #2E75B6;border-radius:14px;padding:2.5rem;text-align:center;cursor:pointer;transition:.2s;background:#f8fbff}
  .drop-zone:hover,.drop-zone.drag-over{background:#EBF5FB;border-color:#1F4E79}
  .drop-zone .icon{font-size:2.5rem}
  .badge-pos{background:#E1F5EE;color:#085041}
  .badge-neg{background:#FAECE7;color:#712B13}
  .badge-neu{background:#FFF3CD;color:#7B5E00}
  .progress-wrap{height:10px;border-radius:6px;background:#e2e8f0;overflow:hidden}
  .progress-fill{height:100%;border-radius:6px;transition:width .8s ease}
  .result-table th{background:#1F4E79;color:#fff;font-size:.85rem;white-space:nowrap}
  .result-table td{font-size:.85rem;vertical-align:middle}
  .spinner-border{width:1.2rem;height:1.2rem}
  .chart-img{border-radius:14px;box-shadow:0 4px 20px rgba(0,0,0,.1);max-width:100%}
  .stat-box{border-radius:12px;padding:1rem;text-align:center}
  .model-chip{border-radius:20px;padding:5px 14px;font-size:.82rem;font-weight:600;cursor:pointer;transition:.2s;display:inline-block;border:1.5px solid}
  .model-chip.active{box-shadow:0 0 0 3px rgba(30,78,121,.3)}
  .example-btn{border-radius:20px;font-size:.78rem;padding:3px 12px;margin:2px}
  @keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
  .fade-in{animation:fadeIn .35s ease}
  #fileProgress{display:none}
</style>
</head>
<body>
<nav class="navbar navbar-dark px-4 py-3 mb-4">
  <div class="container-fluid justify-content-between">
    <span class="navbar-brand fw-bold">🎓 Phân tích Cảm xúc Phản hồi Sinh viên</span>
    <span class="text-white-50 small d-none d-md-block">PhoBERT · CNN+PhoBERT · UIT-VSFC 2025</span>
  </div>
</nav>

<div class="container" style="max-width:980px">

  <!-- TABS -->
  <div class="mb-0 d-flex gap-1">
    <button class="tab-btn active" onclick="switchTab('single')">✏️ Nhập câu đơn</button>
    <button class="tab-btn" onclick="switchTab('file')">📂 Upload file hàng loạt</button>
    <button class="tab-btn" onclick="switchTab('compare')">📊 So sánh kết quả</button>
  </div>

  <!-- ═══ TAB 1: SINGLE ═══════════════════════════════════════ -->
  <div id="tab-single" class="tab-pane active">
    <div class="card mb-4" style="border-radius:0 16px 16px 16px">
      <div class="card-body p-4">
        <!-- Chọn model -->
        <div class="mb-3">
          <label class="text-muted small fw-semibold">CHỌN MÔ HÌNH</label><br>
          {% for name in models %}
          <span class="model-chip border-primary text-primary {% if loop.first %}active{% endif %} me-2 mt-1"
                onclick="selectModel(this,'{{name}}')" data-model="{{name}}">{{name}}</span>
          {% endfor %}
          {% if models|length>1 %}
          <span class="model-chip border-secondary text-secondary me-2 mt-1"
                onclick="selectModel(this,'so_sanh')" data-model="so_sanh">⚖️ So sánh 2 model</span>
          {% endif %}
        </div>
        <!-- Textarea -->
        <textarea id="inputText" class="form-control mb-3" rows="3"
          style="border-radius:12px;font-size:1rem"
          placeholder="Nhập câu phản hồi tiếng Việt về môn học, giảng viên..."></textarea>
        <!-- Ví dụ nhanh -->
        <div class="mb-3">
          <span class="text-muted small">Thử ngay:</span><br>
          {% for ex,_ in examples %}
          <button class="btn btn-outline-secondary example-btn mt-1"
                  onclick="setEx('{{ex}}')">{{ex[:42]}}...</button>
          {% endfor %}
        </div>
        <button class="btn btn-primary w-100 py-2" onclick="analyze()">
          <span id="btnTxt">🔍 Phân tích cảm xúc</span>
          <span id="btnSpin" class="spinner-border ms-2 d-none"></span>
        </button>
      </div>
    </div>
    <!-- Kết quả câu đơn -->
    <div id="singleResult" class="d-none fade-in mb-4"></div>
  </div>

  <!-- ═══ TAB 2: FILE UPLOAD ══════════════════════════════════ -->
  <div id="tab-file" class="tab-pane">
    <div class="card mb-4" style="border-radius:0 16px 16px 16px">
      <div class="card-body p-4">
        <h6 class="fw-bold mb-3">📂 Upload file để phân tích hàng loạt</h6>

        <!-- Chọn model -->
        <div class="mb-3">
          <label class="text-muted small fw-semibold">MÔ HÌNH</label><br>
          {% for name in models %}
          <span class="model-chip border-primary text-primary {% if loop.first %}active{% endif %} me-2"
                onclick="selectFileModel(this,'{{name}}')" data-model="{{name}}">{{name}}</span>
          {% endfor %}
        </div>

        <!-- Drop zone -->
        <div class="drop-zone mb-3" id="dropZone"
             onclick="document.getElementById('fileInput').click()"
             ondragover="event.preventDefault();this.classList.add('drag-over')"
             ondragleave="this.classList.remove('drag-over')"
             ondrop="handleDrop(event)">
          <div class="icon mb-2">📄</div>
          <div class="fw-semibold text-primary">Kéo thả file vào đây hoặc nhấn để chọn</div>
          <div class="text-muted small mt-1">Hỗ trợ: Excel (.xlsx, .xls) và CSV (.csv)</div>
          <div class="text-muted small">Cột dữ liệu: <code>sentence</code> hoặc cột đầu tiên chứa văn bản</div>
          <input type="file" id="fileInput" accept=".xlsx,.xls,.csv" class="d-none"
                 onchange="handleFile(this.files[0])">
        </div>

        <!-- File info -->
        <div id="fileInfo" class="d-none alert alert-info py-2 mb-3"></div>

        <!-- Progress -->
        <div id="fileProgress" class="mb-3">
          <div class="d-flex justify-content-between mb-1">
            <small class="text-muted" id="progressText">Đang phân tích...</small>
            <small class="text-muted" id="progressPct">0%</small>
          </div>
          <div class="progress" style="height:8px;border-radius:8px">
            <div id="progressBar" class="progress-bar bg-primary" style="width:0%"></div>
          </div>
        </div>

        <button id="btnAnalyzeFile" class="btn btn-primary w-100 py-2 d-none" onclick="analyzeFile()">
          <span id="fileBtnTxt">🔍 Phân tích toàn bộ file</span>
          <span id="fileSpin" class="spinner-border ms-2 d-none"></span>
        </button>
      </div>
    </div>

    <!-- Kết quả file -->
    <div id="fileResult" class="d-none fade-in"></div>
  </div>

  <!-- ═══ TAB 3: SO SÁNH KẾT QUẢ MÔ HÌNH ════════════════════ -->
  <div id="tab-compare" class="tab-pane">
    <div class="card mb-4" style="border-radius:0 16px 16px 16px">
      <div class="card-body p-4">
        <h6 class="fw-bold mb-3">📊 Kết quả thực nghiệm trên Test Set UIT-VSFC</h6>
        {% if model_results %}
        <div class="table-responsive">
          <table class="table table-hover align-middle result-table">
            <thead><tr>
              <th>Mô hình</th><th class="text-center">Accuracy</th>
              <th class="text-center">F1-macro</th>
              <th>F1-macro (biểu đồ)</th>
            </tr></thead>
            <tbody>
              {% for name,r in model_results.items() %}
              <tr {% if r.f1_macro==best_f1 %}class="table-success fw-bold"{% endif %}>
                <td>{{name}} {% if r.f1_macro==best_f1 %}<span class="badge bg-success">BEST</span>{% endif %}</td>
                <td class="text-center">{{"%.2f"|format(r.accuracy*100)}}%</td>
                <td class="text-center">{{"%.4f"|format(r.f1_macro)}}</td>
                <td style="min-width:200px">
                  <div class="progress-wrap">
                    <div class="progress-fill {% if r.f1_macro==best_f1 %}bg-success{% else %}bg-primary{% endif %}"
                         style="width:{{(r.f1_macro*100)|round}}%"></div>
                  </div>
                </td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
        {% else %}
        <div class="alert alert-warning">Chưa có kết quả thực nghiệm. Chạy step3→step5b trước.</div>
        {% endif %}
      </div>
    </div>
  </div>

</div><!-- /container -->

<footer class="text-center py-4 text-muted small mt-4">
  Đồ án tốt nghiệp · Phân tích cảm xúc phản hồi sinh viên · PhoBERT & Học sâu · 2025
</footer>

<script>
// ── Tab switching ──────────────────────────────────────────
function switchTab(id){
  document.querySelectorAll('.tab-pane').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+id).classList.add('active');
  event.target.classList.add('active');
}

// ── Model selection ────────────────────────────────────────
let selectedModel = '{{models[0] if models else "PhoBERT"}}';
let selectedFileModel = '{{models[0] if models else "PhoBERT"}}';

function selectModel(el,name){
  el.closest('.card-body').querySelectorAll('[data-model]').forEach(b=>b.classList.remove('active'));
  el.classList.add('active'); selectedModel=name;
}
function selectFileModel(el,name){
  el.closest('.card-body').querySelectorAll('[data-model]').forEach(b=>b.classList.remove('active'));
  el.classList.add('active'); selectedFileModel=name;
}
function setEx(text){ document.getElementById('inputText').value=text; }

// ── Phân tích câu đơn ─────────────────────────────────────
function analyze(){
  const text=document.getElementById('inputText').value.trim();
  if(!text){alert('Vui lòng nhập câu!');return;}
  if(text.split(' ').length<2){alert('Câu quá ngắn!');return;}
  setBtnLoading('btnTxt','btnSpin',true,'Đang phân tích...');
  fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({text,model:selectedModel})})
  .then(r=>r.json()).then(data=>{
    setBtnLoading('btnTxt','btnSpin',false,'🔍 Phân tích cảm xúc');
    if(data.error){alert(data.error);return;}
    showSingleResult(data,text);
  }).catch(e=>{setBtnLoading('btnTxt','btnSpin',false,'🔍 Phân tích cảm xúc');alert('Lỗi: '+e.message);});
}

function showSingleResult(data,text){
  const sec=document.getElementById('singleResult');
  sec.classList.remove('d-none');
  if(data.compare){
    let html='<div class="row g-3">';
    for(const [mName,res] of Object.entries(data.compare)){
      const cls=res.label_id===2?'positive':res.label_id===0?'negative':'neutral';
      html+=`<div class="col-md-6"><div class="result-card result-${cls}">
        <div class="d-flex justify-content-between align-items-center mb-2">
          <b>${mName}</b><span class="badge bg-white text-dark">${res.confidence}%</span>
        </div>
        <div class="text-center my-2"><div style="font-size:2.2rem">${res.icon}</div>
          <div class="fw-bold fs-5">${res.label}</div></div>
        ${probBars(res.probs)}</div></div>`;
    }
    html+='</div>';
    sec.innerHTML=html;
  } else {
    const res=data.result;
    const cls=res.label_id===2?'positive':res.label_id===0?'negative':'neutral';
    sec.innerHTML=`<div class="result-card result-${cls}">
      <div class="d-flex justify-content-between mb-3">
        <div><div class="text-white-50 small">Model: ${data.model}</div>
          <div class="fw-bold">"${text.length>80?text.substring(0,80)+'...':text}"</div></div>
        <div style="font-size:2.8rem;line-height:1">${res.icon}</div>
      </div>
      <div class="row align-items-center">
        <div class="col-auto">
          <div style="font-size:1.6rem;font-weight:700">${res.label}</div>
          <div class="text-white-50 small">${res.confidence}% confidence</div>
        </div>
        <div class="col">${probBars(res.probs)}</div>
      </div></div>`;
  }
  sec.scrollIntoView({behavior:'smooth',block:'nearest'});
}

function probBars(probs){
  const colors={'Tích cực':'#1D9E75','Trung tính':'#F59E0B','Tiêu cực':'#E24B4A'};
  let html='';
  for(const [name,val] of Object.entries(probs)){
    html+=`<div class="d-flex align-items-center gap-2 mb-1">
      <span class="small text-white-50" style="width:82px">${name}</span>
      <div class="flex-grow-1 bg-white bg-opacity-25 rounded" style="height:10px">
        <div class="prob-bar" style="width:${val}%;background:rgba(255,255,255,.8)"></div>
      </div>
      <span class="small fw-bold" style="width:46px;text-align:right">${val}%</span>
    </div>`;
  }
  return html;
}

// ── Upload file ────────────────────────────────────────────
let uploadedFile=null;
let parsedData=[];

function handleDrop(e){
  e.preventDefault();
  document.getElementById('dropZone').classList.remove('drag-over');
  const file=e.dataTransfer.files[0];
  if(file) handleFile(file);
}

function handleFile(file){
  if(!file)return;
  if(!/\.(xlsx|xls|csv)$/i.test(file.name)){
    alert('Chỉ hỗ trợ file .xlsx, .xls hoặc .csv!');return;
  }
  uploadedFile=file;
  document.getElementById('fileInfo').classList.remove('d-none');
  document.getElementById('fileInfo').innerHTML=
    `📄 <b>${file.name}</b> — ${(file.size/1024).toFixed(1)} KB — Sẵn sàng phân tích`;
  document.getElementById('btnAnalyzeFile').classList.remove('d-none');
  document.getElementById('fileResult').classList.add('d-none');
}

function analyzeFile(){
  if(!uploadedFile){alert('Vui lòng chọn file trước!');return;}
  setBtnLoading('fileBtnTxt','fileSpin',true,'Đang phân tích...');
  document.getElementById('fileProgress').style.display='block';
  updateProgress(0,'Đang đọc file...');

  const formData=new FormData();
  formData.append('file',uploadedFile);
  formData.append('model',selectedFileModel);

  // Giả lập progress
  let prog=5;
  const progInterval=setInterval(()=>{
    prog=Math.min(prog+2,90);
    updateProgress(prog,'Đang phân tích...');
  },500);

  fetch('/analyze_file',{method:'POST',body:formData})
  .then(r=>r.json()).then(data=>{
    clearInterval(progInterval);
    updateProgress(100,'Hoàn thành!');
    setBtnLoading('fileBtnTxt','fileSpin',false,'🔍 Phân tích toàn bộ file');
    if(data.error){alert(data.error);return;}
    showFileResult(data);
  }).catch(e=>{
    clearInterval(progInterval);
    setBtnLoading('fileBtnTxt','fileSpin',false,'🔍 Phân tích toàn bộ file');
    document.getElementById('fileProgress').style.display='none';
    alert('Lỗi: '+e.message);
  });
}

function updateProgress(pct,text){
  document.getElementById('progressBar').style.width=pct+'%';
  document.getElementById('progressPct').textContent=pct+'%';
  document.getElementById('progressText').textContent=text;
}

function showFileResult(data){
  document.getElementById('fileProgress').style.display='none';
  const sec=document.getElementById('fileResult');
  sec.classList.remove('d-none');

  const {total,counts,model,chart_b64,preview,download_id}=data;
  const pos=counts['Tích cực']||0, neg=counts['Tiêu cực']||0, neu=counts['Trung tính']||0;

  let html=`
  <div class="card mb-3">
    <div class="card-body p-4">
      <div class="d-flex justify-content-between align-items-center mb-3">
        <h6 class="fw-bold mb-0">📊 Kết quả phân tích — ${total.toLocaleString()} câu (Model: ${model})</h6>
        <a href="/download/${download_id}" class="btn btn-success btn-sm">⬇️ Tải Excel</a>
      </div>
      <!-- Thống kê -->
      <div class="row g-3 mb-4">
        <div class="col-4">
          <div class="stat-box" style="background:#E1F5EE">
            <div class="fs-2 fw-bold text-success">${pos.toLocaleString()}</div>
            <div class="small">😊 Tích cực</div>
            <div class="small text-muted">${(pos/total*100).toFixed(1)}%</div>
          </div>
        </div>
        <div class="col-4">
          <div class="stat-box" style="background:#FAECE7">
            <div class="fs-2 fw-bold text-danger">${neg.toLocaleString()}</div>
            <div class="small">😞 Tiêu cực</div>
            <div class="small text-muted">${(neg/total*100).toFixed(1)}%</div>
          </div>
        </div>
        <div class="col-4">
          <div class="stat-box" style="background:#FFF3CD">
            <div class="fs-2 fw-bold text-warning">${neu.toLocaleString()}</div>
            <div class="small">😐 Trung tính</div>
            <div class="small text-muted">${(neu/total*100).toFixed(1)}%</div>
          </div>
        </div>
      </div>
      <!-- Biểu đồ -->
      <img src="data:image/png;base64,${chart_b64}" class="chart-img w-100 mb-4" alt="Biểu đồ phân tích">
      <!-- Preview bảng -->
      <h6 class="fw-bold mb-2">📋 Xem trước kết quả (10 dòng đầu)</h6>
      <div class="table-responsive">
        <table class="table table-sm table-hover result-table">
          <thead><tr>
            <th>#</th><th>Câu phản hồi</th>
            <th class="text-center">Cảm xúc</th>
            <th class="text-center">Conf.</th>
            <th class="text-center">P(+)</th>
            <th class="text-center">P(0)</th>
            <th class="text-center">P(-)</th>
          </tr></thead>
          <tbody>`;

  for(const row of preview){
    const badgeCls=row.label==='Tích cực'?'badge-pos':row.label==='Tiêu cực'?'badge-neg':'badge-neu';
    const icon=row.label==='Tích cực'?'😊':row.label==='Tiêu cực'?'😞':'😐';
    html+=`<tr>
      <td class="text-muted">${row.idx}</td>
      <td>${row.sentence.length>60?row.sentence.substring(0,60)+'...':row.sentence}</td>
      <td class="text-center"><span class="badge ${badgeCls}">${icon} ${row.label}</span></td>
      <td class="text-center">${row.confidence}%</td>
      <td class="text-center text-success">${row.prob_pos}%</td>
      <td class="text-center text-warning">${row.prob_neu}%</td>
      <td class="text-center text-danger">${row.prob_neg}%</td>
    </tr>`;
  }
  html+=`</tbody></table></div></div></div>`;
  sec.innerHTML=html;
  sec.scrollIntoView({behavior:'smooth',block:'nearest'});
}

// ── Helpers ────────────────────────────────────────────────
function setBtnLoading(txtId,spinId,loading,label){
  document.getElementById(txtId).textContent=label;
  document.getElementById(spinId).classList.toggle('d-none',!loading);
}

// Ctrl+Enter để phân tích
document.addEventListener('keydown',e=>{if(e.ctrlKey&&e.key==='Enter')analyze();});
</script>
</body>
</html>
"""

# ══════════════════════════════════════════════════════════════
#  FLASK APP
# ══════════════════════════════════════════════════════════════
app    = Flask(__name__)
TEMP_RESULTS = {}  # lưu kết quả tạm để download

EXAMPLES = [
    ("Thầy dạy rất nhiệt tình, giải thích rõ ràng và dễ hiểu.", 2),
    ("Môn học quá khó, tài liệu không đủ, thi cử quá nặng.", 0),
    ("Môn học bình thường, không có gì đặc biệt để khen hay chê.", 1),
    ("Cô dạy hay, kiến thức thực tế, em học được nhiều thứ bổ ích.", 2),
    ("Giáo viên giảng nhanh quá, sinh viên không theo kịp bài.", 0),
]

@app.route("/")
def index():
    best_f1 = max((v["f1_macro"] for v in model_results.values()), default=0)
    return render_template_string(HTML,
        models=list(models_loaded.keys()),
        examples=EXAMPLES,
        model_results=model_results,
        best_f1=best_f1)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data       = request.get_json()
        text       = data.get("text","").strip()
        model_name = data.get("model","PhoBERT")
        if not text: return jsonify({"error":"Vui lòng nhập câu!"})
        if model_name == "so_sanh":
            compare = {n: predict_one(text, m) for n,m in models_loaded.items()}
            return jsonify({"compare": compare, "model":"So sánh"})
        if model_name not in models_loaded:
            model_name = list(models_loaded.keys())[0]
        return jsonify({"result": predict_one(text, models_loaded[model_name]),
                        "model": model_name})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/analyze_file", methods=["POST"])
def analyze_file():
    try:
        file       = request.files.get("file")
        model_name = request.form.get("model","PhoBERT")
        if not file: return jsonify({"error":"Không nhận được file!"})

        # Đọc file
        fname = file.filename.lower()
        if fname.endswith(".csv"):
            df = pd.read_csv(file, encoding="utf-8-sig")
        else:
            df = pd.read_excel(file)

        # Tìm cột văn bản
        text_col = None
        for col in df.columns:
            if str(col).lower() in ["sentence","text","câu","van_ban","noi_dung","content","feedback"]:
                text_col = col; break
        if text_col is None:
            text_col = df.columns[0]

        # Lấy texts
        texts = df[text_col].fillna("").astype(str).tolist()
        texts = [t.strip() for t in texts if t.strip() and len(t.strip()) > 2]

        if len(texts) == 0:
            return jsonify({"error": "Không tìm thấy câu nào trong file!"})
        if len(texts) > 5000:
            return jsonify({"error": f"File quá lớn ({len(texts):,} câu). Tối đa 5.000 câu."})

        # Dự đoán
        if model_name not in models_loaded:
            model_name = list(models_loaded.keys())[0]
        model = models_loaded[model_name]
        preds = predict_batch(texts, model)

        # Tạo DataFrame kết quả
        result_df = pd.DataFrame({
            "sentence":   texts,
            "label_id":   [p["label_id"]   for p in preds],
            "label":      [p["label"]       for p in preds],
            "confidence": [p["confidence"]  for p in preds],
            "prob_pos":   [p["prob_pos"]    for p in preds],
            "prob_neu":   [p["prob_neu"]    for p in preds],
            "prob_neg":   [p["prob_neg"]    for p in preds],
        })

        # Thống kê
        counts = result_df["label"].value_counts().to_dict()

        # Vẽ biểu đồ
        chart_b64 = make_charts(result_df)

        # Lưu tạm để download
        import uuid
        did = str(uuid.uuid4())[:8]
        TEMP_RESULTS[did] = result_df

        # Preview 10 dòng
        preview = []
        for i, row in result_df.head(10).iterrows():
            preview.append({
                "idx":        i+1,
                "sentence":   row["sentence"],
                "label":      row["label"],
                "confidence": row["confidence"],
                "prob_pos":   row["prob_pos"],
                "prob_neu":   row["prob_neu"],
                "prob_neg":   row["prob_neg"],
            })

        return jsonify({
            "total":       len(result_df),
            "counts":      counts,
            "model":       model_name,
            "chart_b64":   chart_b64,
            "preview":     preview,
            "download_id": did,
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/download/<did>")
def download(did):
    if did not in TEMP_RESULTS:
        return "Kết quả không tìm thấy hoặc đã hết hạn", 404
    df = TEMP_RESULTS[did]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Kết quả phân tích")
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name="ket_qua_phan_tich_cam_xuc.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    print("\n" + "="*55)
    print("DEMO WEB V2 — SẴN SÀNG!")
    print("="*55)
    print(f"  Models: {list(models_loaded.keys())}")
    print(f"\n  ➜  http://localhost:{PORT}")
    print(f"  Nhấn Ctrl+C để dừng")
    print("="*55+"\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)