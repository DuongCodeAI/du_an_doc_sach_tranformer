import gradio as gr
import numpy as np
import json
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import InferenceClient, login
import os

# ===========================
# ⚙️ CẤU HÌNH MODEL & ĐƯỜNG DẪN
# ===========================
# HF_TOKEN = "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"

GEN_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
EMBED_MODEL = "intfloat/multilingual-e5-large"

CACHE_DIR = r"D:\du_an_doc_sach_tranformer\file_model"
NPY_PATH = r"D:\du_an_doc_sach_tranformer\data_xong\bach_tuyet_embeddings.npy"
JSON_PATH = r"D:\du_an_doc_sach_tranformer\data_xong\bach_tuyet_metadata.json"

# ===========================
# 🔧 KHỞI TẠO
# ===========================
print("🚀 Khởi tạo model và FAISS index...")

if HF_TOKEN and HF_TOKEN.startswith("hf_"):
    login(token=HF_TOKEN)

# Load FAISS index
embeddings = np.load(NPY_PATH).astype("float32")
with open(JSON_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)
metadata = [m for m in metadata if "error" not in m]
min_len = min(len(embeddings), len(metadata))
embeddings, metadata = embeddings[:min_len], metadata[:min_len]

d = embeddings.shape[1]
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(d)
index.add(embeddings)

# Load embedding model
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, token=HF_TOKEN, cache_dir=CACHE_DIR)
embed_model = AutoModel.from_pretrained(
    EMBED_MODEL,
    token=HF_TOKEN,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)
embed_model.eval()

# Load LLM client
gen_client = InferenceClient(model=GEN_MODEL, token=HF_TOKEN)

# ===========================
# 🧠 HÀM XỬ LÝ
# ===========================
def encode_query(text):
    """Encode câu hỏi bằng E5-large local"""
    inputs = embed_tokenizer(
        f"query: {text}",
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(embed_model.device)

    with torch.no_grad():
        outputs = embed_model(**inputs)
        last_hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        emb = (last_hidden * mask).sum(1) / mask.sum(1)
        emb = emb.cpu().numpy().astype("float32")
        faiss.normalize_L2(emb)
    return emb


def search_faiss(query, k=3):
    q_emb = encode_query(query)
    scores, indices = index.search(q_emb, k)
    results = []
    for s, i in zip(scores[0], indices[0]):
        i = int(i)
        if 0 <= i < len(metadata):
            results.append({"score": float(s), "text": metadata[i]["text"]})
    return results


def generate_answer(question, retrieved, max_new_tokens=256):
    """Gọi Llama sinh câu trả lời dựa vào đoạn retrieve được"""
    context = "\n\n".join([f"[{i}] {r['text']}" for i, r in enumerate(retrieved)])
    messages = [
        {"role": "system", "content": "Bạn là trợ lý AI tiếng Việt, chỉ dựa vào tài liệu bên dưới để trả lời."},
        {"role": "user", "content": f"Tài liệu:\n{context}\n\nCâu hỏi: {question}"}
    ]

    try:
        resp = gen_client.chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.7
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Lỗi khi sinh câu trả lời: {e}"


# ===========================
# 💬 HÀM XỬ LÝ GRADIO
# ===========================
def rag_chat(question):
    if not question.strip():
        return "❗ Vui lòng nhập câu hỏi.", ""

    retrieved = search_faiss(question, k=3)
    answer = generate_answer(question, retrieved)
    passages = "\n\n".join([f"[{i}] (score={r['score']:.3f})\n{r['text']}" for i, r in enumerate(retrieved)])
    return answer, passages


# ===========================
# 🖥️ GIAO DIỆN GRADIO
# ===========================
with gr.Blocks(title="RAG Bạch Tuyết") as demo:
    gr.Markdown("## 📚 Chat RAG: Bạch Tuyết và 7 chú lùn\nNhập câu hỏi để tra cứu thông tin từ truyện.")

    with gr.Row():
        with gr.Column(scale=1):
            question = gr.Textbox(label="💭 Câu hỏi của bạn", placeholder="Ví dụ: Vì sao Bạch Tuyết bị trúng độc?")
            run_btn = gr.Button("🔍 Truy vấn", variant="primary")
        with gr.Column(scale=2):
            answer = gr.Textbox(label="🧠 Câu trả lời", lines=5)
            passages = gr.Textbox(label="📖 Đoạn văn được truy xuất", lines=8)

    run_btn.click(fn=rag_chat, inputs=question, outputs=[answer, passages])

    gr.Markdown("---")
    gr.Markdown("🚀 Hệ thống RAG (FAISS + Llama 3.2 + E5-large) - chạy local hoàn toàn")

# ===========================
# 🚀 CHẠY ỨNG DỤNG (AUTO PORT & LOCALHOST)
# ===========================
if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=None, share=False)
