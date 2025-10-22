
import json
import numpy as np
import faiss
from huggingface_hub import InferenceClient, login
import os
from transformers import AutoTokenizer, AutoModel
import torch

# -------------------------------
# 1️⃣ Thiết lập đường dẫn và token
# -------------------------------

NPY_PATH = r"D:\du_an_doc_sach_tranformer\data_xong\bach_tuyet_embeddings.npy"
JSON_PATH = r"D:\du_an_doc_sach_tranformer\data_xong\bach_tuyet_metadata.json"
HF_TOKEN = "add token here"  # token của bạn

if HF_TOKEN and HF_TOKEN.startswith("hf_"):
    login(token=HF_TOKEN)
    print("✅ Đăng nhập Hugging Face thành công!")
else:
    print("⚠️ Không dùng token (có thể giới hạn rate).")

# Kiểm tra dữ liệu
assert os.path.exists(NPY_PATH), f"Không tìm thấy: {NPY_PATH}"
assert os.path.exists(JSON_PATH), f"Không tìm thấy: {JSON_PATH}"
print("✅ Dữ liệu sẵn sàng!")

# -------------------------------
# 2️⃣ Load embeddings & metadata
# -------------------------------
print("🔹 Đang load embeddings & metadata...")
embeddings = np.load(NPY_PATH).astype("float32")
with open(JSON_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

metadata = [m for m in metadata if "error" not in m]
min_len = min(len(embeddings), len(metadata))
embeddings, metadata = embeddings[:min_len], metadata[:min_len]

# FAISS index
d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
faiss.normalize_L2(embeddings)
index.add(embeddings)
print(f"✅ FAISS index: {index.ntotal} vectors (dim={d})")

# -------------------------------
# 3️⃣ Model sinh (API) và model embed (LOCAL - FIX)
# -------------------------------
GEN_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
EMBED_MODEL = "intfloat/multilingual-e5-large"  # Giữ, nhưng load local

gen_client = InferenceClient(model=GEN_MODEL, token=HF_TOKEN)

# FIX: Load LOCAL cho embed (tải ~500MB lần đầu, cache sau)
CACHE_DIR = r"D:\du_an_doc_sach_tranformer\file_model"
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, token=HF_TOKEN, cache_dir=CACHE_DIR)
embed_model = AutoModel.from_pretrained(
    EMBED_MODEL,
    token=HF_TOKEN,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)
embed_model.eval()
print(f"✅ Model sinh (API): {GEN_MODEL}")
print(f"✅ Model embed (local): {EMBED_MODEL} trên {'GPU' if torch.cuda.is_available() else 'CPU'}")

# -------------------------------
# 4️⃣ Hàm encode query + FAISS search (FIXED - Local)
# -------------------------------
def encode_query_local(text):
    """Encode query LOCAL (mean pooling chuẩn + normalize L2)"""
    try:
        inputs = embed_tokenizer(
            f"query: {text}",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(embed_model.device)

        with torch.no_grad():
            outputs = embed_model(**inputs)
            last_hidden = outputs.last_hidden_state
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            vec = (last_hidden * attention_mask).sum(1) / attention_mask.sum(1)
            vec = vec.cpu().numpy().astype("float32")

        faiss.normalize_L2(vec)
        print("🧩 Query embedding shape:", vec.shape)
        return vec
    except Exception as e:
        print("⚠️ Lỗi encode_query_local:", e)
        dummy = embeddings[0:1].copy()
        faiss.normalize_L2(dummy)
        return dummy


def search_faiss_api(query_text, k=5):
    q_emb = encode_query_local(query_text)
    if q_emb.shape[1] != d:
        print(f"⚠️ Dimension mismatch! Query={q_emb.shape[1]}, Index={d}")
        return []
    scores, indices = index.search(q_emb, k)
    results = []
    for sc, idx in zip(scores[0], indices[0]):
        idx = int(idx)
        if 0 <= idx < len(metadata):
            results.append({
                "score": float(sc),
                "chunk_idx": metadata[idx].get("chunk_idx", idx),
                "text": metadata[idx]["text"]
            })
    return results

# -------------------------------
# 5️⃣ Build prompt và sinh text
# -------------------------------
def build_messages(question, retrieved):
    system = {
        "role": "system",
        "content": (
            "Bạn là trợ lý AI tiếng Việt. "
            "Chỉ dựa vào các đoạn văn dưới để trả lời. "
            "Nếu không có thông tin, nói 'Không tìm thấy trong tài liệu.'"
        )
    }
    passages = []
    for i, r in enumerate(retrieved):
        passages.append(f"[{i}] (score={r['score']:.3f}):\n{r['text'][:800]}")
    user = {
        "role": "user",
        "content": (
            "Các đoạn văn:\n\n" +
            "\n\n".join(passages) +
            f"\n\nCâu hỏi: {question}\n\nTrả lời tự nhiên, chi tiết bằng tiếng Việt."
        )
    }
    return [system, user]

def generate_with_api(messages, max_new_tokens=256):
    try:
        response = gen_client.chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.7,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Chat completion fail ({e})")
        return "Không thể sinh câu trả lời."

# -------------------------------
# 6️⃣ Pipeline RAG
# -------------------------------
def rag_pipeline(question, top_k=3):
    retrieved = search_faiss_api(question, k=top_k)
    messages = build_messages(question, retrieved)
    answer = generate_with_api(messages)

    print("\n=== 🔍 Các đoạn văn RAG ===")
    for i, r in enumerate(retrieved):
        print(f"[{i}] score={r['score']:.4f}")
        print(r['text'][:300].replace("\n", " ") + ("..." if len(r['text']) > 300 else ""))
        print("----")

    print("\n=== 💬 Câu trả lời LLM ===")
    print(answer)
    return answer

# ============================================================
# 🔍 THÊM PHẦN TEST DEBUG TỰ ĐỘNG Ở ĐÂY
# ============================================================

print("\n================= 🔧 KIỂM TRA MÔI TRƯỜNG =================")
print("DEVICE:", "cuda" if torch.cuda.is_available() else "cpu")
print("Embeddings shape:", embeddings.shape)
print("Index ntotal:", index.ntotal)
print("d (dim):", d)
print("Metadata length:", len(metadata))
if len(metadata) > 0:
    print("Metadata[0] keys:", list(metadata[0].keys()))

print("\n================= 🧠 TEST ENCODE QUERY LOCAL =================")
try:
    q_test = "Kiểm tra embedding của câu hỏi"
    vec = encode_query_local(q_test)
    print("vec.shape:", getattr(vec, "shape", None))
    print("vec (first 10 values):", vec.flatten()[:10])
    print("vec L2 norm:", np.linalg.norm(vec))

    if len(metadata) > 0:
        text0 = metadata[0]["text"]
        vec0 = encode_query_local(text0)
        scores_with_first = np.dot(embeddings[:5], vec0.flatten())
        print("Dot scores with first 5 embeddings:", scores_with_first)
except Exception as e:
    print("❌ Encode test failed:", e)

print("\n================= 🔎 TEST FAISS SEARCH =================")
if len(metadata) > 0:
    q_search = metadata[0]["text"][:200]
    results = search_faiss_api(q_search, k=5)
    print("Search results (top k):")
    for i, r in enumerate(results):
        print(i, "score:", r["score"], "chunk_idx:", r["chunk_idx"])
else:
    print("⚠️ No metadata to test search.")

print("\n================= 💬 TEST GENERATION (API) =================")
try:
    test_msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hãy nói một câu tiếng Việt ngắn: Xin chào thế giới!"}
    ]
    resp = gen_client.chat_completion(messages=test_msgs, max_tokens=32, temperature=0.3)
    print("resp type:", type(resp))
    try:
        print("choices keys:", [c.keys() for c in resp.choices])
        print("first message:", resp.choices[0].message.content)
    except Exception as e:
        print("Cannot parse resp, dump repr:", repr(resp)[:800])
except Exception as e:
    print("❌ API generation test failed:", e)

print("\n================= ✅ TEST PIPELINE MINI =================")
try:
    if len(metadata) > 0:
        q_demo = metadata[0]["text"][:60]
    else:
        q_demo = "Nội dung thử nghiệm"
    print("Query:", q_demo)
    retr = search_faiss_api(q_demo, k=3)
    print("Retrieved count:", len(retr))
    for i, r in enumerate(retr):
        print(f"[{i}] score={r['score']:.4f} len_text={len(r['text'])}")
    msgs = build_messages(q_demo, retr)
    print("SYSTEM:", msgs[0]["content"][:200])
    print("USER:", msgs[1]["content"][:400])
    ans = generate_with_api(msgs, max_new_tokens=64)
    print("Generation result:", ans)
except Exception as e:
    print("❌ Pipeline mini test failed:", e)

# ============================================================
# 7️⃣ Run interactive
# ============================================================
if __name__ == "__main__":
    print("\n📚 Hệ thống RAG Bạch Tuyết (local embed + API gen) sẵn sàng!")
    while True:
        q = input("\n❓ Câu hỏi (hoặc 'exit'):\n> ").strip()
        if q.lower() == "exit":
            break
        if q:
            print("\n--- 🔬 TEST ENCODE QUERY SAU INPUT ---")
            q_vec = encode_query_local(q)
            print("Shape:", q_vec.shape)
            print("First 10 values:", q_vec.flatten()[:10])
            print("L2 norm:", np.linalg.norm(q_vec))
            print("--- 🔬 END ENCODE TEST ---\n")

            rag_pipeline(q, top_k=3)
