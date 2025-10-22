import os
import re
import numpy as np
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import AutoTokenizer, AutoModel
import torch

# ============================================================
# 1️⃣ Thiết lập token và đường dẫn
# ============================================================

os.environ["HF_TOKEN"] = "add token here"

INPUT_FILE = r"D:\du_an_doc_sach_tranformer\data\Bạch tuyết và 7 chú lùn.txt"
OUTPUT_DIR = r"D:\du_an_doc_sach_tranformer\data_xong"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "bach_tuyet_embeddings.npy")
METADATA_FILE = os.path.join(OUTPUT_DIR, "bach_tuyet_metadata.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        text = f.read().strip()
except FileNotFoundError:
    print(f"❌ File {INPUT_FILE} not found.")
    raise

# ============================================================
# 2️⃣ Model embedding local: intfloat/multilingual-e5-large
# ============================================================

MODEL_NAME = "intfloat/multilingual-e5-large"
CACHE_DIR = r"D:\du_an_doc_sach_tranformer\file_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ["HF_TOKEN"], cache_dir=CACHE_DIR)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    token=os.environ["HF_TOKEN"],
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)
model.eval()
device = model.device
print(f"✅ Loaded embedding model: {MODEL_NAME} on {device}")

# ============================================================
# 3️⃣ Hàm chia đoạn văn
# ============================================================

def split_text_into_chunks(text, min_words_per_chunk=25, max_words_per_chunk=40):
    cleaned_text = re.sub(r'\n\s*\n+', '\n', text.strip())
    paragraphs = [p.strip() for p in cleaned_text.split('\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        sentences = re.split(r'(?<=[\.\!\?])\s+', para)
        for sent in sentences:
            words = sent.split()
            if not words:
                continue

            while len(words) > max_words_per_chunk:
                sub_chunk = " ".join(words[:max_words_per_chunk])
                chunks.append(sub_chunk)
                words = words[max_words_per_chunk:]

            sent_len = len(words)
            if current_len + sent_len > max_words_per_chunk:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [" ".join(words)]
                current_len = sent_len
            else:
                current_chunk.append(" ".join(words))
                current_len += sent_len

            if re.search(r'[\.!?]$', sent.strip()):
                if current_chunk:
                    chunk_text = " ".join(current_chunk).strip()
                    if len(chunk_text.split()) >= min_words_per_chunk:
                        chunks.append(chunk_text)
                    current_chunk = []
                    current_len = 0

    if current_chunk:
        chunk_text = " ".join(current_chunk).strip()
        if len(chunk_text.split()) >= min_words_per_chunk:
            chunks.append(chunk_text)

    return chunks

# ============================================================
# 4️⃣ Encode LOCAL bằng e5-large (mean pooling + normalize)
# ============================================================

def encode_text_local(text):
    """Encode 1 đoạn văn bản bằng model local"""
    prefixed = f"passage: {text}"  # E5 cần prefix "passage:"
    inputs = tokenizer(prefixed, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state  # [1, seq_len, dim]
        vec = hidden.mean(dim=1)            # Mean pooling
    vec = vec.cpu().numpy().astype("float32")
    # Normalize L2
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    return vec

# ============================================================
# 5️⃣ Tạo embeddings và metadata
# ============================================================

chunks = split_text_into_chunks(text, min_words_per_chunk=25, max_words_per_chunk=40)
lengths = [len(c.split()) for c in chunks]
print(f"📜 Tổng số đoạn: {len(chunks)}")
print(f"📏 Trung bình: {np.mean(lengths):.1f} từ | Min: {min(lengths)}, Max: {max(lengths)}")

embeddings = []
metadata = []

for i, chunk in enumerate(chunks):
    try:
        vec = encode_text_local(chunk)
        embeddings.append(vec)
        metadata.append({
            "chunk_idx": i,
            "text": chunk,
            "num_words": len(chunk.split()),
            "start_char": text.find(chunk) if i == 0 else text.find(chunk, metadata[-1]["start_char"] + len(metadata[-1]["text"]))
        })
        print(f"✅ Chunk {i+1}/{len(chunks)} encoded | dim={vec.shape[1]} | words={len(chunk.split())}")
    except Exception as e:
        print(f"❌ Lỗi chunk {i+1}: {e}")
        metadata.append({"chunk_idx": i, "text": chunk, "error": str(e)})

# ============================================================
# 6️⃣ Lưu embeddings & metadata
# ============================================================

embeddings = [e for e in embeddings if e is not None]
if embeddings:
    embedding_np = np.vstack(embeddings).astype(np.float32)
    np.save(OUTPUT_FILE, embedding_np)
    print(f"🎯 Lưu {embedding_np.shape[0]} vectors (dim={embedding_np.shape[1]}) → {OUTPUT_FILE}")
else:
    print("⚠️ Không có embedding nào được tạo.")

with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
print(f"📋 Metadata saved → {METADATA_FILE}")
