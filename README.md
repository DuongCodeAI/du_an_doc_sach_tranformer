
# 🦙✨ Chatbot - Truyện Cổ Tích Việt Nam( faiss + rag + tranformer)

> Dự án xây dựng chatbot hỏi–đáp (QA) về các **truyện cổ tích Việt Nam**, ứng dụng công nghệ **Transformer**, **FAISS**, và **RAG (Retrieval-Augmented Generation)**.
> Kết hợp sức mạnh giữa **model multilingual-e5-large** để encode dữ liệu và ** model Llama 3.2** để sinh câu trả lời tự nhiên bằng tiếng Việt.

---

## 🌸 Mục lục

* [1️⃣ Giới thiệu tổng quan](#1️⃣-giới-thiệu-tổng-quan)
* [2️⃣ Cấu trúc dự án](#2️⃣-cấu-trúc-dự-án)
* [3️⃣ Công nghệ sử dụng](#3️⃣-công-nghệ-sử-dụng)
* [4️⃣ Luồng hoạt động](#4️⃣-luồng-hoạt-động)
* [5️⃣ Hướng dẫn cài đặt & chạy](#5️⃣-hướng-dẫn-cài-đặt--chạy)
* [6️⃣ Giao diện minh họa](#6️⃣-giao-diện-minh-họa)
* [7️⃣ Định hướng phát triển](#7️⃣-định-hướng-phát-triển)
* [8️⃣ Tác giả & liên hệ](#8️⃣-tác-giả--liên-hệ)

---

## 1️⃣ Giới thiệu tổng quan

Dự án này giúp người dùng **đặt câu hỏi** và **nhận câu trả lời tự động** từ các **tập truyện cổ tích Việt Nam** thông qua kỹ thuật **RAG** – kết hợp giữa **retrieval (tìm kiếm thông tin)** và **generation (sinh câu trả lời)**.

🧠 Cụ thể:

* Dữ liệu là **các file `.txt` truyện cổ tích Việt Nam**.
* Mỗi truyện được **chia nhỏ (chunk)**, sau đó **embedding** bằng mô hình **E5**.
* Khi người dùng hỏi, hệ thống dùng **FAISS + RAG** để tìm các đoạn văn liên quan.
* Sau đó, mô hình **Llama-3.2-3B-Instruct** sinh ra **câu trả lời tiếng Việt tự nhiên** dựa trên ngữ cảnh đó.

---

## 2️⃣ Cấu trúc dự án

```
D:\du_an_doc_sach_tranformer
│
├── data/                        # Thư mục chứa dữ liệu gốc (.txt các truyện cổ tích)
│
├── data_xong/                   # Dữ liệu sau khi encode xong
│   ├── *.npy                    # Vector embeddings
│   └── *.json                   # Metadata chứa text & chunk index
│
├── train/
   ├── encode.py                # File encode dữ liệu (E5)
   ├── decode.py                # File decode + sinh câu trả lời (RAG + FAISS + Llama)
   └── giaoDien.py              # Giao diện Gradio hiển thị chatbot
```

---

## 3️⃣ Công nghệ sử dụng

| Thành phần           | Công nghệ / Mô hình                             | Mô tả                                  |
| -------------------- | ----------------------------------------------- | -------------------------------------- |
| **Embedding Model**    🧩 `intfloat/multilingual-e5-base`              | Encode + decode câu hỏi & dữ liệu      |
|                                                                         | văn bản                               |
| **Generative Model** | 🦙 `meta-llama/Llama-3.2-3B-Instruct`           | Sinh câu trả lời tiếng Việt            |
| **Search Engine**    | 🔍 `FAISS`                                      | Tìm kiếm vector tương đồng nhanh chóng |
| **Pipeline**         | 🧠 `RAG`                                        | Kết hợp tìm kiếm + sinh câu trả lời    |
| **Giao diện**        | 💬 `Gradio`                                     | Giao diện web chat thân thiện          |
| **Framework**        | ⚙️ `Transformers`, `PyTorch`, `HuggingFace Hub` | Triển khai model và xử lý dữ liệu      |

---

## 4️⃣ Luồng hoạt động

1. **Tiền xử lý dữ liệu**

   * Đọc file `.txt` trong thư mục `data/`.
   * Tách truyện thành các đoạn nhỏ (chunk).
   * Encode bằng **E5** → sinh ra `.npy` và `.json`.
   * Có thể lặp lại nhiều truyện khác, trong code chỉ ví dụ về truyện bạch tuyết 

2. **Truy vấn người dùng**

   * Người dùng nhập câu hỏi vào giao diện.

3. **Truy xuất thông tin**

   * Câu hỏi được **encode** bằng E5 → tìm kiếm các đoạn gần nhất bằng **FAISS**.

4. **Sinh câu trả lời**

   * Các đoạn liên quan được gửi vào **Llama 3.2** để sinh **câu trả lời tiếng Việt tự nhiên**.

5. **Hiển thị kết quả**

   * Giao diện Gradio hiển thị câu trả lời cùng các đoạn văn được truy xuất.

---

## 5️⃣ Hướng dẫn cài đặt & chạy

### 🔧 Yêu cầu

* Python 3.10+
* GPU (khuyến nghị)
* Môi trường ảo (venv)

### 📦 Cài đặt

```bash
pip install torch transformers faiss-cpu gradio numpy tenacity
pip install huggingface_hub
```

### 🚀 Chạy pipeline encode

```bash
python train/encode.py
```

### 🚀 Chạy chatbot (RAG)

```bash
python train/decode.py
```

### 💬 Mở giao diện web

```bash
python train/giaoDien.py
```

Sau khi chạy, Gradio sẽ hiển thị link:

```
Running on local URL: http://127.0.0.1:7860
```

➡️ Mở link này để chat với chatbot.

---

## 6️⃣ Giao diện minh họa

| 💬 Màn hình giao diện               | ⚙️ Quy trình pipeline             |
| ----------------------------------- | --------------------------------- |
| ![Giao diện chat](image/image2.jpg) | ![Pipeline RAG](image/image3.jpg) |

---

## 7️⃣ Định hướng phát triển

✅ **Hiện tại:**

* Hỗ trợ tiếng Việt, dùng FAISS + Llama 3.2.
* Tìm kiếm và trả lời dựa vào nội dung trong file `.txt`.

🚀 **Tương lai:**

* Thêm hỗ trợ **tóm tắt nội dung truyện**.
* Mở rộng cho **toàn bộ kho truyện cổ tích Việt Nam**.
* Cho phép **train thêm embedding** tùy chọn.
* Tích hợp **ghi log hội thoại**, **bộ nhớ ngữ cảnh**, và **tối ưu RAG**.

