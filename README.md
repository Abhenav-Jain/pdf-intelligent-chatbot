# 🤖 PDF Intelligent Chatbot

A smart PDF chatbot that answers questions **strictly from uploaded documents** using LangChain, Mistral AI, and Hybrid Search.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 Hybrid Search | BM25 (keyword) + FAISS (semantic) combined |
| 🧠 Query Rewriting | Vague questions auto-improved before search |
| 💬 Conversation Memory | Remembers last 5 exchanges |
| 📄 Source Page Tracking | Shows which pages the answer came from |
| 🚫 Hallucination-Free | Answers only from PDF — never from AI training data |

---

## 🛠️ Tech Stack

- **LLM** — Mistral AI (`mistral-small-2506`)
- **Framework** — LangChain
- **PDF Parsing** — PyMuPDF (fitz)
- **Vector Store** — FAISS
- **Keyword Search** — BM25
- **UI** — Streamlit

---

## 📁 Project Structure

```
PDF Intelligent Chatbot/
│
├── logic_file.py      # Core logic — PDF processing, retrieval, LLM chain
├── ui.py              # Streamlit UI — upload, chat display, input
├── Requirements.txt   # Dependencies
├── .env               # API keys (not pushed)
└── .gitignore
```

---

## ⚙️ Setup & Run

**1. Clone the repo**
```bash
git clone https://github.com/Abhenav-Jain/pdf-intelligent-chatbot.git
cd pdf-intelligent-chatbot
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r Requirements.txt
```

**4. Create `.env` file**
```
MISTRAL_API_KEY=your_mistral_api_key_here
```

**5. Run the app**
```bash
streamlit run ui.py
```

---

## 🚀 How It Works

```
PDF Upload
    ↓
Text extracted page-by-page (PyMuPDF)
    ↓
Split into chunks (1000 chars, 200 overlap)
    ↓
BM25 + FAISS Hybrid Retriever built
    ↓
User asks a question
    ↓
Query rewritten for better search
    ↓
Top 5 relevant chunks retrieved
    ↓
Mistral LLM answers from chunks only
    ↓
Answer + Source Pages displayed
```

---

## 📸 Screenshots

> Upload a PDF → Ask anything → Get answers with source pages

---

## 🔑 Get Mistral API Key

1. Go to [https://console.mistral.ai](https://console.mistral.ai)
2. Sign up / Login
3. Create API Key
4. Paste in `.env` file

---

## ⚠️ Note

- `.env` file is **not pushed** to GitHub (API key stays safe)
- Works best with text-based PDFs (not scanned/image PDFs)
- Large PDFs (>200k chars) are automatically trimmed
