"""
Task 1: PDF Intelligence Chatbot — Streamlit UI (Enhanced)
==========================================================
UI handles:
  Step 1 → PDF Upload
  Step 4 → User question input
  #4     → Source page display

All smart logic is in logic_file.py
"""

import streamlit as st
from logic_file import load_pdf_from_bytes, build_chain, ask_question

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  [data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
  }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

  .main { background: #0a0c12; }
  .block-container { padding-top: 2rem; max-width: 860px; }

  .chat-user {
    background: #1a1f2e;
    border: 1px solid #2d3550;
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #e2e8f0;
    font-size: 0.95rem;
    text-align: right;
  }
  .chat-bot {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #3b82f6;
    border-radius: 4px 12px 12px 12px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #cbd5e1;
    font-size: 0.95rem;
    line-height: 1.6;
  }
  .chat-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 4px;
  }
  .label-user  { color: #60a5fa; text-align: right; }
  .label-bot   { color: #34d399; }
  .label-source{ color: #a78bfa; font-size: 0.68rem; margin-top: 4px; }

  [data-testid="stFileUploader"] {
    border: 2px dashed #2d3550 !important;
    border-radius: 10px;
    background: #0f1117;
  }
  [data-testid="stTextInput"] input {
    background: #111827 !important;
    border: 1px solid #2d3550 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
  }
  .stButton button {
    background: #3b82f6 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600;
    height: 42px;
  }
  .stButton button:hover { background: #2563eb !important; }

  .pdf-card {
    background: #0d1f3c;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 12px;
    font-size: 0.85rem;
    color: #93c5fd;
  }
  .pill-ready {
    display: inline-block; background: #064e3b; color: #34d399;
    border-radius: 20px; padding: 2px 10px; font-size: 0.75rem; font-weight: 600;
  }
  .pill-wait {
    display: inline-block; background: #1e1a08; color: #fbbf24;
    border-radius: 20px; padding: 2px 10px; font-size: 0.75rem; font-weight: 600;
  }
  .source-tag {
    display: inline-block;
    background: #1e1b4b;
    color: #a78bfa;
    border: 1px solid #4c1d95;
    border-radius: 12px;
    padding: 1px 8px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 4px;
  }

  h1 { color: #f1f5f9 !important; font-weight: 600 !important; }
  h3 { color: #94a3b8 !important; font-weight: 500 !important; }
  .chat-container { max-height: 560px; overflow-y: auto; padding-right: 4px; }
  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ────────────────────────────────────────────────────────
for key, default in [
    ("chat_history", []),    # list of {role, content, pages?}
    ("retriever", None),     # hybrid retriever (replaces raw pdf_text)
    ("pdf_name", ""),
    ("pdf_pages", 0),
    ("pdf_chars", 0),
    ("chain", None),
    ("pdf_processed", ""),   # guard against re-processing on rerun
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── SIDEBAR — Step 1: PDF Upload ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 PDF Upload")
    st.markdown("---")
    st.success("✅ API Key loaded from .env")
    st.markdown("### Upload your PDF")

    uploaded = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

    if uploaded and uploaded.name != st.session_state.pdf_processed:
        with st.spinner("📖 PDF padh raha hoon + embeddings bana raha hoon..."):
            try:
                result = load_pdf_from_bytes(uploaded.read(), uploaded.name)

                st.session_state.retriever     = result["retriever"]
                st.session_state.pdf_name      = result["filename"]
                st.session_state.pdf_pages     = result["pages"]
                st.session_state.pdf_chars     = result["chars"]
                st.session_state.chain         = build_chain()
                st.session_state.chat_history  = []
                st.session_state.pdf_processed = uploaded.name

            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.pdf_name:
        st.markdown(f"""
        <div class="pdf-card">
            📄 <b>{st.session_state.pdf_name}</b><br>
            {st.session_state.pdf_pages} pages &nbsp;|&nbsp; {st.session_state.pdf_chars:,} chars<br>
            <span class="pill-ready">● Ready</span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Clear & Upload New PDF"):
            for k in ["retriever", "pdf_name", "pdf_pages", "pdf_chars",
                      "chain", "chat_history", "pdf_processed"]:
                st.session_state[k] = None if k in ("retriever", "chain") else (
                    [] if k == "chat_history" else (0 if k in ("pdf_pages", "pdf_chars") else "")
                )
            st.rerun()
    else:
        st.markdown('<span class="pill-wait">⏳ Waiting for PDF</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Model:** `mistral-small-2506`")
    st.markdown("**Search:** Hybrid (BM25 + FAISS)")
    st.markdown("**Memory:** Last 5 exchanges")
    st.markdown("**Framework:** LangChain + PyMuPDF")


# ── MAIN AREA ─────────────────────────────────────────────────────────────────
st.markdown("# 🤖 PDF Intelligence Chatbot")
st.markdown("### PDF upload karo → Questions pucho → Answers pao")
st.markdown("---")

# Chat history display
if st.session_state.chat_history:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown('<div class="chat-label label-user">You</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="chat-label label-bot">Assistant</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bot">{msg["content"]}</div>', unsafe_allow_html=True)
            # #4: Show source pages if available
            pages = msg.get("pages", [])
            if pages:
                tags = "".join(f'<span class="source-tag">📄 Page {p}</span>' for p in pages)
                st.markdown(
                    f'<div class="label-source">Sources: {tags}</div>',
                    unsafe_allow_html=True
                )
    st.markdown('</div>', unsafe_allow_html=True)

else:
    if st.session_state.pdf_name:
        st.info("✅ PDF loaded! Ab apna pehla question pucho.")
    else:
        st.info("👈 Sidebar mein PDF upload karo shuru karne ke liye.")


# ── Step 4: Question Input ────────────────────────────────────────────────────
if st.session_state.chain:
    st.markdown("---")
    col1, col2 = st.columns([5, 1])

    with col1:
        user_q = st.text_input(
            "",
            placeholder="PDF ke baare mein kuch bhi pucho...",
            label_visibility="collapsed",
            key="question_input",
        )
    with col2:
        send = st.button("Send ➤")

    if send and user_q.strip():
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_q})

        with st.spinner("🤔 Soch raha hoon..."):
            try:
                # Full smart pipeline — pass entire history for memory (#2)
                answer, source_pages = ask_question(
                    st.session_state.chain,
                    st.session_state.retriever,
                    user_q,
                    st.session_state.chat_history,
                )
            except Exception as e:
                answer = f"❌ Error: {e}"
                source_pages = []

        # Store answer + source pages together
        st.session_state.chat_history.append({
            "role":    "assistant",
            "content": answer,
            "pages":   source_pages,   # #4: stored for display
        })
        st.rerun()
