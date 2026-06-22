"""
PDF Intelligence Assistant — Streamlit UI
==========================================
UI responsibilities:
  - Document upload and ingestion status
  - Document type selection (passed to logic_file.ask_question_stream for
    domain-aware query rewriting)
  - Chat-style question input
  - Markdown-rendered assistant responses (table / comparison / paragraph
    aware — logic_file.py detects the requested format and the model
    structures its answer accordingly)
  - LIVE streaming of partial answers as they're generated, instead of
    waiting for the full response (source pages appear as soon as
    retrieval finishes, answer text fills in incrementally)
  - Source page citation display
  - Truncation warning display (large PDFs)

All retrieval / LLM logic lives in logic_file.py

No third-party markdown package required — render_markdown() below is a
small, self-contained renderer built only on the standard library. It
covers exactly the formatting logic_file.py's system prompt instructs the
model to produce: headings, bold/italic, bullet/numbered lists, fenced
code blocks, tables, blockquotes, horizontal rules, and links.
"""

import html as html_lib
import re

import streamlit as st

from logic_file import load_pdf_from_bytes, build_chain, ask_question_stream

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Intelligence Assistant",
    page_icon=None,
    layout="wide",
)

DOCUMENT_TYPES = ["technical", "legal", "medical", "financial", "general"]


# ── Markdown Rendering (dependency-free) ─────────────────────────────────────
def _inline_md(text: str) -> str:
    """Apply inline markdown formatting: bold, italic, inline code, links."""
    text = html_lib.escape(text, quote=False)

    # Protect inline code spans from further inline transforms
    code_spans = []

    def _stash_code(m):
        code_spans.append(m.group(1))
        return f"\x00CODE{len(code_spans) - 1}\x00"

    text = re.sub(r"`([^`]+?)`", _stash_code, text)

    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"__(.+?)__", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^\*\n]+?)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"(?<!_)_([^_\n]+?)_(?!_)", r"<em>\1</em>", text)
    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^\)\s]+)\)",
        r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>',
        text,
    )

    for i, code in enumerate(code_spans):
        text = text.replace(f"\x00CODE{i}\x00", f"<code>{code}</code>")
    return text


def render_markdown(text: str) -> str:
    """
    Convert assistant markdown output into HTML for display inside the
    styled chat bubble. Supports: # / ## ... headings, **bold**, *italic*,
    `inline code`, fenced ```code blocks```, - / * bullet lists, 1. numbered
    lists, > blockquotes, --- horizontal rules, | table | rows, and links.

    Called repeatedly with the growing accumulated text while a response is
    streaming in, and once more with the final text — so it must tolerate
    being run on a string that is still mid-stream (e.g. an unclosed code
    fence or partial table row). Worst case is a brief visual flicker while
    that block finishes arriving; nothing breaks or leaks unescaped HTML.
    """
    lines = text.replace("\r\n", "\n").split("\n")
    html_out = []
    i, n = 0, len(lines)
    paragraph_buffer = []

    def flush_paragraph():
        if paragraph_buffer:
            joined = " ".join(paragraph_buffer).strip()
            if joined:
                html_out.append(f"<p>{_inline_md(joined)}</p>")
            paragraph_buffer.clear()

    while i < n:
        stripped = lines[i].strip()

        # Fenced code block
        if stripped.startswith("```"):
            flush_paragraph()
            lang = stripped[3:].strip()
            code_lines = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing fence (or end of text if still streaming)
            code_text = html_lib.escape("\n".join(code_lines))
            lang_class = f' class="language-{lang}"' if lang else ""
            html_out.append(f"<pre><code{lang_class}>{code_text}</code></pre>")
            continue

        # Horizontal rule
        if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", stripped):
            flush_paragraph()
            html_out.append("<hr>")
            i += 1
            continue

        # Heading
        heading_match = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            content = _inline_md(heading_match.group(2))
            html_out.append(f"<h{level}>{content}</h{level}>")
            i += 1
            continue

        # Blockquote
        if stripped.startswith(">"):
            flush_paragraph()
            quote_lines = []
            while i < n and lines[i].strip().startswith(">"):
                quote_lines.append(lines[i].strip().lstrip(">").strip())
                i += 1
            html_out.append(f"<blockquote><p>{_inline_md(' '.join(quote_lines))}</p></blockquote>")
            continue

        # Table (header row + |---|---| separator row)
        if (
            "|" in stripped
            and i + 1 < n
            and "-" in lines[i + 1]
            and re.match(r"^\|?[\s:|-]+\|?$", lines[i + 1].strip())
        ):
            flush_paragraph()
            header_cells = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2
            rows = []
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            thead = "".join(f"<th>{_inline_md(c)}</th>" for c in header_cells)
            tbody = "".join(
                "<tr>" + "".join(f"<td>{_inline_md(c)}</td>" for c in row) + "</tr>" for row in rows
            )
            html_out.append(f"<table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>")
            continue

        # Unordered list
        if re.match(r"^[-*+]\s+(.*)", stripped):
            flush_paragraph()
            items = []
            while i < n:
                m = re.match(r"^[-*+]\s+(.*)", lines[i].strip())
                if not m:
                    break
                items.append(m.group(1))
                i += 1
            html_out.append("<ul>" + "".join(f"<li>{_inline_md(it)}</li>" for it in items) + "</ul>")
            continue

        # Ordered list
        if re.match(r"^\d+\.\s+(.*)", stripped):
            flush_paragraph()
            items = []
            while i < n:
                m = re.match(r"^\d+\.\s+(.*)", lines[i].strip())
                if not m:
                    break
                items.append(m.group(1))
                i += 1
            html_out.append("<ol>" + "".join(f"<li>{_inline_md(it)}</li>" for it in items) + "</ol>")
            continue

        # Blank line → paragraph break
        if stripped == "":
            flush_paragraph()
            i += 1
            continue

        # Default: accumulate into paragraph
        paragraph_buffer.append(stripped)
        i += 1

    flush_paragraph()
    return "\n".join(html_out)


def render_user_text(text: str) -> str:
    """Escape user input before inserting into raw HTML to prevent injection."""
    return html_lib.escape(text).replace("\n", "<br>")


def bot_bubble_html(text: str) -> str:
    return f'<div class="chat-bot">{render_markdown(text)}</div>'


def sources_html(pages: list) -> str:
    if not pages:
        return ""
    tags = "".join(f'<span class="source-tag">Page {p}</span>' for p in pages)
    return f'<div class="label-source">Sources: {tags}</div>'


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  [data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
  }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

  .main { background: #0a0c12; }
  .block-container { padding-top: 2rem; max-width: 900px; }

  .section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 6px;
  }

  .status-badge {
    display: inline-flex;
    align-items: center;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-weight: 600;
  }
  .status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: currentColor;
    display: inline-block;
    margin-right: 6px;
  }
  .status-ready  { background: #0a2e22; color: #34d399; }
  .status-wait   { background: #1e1a08; color: #fbbf24; }
  .status-config { background: #0a2e22; color: #34d399; }
  .status-live   { background: #1e1b4b; color: #a78bfa; }
  .status-live .status-dot { animation: pulse-dot 1.1s ease-in-out infinite; }
  @keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

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
    padding: 14px 18px;
    margin: 8px 0;
    color: #cbd5e1;
    font-size: 0.95rem;
    line-height: 1.65;
  }

  .chat-bot p { margin: 0 0 10px 0; }
  .chat-bot p:last-child { margin-bottom: 0; }
  .chat-bot strong { color: #f1f5f9; }
  .chat-bot a { color: #60a5fa; }
  .chat-bot ul, .chat-bot ol { margin: 6px 0 10px 22px; padding: 0; }
  .chat-bot li { margin-bottom: 4px; }
  .chat-bot h1, .chat-bot h2, .chat-bot h3, .chat-bot h4, .chat-bot h5, .chat-bot h6 {
    color: #f1f5f9; margin: 14px 0 8px 0; font-weight: 600;
  }
  .chat-bot h1 { font-size: 1.3rem; }
  .chat-bot h2 { font-size: 1.18rem; }
  .chat-bot h3 { font-size: 1.08rem; }
  .chat-bot h4, .chat-bot h5, .chat-bot h6 { font-size: 1rem; }
  .chat-bot blockquote {
    border-left: 3px solid #334155;
    margin: 8px 0;
    padding: 4px 12px;
    color: #94a3b8;
  }
  .chat-bot blockquote p { margin: 0; }
  .chat-bot code {
    font-family: 'JetBrains Mono', monospace;
    background: #1e293b;
    border: 1px solid #2d3550;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.85rem;
    color: #93c5fd;
  }
  .chat-bot pre {
    background: #0a0e17;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 12px 14px;
    overflow-x: auto;
    margin: 10px 0;
  }
  .chat-bot pre code {
    background: transparent;
    border: none;
    padding: 0;
    color: #d1d9e6;
  }
  .chat-bot table {
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
    font-size: 0.88rem;
  }
  .chat-bot th, .chat-bot td {
    border: 1px solid #2d3550;
    padding: 6px 10px;
    text-align: left;
  }
  .chat-bot th { background: #1a1f2e; color: #f1f5f9; }
  .chat-bot hr { border-color: #1e293b; margin: 10px 0; }

  .chat-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 4px;
  }
  .label-user  { color: #60a5fa; text-align: right; }
  .label-bot   { color: #34d399; }
  .label-source{ color: #a78bfa; font-size: 0.7rem; margin-top: 6px; }

  [data-testid="stFileUploader"] {
    border: 2px dashed #2d3550 !important;
    border-radius: 10px;
    background: #0f1117;
  }
  [data-testid="stChatInput"] textarea {
    background: #111827 !important;
    border: 1px solid #2d3550 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
  }
  .stButton button {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    font-weight: 600;
    height: 42px;
  }
  .stButton button:hover { background: #2d3a52 !important; border-color: #3b82f6 !important; }

  .doc-card {
    background: #0d1f3c;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    padding: 12px 14px;
    margin-top: 12px;
    font-size: 0.85rem;
    color: #93c5fd;
  }
  .doc-card .doc-name { color: #f1f5f9; font-weight: 600; }
  .doc-card .doc-meta { color: #93c5fd; opacity: 0.85; }

  .source-tag {
    display: inline-block;
    background: #1e1b4b;
    color: #a78bfa;
    border: 1px solid #4c1d95;
    border-radius: 12px;
    padding: 1px 9px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 4px;
  }

  .app-title { color: #f1f5f9 !important; font-weight: 700 !important; letter-spacing: -0.01em; }
  .app-subtitle { color: #94a3b8 !important; font-weight: 400 !important; font-size: 0.95rem; }
  .chat-container { max-height: 560px; overflow-y: auto; padding-right: 4px; }
  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ────────────────────────────────────────────────────────
for key, default in [
    ("chat_history", []),        # list of {role, content, pages?}
    ("retriever", None),         # hybrid retriever
    ("pdf_name", ""),
    ("pdf_pages", 0),
    ("pdf_chars", 0),
    ("chain", None),
    ("pdf_processed", ""),       # guard against re-processing on rerun
    ("truncation_warning", None),
    ("document_type", "technical"),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── SIDEBAR — Document Upload ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">Document</div>', unsafe_allow_html=True)
    st.markdown(
        '<span class="status-badge status-config"><span class="status-dot"></span>'
        'API key configured</span>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown('<div class="section-label">Upload PDF</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

    if uploaded and uploaded.name != st.session_state.pdf_processed:
        with st.spinner("Processing document and generating embeddings..."):
            try:
                result = load_pdf_from_bytes(uploaded.read(), uploaded.name)

                st.session_state.retriever          = result["retriever"]
                st.session_state.pdf_name           = result["filename"]
                st.session_state.pdf_pages          = result["pages"]
                st.session_state.pdf_chars          = result["chars"]
                st.session_state.truncation_warning = result.get("truncation_warning")
                st.session_state.chain              = build_chain()
                st.session_state.chat_history       = []
                st.session_state.pdf_processed      = uploaded.name

            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.pdf_name:
        st.markdown(f"""
        <div class="doc-card">
            <span class="doc-name">{st.session_state.pdf_name}</span><br>
            <span class="doc-meta">{st.session_state.pdf_pages} pages &nbsp;|&nbsp; {st.session_state.pdf_chars:,} characters</span><br>
            <span class="status-badge status-ready"><span class="status-dot"></span>Ready</span>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.truncation_warning:
            st.warning(st.session_state.truncation_warning)

        st.write("")
        st.markdown('<div class="section-label">Document Type</div>', unsafe_allow_html=True)
        st.session_state.document_type = st.selectbox(
            "",
            DOCUMENT_TYPES,
            index=DOCUMENT_TYPES.index(st.session_state.document_type),
            label_visibility="collapsed",
            help="Used for domain-aware query rewriting before retrieval.",
        )

        if st.button("Reset Session"):
            for k in ["retriever", "pdf_name", "pdf_pages", "pdf_chars",
                      "chain", "chat_history", "pdf_processed", "truncation_warning"]:
                st.session_state[k] = None if k in ("retriever", "chain", "truncation_warning") else (
                    [] if k == "chat_history" else (0 if k in ("pdf_pages", "pdf_chars") else "")
                )
            st.rerun()
    else:
        st.markdown(
            '<span class="status-badge status-wait"><span class="status-dot"></span>'
            'Awaiting document</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<div class="section-label">System</div>', unsafe_allow_html=True)
    st.markdown("**Model:** `mistral-small-2506`")
    st.markdown("**Search:** Hybrid (BM25 + FAISS)")
    st.markdown("**Memory:** Last 3 exchanges")
    st.markdown("**Framework:** LangChain + PyMuPDF")
    st.markdown("**Response:** Streamed live")


# ── MAIN AREA ─────────────────────────────────────────────────────────────────
st.markdown('<h1 class="app-title">PDF Intelligence Assistant</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">Upload a document, ask questions, and get answers grounded in its content. '
    'Ask for a table, a comparison, or a paragraph and the answer is structured accordingly.</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# Chat history display (completed exchanges only — the in-progress exchange,
# if any, is rendered live further down before being committed here)
if st.session_state.chat_history:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown('<div class="chat-label label-user">You</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="chat-user">{render_user_text(msg["content"])}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="chat-label label-bot">Assistant</div>', unsafe_allow_html=True)
            st.markdown(bot_bubble_html(msg["content"]), unsafe_allow_html=True)
            sh = sources_html(msg.get("pages", []))
            if sh:
                st.markdown(sh, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    if st.session_state.pdf_name:
        st.info("Document loaded successfully. Ask your first question to begin.")
    else:
        st.info("Upload a PDF from the sidebar to get started.")


# ── Question Input ────────────────────────────────────────────────────────
user_q = st.chat_input(
    "Ask anything about the document...",
    disabled=not bool(st.session_state.chain),
)

if user_q and user_q.strip():
    st.session_state.chat_history.append({"role": "user", "content": user_q})

    # Render the new exchange live, in place, as it streams in — this is
    # the "partial results" turn: it is not yet in chat_history's rendered
    # loop above, so we draw it manually here and commit it once finished.
    st.markdown('<div class="chat-label label-user">You</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="chat-user">{render_user_text(user_q)}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="chat-label label-bot">Assistant</div>', unsafe_allow_html=True)
    status_placeholder = st.empty()
    answer_placeholder = st.empty()
    sources_placeholder = st.empty()

    status_placeholder.markdown(
        '<span class="status-badge status-live"><span class="status-dot"></span>'
        'Retrieving relevant content...</span>',
        unsafe_allow_html=True,
    )

    accumulated = ""
    source_pages = []
    try:
        source_pages, stream_gen = ask_question_stream(
            st.session_state.chain,
            st.session_state.retriever,
            user_q,
            st.session_state.chat_history,
            document_type=st.session_state.document_type,
        )

        # Source pages are known as soon as retrieval finishes — show them
        # immediately rather than waiting for the full answer to generate.
        status_placeholder.markdown(
            '<span class="status-badge status-live"><span class="status-dot"></span>'
            'Generating answer...</span>',
            unsafe_allow_html=True,
        )
        sh = sources_html(source_pages)
        if sh:
            sources_placeholder.markdown(sh, unsafe_allow_html=True)

        for chunk in stream_gen:
            accumulated += chunk
            answer_placeholder.markdown(bot_bubble_html(accumulated), unsafe_allow_html=True)

        status_placeholder.empty()

    except Exception as e:
        accumulated = f"Error: {e}"
        answer_placeholder.markdown(bot_bubble_html(accumulated), unsafe_allow_html=True)
        status_placeholder.empty()

    st.session_state.chat_history.append({
        "role":    "assistant",
        "content": accumulated,
        "pages":   source_pages,
    })
    st.rerun()