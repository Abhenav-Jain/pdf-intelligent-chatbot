"""
Task 1: PDF Intelligence Chatbot — Core Logic

  #1 → Semantic Chunking + FAISS Vector Retrieval
  #2 → Conversation Memory (last 5 exchanges)
  #3 → Query Rewriting (vague → specific)
  #4 → Source Page Tracking
  #5 → Hybrid Search (BM25 keyword + Semantic)

Steps:
  Step 2 → Extract text from PDF
  Step 3 → Chunk + build hybrid retriever
  Step 5 → Build prompt (system + instructions + context + history + question)
  Step 6 → Build LLM chain
  Step 7 → Rewrite query → retrieve chunks → answer with sources
"""

import sys
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# Manual EnsembleRetriever — no extra dependency, works on all versions
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field

class EnsembleRetriever(BaseRetriever):
    """Combines multiple retrievers with weighted reciprocal rank scoring."""
    retrievers: list = Field(default_factory=list)
    weights: list = Field(default_factory=list)

    def _get_relevant_documents(self, query: str) -> list:
        all_docs = {}
        for retriever, weight in zip(self.retrievers, self.weights):
            docs = retriever.invoke(query)
            for rank, doc in enumerate(docs):
                key = doc.page_content[:100]
                if key not in all_docs:
                    all_docs[key] = {"doc": doc, "score": 0.0}
                all_docs[key]["score"] += weight * (1.0 / (rank + 1))
        sorted_docs = sorted(all_docs.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs]


# ── Environment Setup ─────────────────────────────────────────────────────────
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in .env file!")

MODEL         = "mistral-small-2506"
CHAR_LIMIT    = 200_000   
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
TOP_K_CHUNKS  = 5         


# ── STEP 2: Extract text from PDF 
def extract_text_from_pdf(pdf_path: str) -> list:
    """
    Extracts text page-by-page from a PDF file path.
    Returns list of {page, text} dicts for source tracking.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": page_num, "text": text})
    doc.close()
    if not pages:
        raise ValueError("No readable text found. It may be a scanned/image PDF.")
    return pages


def extract_text_from_bytes(pdf_bytes: bytes) -> list:
    """
    Extracts text page-by-page from raw PDF bytes.
    Returns list of {page, text} dicts for source tracking.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": page_num, "text": text})
    doc.close()
    if not pages:
        raise ValueError("No readable text found. It may be a scanned/image PDF.")
    return pages


# ── STEP 3: Chunk + Build Hybrid Retriever 

def _pages_to_chunks(pages: list) -> tuple:
    """
    Splits page texts into overlapping chunks.
    Returns (chunk_texts, chunk_metadatas) — metadata carries page number.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunk_texts = []
    chunk_metas = []

    for page_data in pages:
        splits = splitter.split_text(page_data["text"])
        for split in splits:
            chunk_texts.append(split)
            chunk_metas.append({"page": page_data["page"]})

    return chunk_texts, chunk_metas


def build_hybrid_retriever(pages: list):
    """
    #1 + #5: Builds an EnsembleRetriever combining:
      - BM25 (keyword-based, exact term matching)
      - FAISS (semantic / meaning-based)
    50/50 weighted so both contribute equally.
    """
    chunk_texts, chunk_metas = _pages_to_chunks(pages)

    # Semantic retriever (FAISS)
    embeddings = MistralAIEmbeddings(api_key=MISTRAL_API_KEY)
    vectorstore = FAISS.from_texts(chunk_texts, embeddings, metadatas=chunk_metas)
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_CHUNKS})

    # Keyword retriever (BM25)
    bm25_retriever = BM25Retriever.from_texts(chunk_texts, metadatas=chunk_metas)
    bm25_retriever.k = TOP_K_CHUNKS

    #Hybrid ensemble retriever
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.5, 0.5],
    )

    return hybrid_retriever


def load_pdf_from_bytes(pdf_bytes: bytes, filename: str) -> dict:
    """
    Full pipeline for UI upload:
      1. Extract text with page numbers
      2. Build hybrid retriever
    Returns dict with retriever, filename, page count, char count.
    """
    pages = extract_text_from_bytes(pdf_bytes)

    total_chars = sum(len(p["text"]) for p in pages)
    if total_chars > CHAR_LIMIT:
        kept, total = [], 0
        for p in pages:
            if total + len(p["text"]) > CHAR_LIMIT:
                break
            kept.append(p)
            total += len(p["text"])
        pages = kept
        print(f"Large PDF — kept {len(pages)} pages ({total:,} chars).")

    retriever = build_hybrid_retriever(pages)

    return {
        "retriever": retriever,
        "filename":  filename,
        "pages":     len(pages),
        "chars":     sum(len(p["text"]) for p in pages),
    }


def load_pdf(pdf_path: str) -> dict:
    """CLI version of load_pdf_from_bytes."""
    print(f"Loading: {pdf_path}")
    pages = extract_text_from_pdf(pdf_path)
    retriever = build_hybrid_retriever(pages)
    filename = pdf_path.split("/")[-1]
    chars = sum(len(p["text"]) for p in pages)
    print(f"Loaded '{filename}' — {chars:,} chars, {len(pages)} pages\n")
    return {"retriever": retriever, "filename": filename, "pages": len(pages), "chars": chars}


# ── STEP 3 (CLI): In-memory store 
pdf_store = {
    "retriever": None,
    "filename":  "",
    "pages":     0,
    "chars":     0,
}


# ── #3: Query Rewriting 
def rewrite_query(question: str) -> str:
    """
    #3: Rewrites vague/short user questions into more specific,
    search-friendly queries before retrieval.
    e.g. "what is RNN" -> "What is a Recurrent Neural Network, how does it work?"
    """
    llm = ChatMistralAI(api_key=MISTRAL_API_KEY, model=MODEL, temperature=0)

    rewrite_prompt = (
        "Rewrite the following question to be more specific and search-friendly "
        "for retrieving content from a technical document. "
        "Return ONLY the rewritten question, nothing else.\n\n"
        f"Original: {question}\n"
        "Rewritten:"
    )

    result = llm.invoke(rewrite_prompt)
    rewritten = result.content.strip()
    return rewritten if rewritten else question


# ── #4: Retrieve chunks with source pages 
def retrieve_with_sources(retriever, question: str) -> tuple:
    """
    #4: Retrieves relevant chunks and extracts source page numbers.
    Returns (combined_context_text, sorted_unique_page_numbers).
    """
    docs = retriever.invoke(question)

    context_parts = []
    pages_seen = set()

    for doc in docs:
        context_parts.append(doc.page_content)
        page = doc.metadata.get("page")
        if page:
            pages_seen.add(page)

    context = "\n\n---\n\n".join(context_parts)
    source_pages = sorted(pages_seen)
    return context, source_pages


# ── STEP 5: Build prompt with system + instructions + history + context + question 
def build_langchain_prompt() -> ChatPromptTemplate:
    """
    Step 5: Same system + instruction prompts as before.
    Added {chat_history} placeholder for memory (#2).
    """

    # Component 1: System Prompt (unchanged)
    system_prompt = (
        "You are a precise document assistant. "
        "Your ONLY source of truth is the PDF content provided by the user. "
        "Never use external knowledge or make assumptions beyond "
        "what is explicitly stated in the document."
    )

    # Component 2 + 3 + 4 — instructions, history, context, question
    human_prompt = """INSTRUCTIONS:
- Answer strictly and only from the PDF content provided below.
- If the answer is not found in the document, respond EXACTLY with:
  "The information is not available in the document."
- Do not use external knowledge, prior training data, or assumptions.
- Be concise and accurate. Quote or paraphrase directly from the document.

{chat_history}

PDF CONTENT:
---
{pdf_content}
---

QUESTION: {user_question}"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template(human_prompt),
    ])

    return prompt


# ── STEP 6: Build chain ───────────────────────────────────────────────────────
def build_chain():
    """Step 6: Prompt -> Mistral -> StrOutputParser."""
    llm = ChatMistralAI(
        api_key=MISTRAL_API_KEY,
        model=MODEL,
        temperature=0,
    )
    prompt = build_langchain_prompt()
    parser = StrOutputParser()
    return prompt | llm | parser


# ── #2: Memory helpers 
def format_chat_history(chat_history: list, last_k: int = 5) -> str:
    """
    #2: Formats last K chat exchanges into a string for the prompt.
    """
    recent = chat_history[-(last_k * 2):]

    if not recent:
        return ""

    lines = ["CONVERSATION HISTORY:"]
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")

    return "\n".join(lines) + "\n"


# ── STEP 7: Ask question (full smart pipeline) ────────────────────────────────
def ask_question(
    chain,
    retriever,
    user_question: str,
    chat_history: list = None,
) -> tuple:
    """
    Step 7: Full smart pipeline:
      #3 -> Rewrite query
      #1+#5 -> Retrieve relevant chunks via hybrid search
      #4 -> Extract source page numbers
      #2 -> Format conversation history
      -> Invoke chain -> return (answer, source_pages)

    Returns:
        (answer_string, [source_page_numbers])
    """
    if retriever is None:
        return "Please load a PDF first.", []

    # #3: Rewrite query for better retrieval
    search_query = rewrite_query(user_question)

    # #1 + #5 + #4: Hybrid retrieval with source pages
    pdf_content, source_pages = retrieve_with_sources(retriever, search_query)

    # #2: Format conversation history
    history_text = format_chat_history(chat_history or [])

    # Invoke chain
    answer = chain.invoke({
        "pdf_content":   pdf_content,
        "user_question": user_question,
        "chat_history":  history_text,
    })

    return answer, source_pages


# ── CLI Entry Point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "document.pdf"

    result = load_pdf(pdf_path)
    pdf_store.update(result)

    chain = build_chain()
    cli_history = []

    print("=" * 55)
    print(" PDF Chatbot ready! (Smart Edition)")
    print("   Type 'exit' to quit.")
    print("=" * 55)

    while True:
        user_input = input("\n Your question: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            print(" Bye!")
            break

        cli_history.append({"role": "user", "content": user_input})

        answer, pages = ask_question(
            chain,
            pdf_store["retriever"],
            user_input,
            cli_history,
        )

        print(f"\n Answer:\n{answer}")
        if pages:
            print(f" Sources: Page(s) {', '.join(map(str, pages))}")

        cli_history.append({"role": "assistant", "content": answer})
