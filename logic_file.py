"""
Task 1: PDF Intelligence Chatbot — Core Logic

  #1 → Semantic Chunking + FAISS Vector Retrieval
  #2 → Conversation Memory (last 3 exchanges, dynamic trim)
  #3 → Query Rewriting (vague → specific, domain-aware)
  #4 → Source Page Tracking
  #5 → Hybrid Search (BM25 keyword + Semantic)
  #6 → Format-Aware Answers (table / comparison / paragraph)
  #7 → Streaming Answers (partial results as they're generated)

All Changes Log:
  Round 1 — Prompt Engineering Improvements:
    [FIX]  Added max_tokens to both LLM instances (was missing)
    [TUNE] TOP_K_CHUNKS: 5 → 7
    [TUNE] CHUNK_SIZE: 1000 → 700
    [TUNE] CHUNK_OVERLAP: 200 → 140 (maintains 20% ratio)
    [TUNE] MEMORY_LAST_K: 5 → 3
    [TUNE] System prompt: added partial-info instruction
    [TUNE] Human prompt: added page citation enforcement
    [TUNE] Query rewrite: now domain-aware via document_type param
    [TUNE] load_pdf*: now returns truncation_warning for UI display

  Round 2 — Markdown + Context Window + Hallucination:
    [MD]   MAX_TOKENS: 1024 → 2048 (markdown answers are longer)
    [MD]   System prompt: added markdown formatting instruction
    [MD]   Human prompt: added "Respond in Markdown" footer
    [CTX]  New estimate_token_count() helper (4 chars ≈ 1 token)
    [CTX]  New CONTEXT_TOKEN_BUDGET = 28_000 (safe limit under 32k)
    [CTX]  New CHUNK_TOKEN_BUDGET = 6_000 (caps chunk tokens in prompt)
    [CTX]  ask_question(): dynamic history trim when context is tight
    [CTX]  ask_question(): chunk text capped at CHUNK_TOKEN_BUDGET
    [HAL]  System prompt: added uncertainty signalling instruction
    [HAL]  Human prompt: added [Low Confidence] prefix instruction
    [HAL]  rewrite_query(): temperature 0 → 0.1 (diverse rewriting)

  Round 3 — Format-Aware Answers + Streaming:
    [FMT]    New detect_response_format() — inspects the user's question for
             explicit table / comparison-difference / paragraph requests and
             returns an instruction string to inject into the prompt, so the
             model actually renders a Markdown table for tables/comparisons
             instead of defaulting to bullet points.
    [FMT]    Human prompt: added {format_instruction} placeholder.
    [FIX]    _strip_format_keywords() now actually wired into
             _prepare_pipeline_inputs() (it existed but was never called).
             Bug: "Give me the error codes table" was BM25 keyword-matching
             the unrelated "TABLE OF CONTENTS" page because both contain the
             literal word "table", diluting the real Error Codes pages out
             of the top retrieved chunks and causing the model to (wrongly)
             report the information as unavailable. Retrieval now searches
             on a cleaned query with format words removed; detect_response_format()
             still sees the original question so the requested format is kept.
    [FIX]    System prompt: added explicit decoupling — a FORMAT REQUIREMENT
             only changes presentation, never the availability decision, and
             tabular source content is preserved as a Markdown table by
             default even without an explicit "table" request.
    [REFAC]  Extracted shared retrieval + context-budget logic out of
             ask_question() into _prepare_pipeline_inputs(), reused by the
             new streaming path so both stay in sync.
    [STREAM] New ask_question_stream() — returns (source_pages, generator).
             The generator yields the answer text incrementally (as the LLM
             produces it) instead of blocking until the full answer is ready,
             so the UI can render partial results live.

  Round 4 — Multi-Page Table Completeness:
    [FIX] New _wants_comprehensive_listing() — detects table/comparison/
          "all of X" requests that imply an exhaustive answer, not a
          quick lookup.
    [FIX] New _boost_retriever_k() / _restore_retriever_k() — temporarily
          widen each sub-retriever's k for a single comprehensive call
          (mutating in place, no re-embedding needed), then restore.
          Bug: "give me the error codes table" only pulled 7-14 chunks
          total (TOP_K_CHUNKS=7 per sub-retriever), nowhere near enough to
          cover an error-codes table spanning 6 PDF pages — the middle
          pages were silently missing from the answer.
    [TUNE] New COMPREHENSIVE_TOP_K_CHUNKS=20 and
           CHUNK_TOKEN_BUDGET_COMPREHENSIVE=12_000, used only when
           _wants_comprehensive_listing() is true.
    [TUNE] MAX_TOKENS: 2048 → 3072 — a full multi-row table was getting
           cut off mid-table at the old limit.
    [FIX]  Human prompt: the "not available" sentence and the
           partial-information path are now explicitly mutually exclusive.
           Bug: the model was prefixing a perfectly good partial table
           with "The information is not available in the document." —
           contradicting itself in the same response.

  Round 5 — Full Multi-Page Section Coverage + Citation Consistency:
    [FIX] New _cluster_pages() / _expand_to_full_section() — after a
          comprehensive retrieval pass, detect when retrieved pages form
          one contiguous section (e.g. pages 60-65 for an error-codes
          table) and pull in EVERY chunk from that range out of the
          retriever's all_chunks, not just whichever ones ranked inside
          the boosted top-20. Fixes rows (E-14, E-6B, E-93, etc.) that
          were still missing even after the Round 4 k-boost, because
          their individual chunk ranked just outside the top-20.
    [FIX] EnsembleRetriever: new all_chunks field (text, page) pairs,
          populated in build_hybrid_retriever() — needed by the expansion
          above since BM25/FAISS only expose their top-k, not the full set.
    [FIX] retrieve_with_sources(): for comprehensive requests, chunks are
          now sorted by page number before being joined into context, so
          a multi-page table reads in document order instead of jumping
          around in relevance-rank order (the E-4/E-5/E-6A → E-41S...
          → E-10A... ordering bug).
    [FIX] retrieve_with_sources(): each chunk is now labelled
          "[PDF Page N]" using the TRUE PDF page index before being
          joined into context, and the human prompt now tells the model
          to cite using that label. Bug: the model was instead reading
          page numbers printed in the document's own header/footer text
          (e.g. "Nov 2016 60"), which is offset by several pages from the
          true PDF index (e.g. true page 66) — so the "(Page 60)" cited
          inline in the answer didn't match the "Sources: Page 66" tag
          shown in the UI for the same content.

  Round 6 — Reliability Hardening for Multi-Page Tables:
    [TUNE] COMPREHENSIVE_TOP_K_CHUNKS: 20 → 30. Even after Round 5's
           cluster expansion, a handful of rows (E-10, E-14, E-15C, E-15P)
           were still occasionally missing — wider initial retrieval
           means the expansion has less work to do and is less dependent
           on the cluster happening to bridge every gap correctly.
    [FIX]  rewrite_query() temperature is now a parameter; comprehensive
           requests use temperature=0 (was always 0.1). The slight
           wording randomness in query rewriting was shifting exactly
           which chunks landed in the initial top-k between identical
           runs of "give me the error codes table" — fine for a quick
           single-fact lookup, but it made multi-page table completeness
           non-deterministic. Determinism matters more than diversity
           once the request already implies "give me everything".
    [FIX]  System prompt: forbids the "(Same as Error Code X)" shorthand
           in tables. Verified bug: the model used this shorthand for
           E-10C and E-10D, claiming they shared E-10B's correction text
           — they actually match E-10A's wording instead (E-10B has an
           extra "low oil" caveat sentence that E-10A/C/D/F don't). Also
           saw E-41P and E-41S's DISPLAY labels swapped in the same
           answer. Both errors came from the model collapsing near-
           duplicate rows instead of transcribing each one independently
           — now it must write out full text per row.

Steps:
  Step 2 → Extract text from PDF
  Step 3 → Chunk + build hybrid retriever
  Step 5 → Build prompt (system + instructions + format + context + history + question)
  Step 6 → Build LLM chain
  Step 7 → Rewrite query → retrieve chunks → answer with sources (blocking or streamed)
"""

import re
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

from langchain_core.retrievers import BaseRetriever
from pydantic import Field


class EnsembleRetriever(BaseRetriever):
    """Combines multiple retrievers with weighted reciprocal rank scoring."""
    retrievers: list = Field(default_factory=list)
    weights: list = Field(default_factory=list)
    # [FIX] Round 5: every (chunk_text, page_number) pair, kept around so a
    # comprehensive request can pull in ALL chunks from a detected
    # multi-page section — not just whichever ones ranked in the top-k.
    all_chunks: list = Field(default_factory=list)

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

# Round 1 tuning
CHUNK_SIZE    = 700      
CHUNK_OVERLAP = 140      
TOP_K_CHUNKS  = 7        
MEMORY_LAST_K = 3        

# [MD] Round 2: MAX_TOKENS 1024 → 2048
# Markdown-formatted answers use more tokens (headings, bullets, spacing).
# [FIX] Round 4: 2048 → 3072 — a full multi-page error-codes table (40+ rows)
# was getting cut off mid-table at 2048 tokens.
MAX_TOKENS    = 3072

# [CTX] Round 2: Context window budget constants
# Mistral Small context window = 32,768 tokens.
# We reserve 28,000 for input (system + history + chunks + instructions)
# and leave ~4,000+ for the model's output (MAX_TOKENS=3072 + buffer).
CONTEXT_TOKEN_BUDGET = 28_000

# [CTX] Max tokens consumed by retrieved PDF chunks in the prompt.
# 7 chunks × 700 chars ≈ 1,225 tokens normally, but large pages can spike.
# Cap at 6,000 tokens (~24,000 chars) to leave room for history + system.
CHUNK_TOKEN_BUDGET   = 6_000

# [FIX] Round 4: comprehensive/listing requests need a wider retrieval net.
# Bug: "give me the error codes table" only retrieved 7-14 chunks (the
# ensemble's two sub-retrievers each return TOP_K_CHUNKS=7), which isn't
# enough to cover a table that spans 6 PDF pages — the middle pages of the
# table were silently dropped, and the model answered from a partial table
# without saying so clearly. For requests that imply "give me everything"
# (a table, a comparison, "all of X"), we temporarily widen retrieval.
COMPREHENSIVE_TOP_K_CHUNKS       = 30
CHUNK_TOKEN_BUDGET_COMPREHENSIVE = 12_000


# ── [CTX] Token estimation helper ────────────────────────────────────────────
def estimate_token_count(text: str) -> int:
    """
    [CTX] Rough token estimator: 1 token ≈ 4 characters (industry standard
    approximation for English/technical text without running a full tokenizer).

    Used to dynamically budget context window usage before invoking the chain,
    preventing silent truncation when history + chunks + prompts exceed 32k.
    """
    return max(1, len(text) // 4)


# ── STEP 2: Extract text from PDF ─────────────────────────────────────────────
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


# ── STEP 3: Chunk + Build Hybrid Retriever ────────────────────────────────────
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

    # Hybrid ensemble retriever
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.5, 0.5],
        # [FIX] Round 5: keep every (text, page) pair around so a
        # comprehensive request can later expand to a full page range.
        all_chunks=list(zip(chunk_texts, (m["page"] for m in chunk_metas))),
    )

    return hybrid_retriever


def load_pdf_from_bytes(pdf_bytes: bytes, filename: str) -> dict:
    """
    Full pipeline for UI upload:
      1. Extract text with page numbers
      2. Build hybrid retriever

    Returns dict with retriever, filename, page count, char count,
    and truncation_warning (None or a string message).
    """
    pages = extract_text_from_bytes(pdf_bytes)
    truncation_warning = None

    total_chars = sum(len(p["text"]) for p in pages)
    if total_chars > CHAR_LIMIT:
        kept, total = [], 0
        for p in pages:
            if total + len(p["text"]) > CHAR_LIMIT:
                break
            kept.append(p)
            total += len(p["text"])
        dropped = len(pages) - len(kept)
        pages = kept
        truncation_warning = (
            f"Large PDF detected — only first {len(pages)} pages were loaded "
            f"({dropped} pages dropped to stay within the {CHAR_LIMIT:,} character limit)."
        )
        print(f"Warning: {truncation_warning}")

    retriever = build_hybrid_retriever(pages)

    return {
        "retriever":          retriever,
        "filename":           filename,
        "pages":              len(pages),
        "chars":              sum(len(p["text"]) for p in pages),
        "truncation_warning": truncation_warning,
    }


def load_pdf(pdf_path: str) -> dict:
    """CLI version of load_pdf_from_bytes."""
    print(f"Loading: {pdf_path}")
    pages = extract_text_from_pdf(pdf_path)
    truncation_warning = None

    total_chars = sum(len(p["text"]) for p in pages)
    if total_chars > CHAR_LIMIT:
        kept, total = [], 0
        for p in pages:
            if total + len(p["text"]) > CHAR_LIMIT:
                break
            kept.append(p)
            total += len(p["text"])
        dropped = len(pages) - len(kept)
        pages = kept
        truncation_warning = (
            f"Large PDF — kept {len(pages)} pages ({dropped} pages dropped, "
            f"{sum(len(p['text']) for p in pages):,} chars loaded)."
        )
        print(f"Warning: {truncation_warning}")

    retriever = build_hybrid_retriever(pages)
    filename = pdf_path.split("/")[-1]
    chars = sum(len(p["text"]) for p in pages)
    print(f"Loaded '{filename}' — {chars:,} chars, {len(pages)} pages\n")

    return {
        "retriever":          retriever,
        "filename":           filename,
        "pages":              len(pages),
        "chars":              chars,
        "truncation_warning": truncation_warning,
    }


# ── STEP 3 (CLI): In-memory store ─────────────────────────────────────────────
pdf_store = {
    "retriever":          None,
    "filename":           "",
    "pages":              0,
    "chars":              0,
    "truncation_warning": None,
}


# ── #3: Query Rewriting ───────────────────────────────────────────────────────
def rewrite_query(question: str, document_type: str = "technical", temperature: float = 0.1) -> str:
    """
    #3: Rewrites vague/short user questions into more specific,
    search-friendly queries before retrieval.

    [HAL] temperature: 0 → 0.1
    A tiny amount of creativity in rewriting produces slightly varied phrasings,
    which improves retrieval diversity. Better retrieved chunks = more grounded
    answers = less hallucination. The answer LLM stays at temperature=0.

    [FIX] Round 6: temperature is now a parameter (default unchanged at 0.1).
    For comprehensive/listing requests, _prepare_pipeline_inputs() passes 0
    instead — run-to-run wording variance in the rewritten query was shifting
    exactly which chunks landed in the initial top-k before the page-range
    expansion kicked in, occasionally letting a chunk slip through a gap in
    the detected cluster. Determinism matters more than diversity once the
    request is "give me everything", not "find the one relevant fact".

    [TUNE] domain-aware: document_type param lets the rewriter use domain
    terminology (legal, medical, financial, technical) for better expansion.
    """
    llm = ChatMistralAI(
        api_key=MISTRAL_API_KEY,
        model=MODEL,
        temperature=temperature,  # [HAL] 0.1 normally; 0 for comprehensive requests (see above)
        max_tokens=MAX_TOKENS,
    )

    rewrite_prompt = (
        f"You are helping retrieve information from a {document_type} document. "
        "Rewrite the following question to be more specific and search-friendly "
        f"for retrieving content from this type of document. "
        "Expand abbreviations, add relevant domain terminology, and make the "
        "intent explicit. "
        "Return ONLY the rewritten question, nothing else.\n\n"
        f"Original: {question}\n"
        "Rewritten:"
    )

    result = llm.invoke(rewrite_prompt)
    rewritten = result.content.strip()
    return rewritten if rewritten else question


# ── #4: Retrieve chunks with source pages ─────────────────────────────────────
def _boost_retriever_k(retriever, new_k: int) -> list:
    """
    [FIX] Round 4: temporarily increases how many chunks each sub-retriever
    returns for a single call. Used for comprehensive/listing requests
    (e.g. "give me the full error codes table") where the default
    TOP_K_CHUNKS=7 per sub-retriever isn't enough to cover a table that
    spans several pages — mutating .k / .search_kwargs in place is cheap
    (no re-embedding needed) since the FAISS vectorstore itself is unchanged.

    Returns a list of (object, attr_name, original_value) tuples so the
    caller can restore the original settings via _restore_retriever_k().
    """
    restores = []
    sub_retrievers = getattr(retriever, "retrievers", [retriever])
    for sub in sub_retrievers:
        if hasattr(sub, "k"):
            restores.append((sub, "k", sub.k))
            sub.k = new_k
        if hasattr(sub, "search_kwargs"):
            old_kwargs = dict(sub.search_kwargs)
            restores.append((sub, "search_kwargs", old_kwargs))
            sub.search_kwargs = {**sub.search_kwargs, "k": new_k}
    return restores


def _restore_retriever_k(restores: list) -> None:
    """[FIX] Round 4: undo _boost_retriever_k() after a comprehensive call."""
    for obj, attr, old_value in restores:
        setattr(obj, attr, old_value)


class _SimpleDoc:
    """[FIX] Round 5: minimal stand-in for a retrieved chunk, used only by
    _expand_to_full_section() when pulling extra chunks straight out of
    all_chunks (which doesn't go through BM25/FAISS, so there's no real
    Document object to reuse)."""
    __slots__ = ("page_content", "metadata")

    def __init__(self, text: str, page: int):
        self.page_content = text
        self.metadata = {"page": page}


def _cluster_pages(pages: list, max_gap: int = 2) -> list:
    """
    [FIX] Round 5: groups a list of page numbers into clusters where
    consecutive pages are within `max_gap` of each other. Lets us tell the
    difference between "these chunks all belong to one contiguous document
    section" (e.g. a multi-page error-codes table) versus "these chunks
    are scattered across unrelated pages" (the normal case).
    """
    if not pages:
        return []
    pages = sorted(set(pages))
    clusters = [[pages[0]]]
    for p in pages[1:]:
        if p - clusters[-1][-1] <= max_gap:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return clusters


def _expand_to_full_section(retriever, retrieved_docs: list) -> list:
    """
    [FIX] Round 5: after a comprehensive retrieval pass, check whether the
    retrieved pages cluster into one contiguous section. If so, pull in
    EVERY chunk from that page range out of retriever.all_chunks — not
    just whichever ones happened to rank inside the (already widened)
    top-k.

    Concrete bug this fixes: "give me the error codes table" retrieved
    most of a 6-page error-code table (pages 60-65) even with a boosted
    k=20, but a handful of specific rows (e.g. E-14, E-6B, E-93) still
    didn't make the cut because their individual chunk ranked just outside
    the top-20. Expanding to the full detected page range guarantees
    nothing in that section is silently dropped.

    Small or non-clustered page sets (the normal case for most questions)
    are left untouched — this only kicks in once a real multi-page section
    is detected.
    """
    all_chunks = getattr(retriever, "all_chunks", None)
    if not all_chunks:
        return retrieved_docs

    pages_seen = [doc.metadata.get("page") for doc in retrieved_docs if doc.metadata.get("page")]
    clusters = _cluster_pages(pages_seen, max_gap=2)
    if not clusters:
        return retrieved_docs

    # Only expand the single largest cluster — that's almost certainly the
    # section the user is actually asking about. Small 1-2 page clusters
    # aren't worth expanding (most questions only need 1-2 pages anyway).
    largest = max(clusters, key=len)
    if len(largest) < 3:
        return retrieved_docs

    page_range = set(range(largest[0], largest[-1] + 1))
    already_have = {doc.page_content for doc in retrieved_docs}

    expanded = list(retrieved_docs)
    for text, page in all_chunks:
        if page in page_range and text not in already_have:
            expanded.append(_SimpleDoc(text, page))
            already_have.add(text)
    return expanded


def retrieve_with_sources(retriever, question: str, k_override: int = None) -> tuple:
    """
    #4: Retrieves relevant chunks and extracts source page numbers.

    [FIX] Round 4: k_override temporarily widens retrieval for this single
    call (see _boost_retriever_k) — used for comprehensive/listing requests
    where the default top-k is too narrow to cover a multi-page table.
    Settings are restored immediately after, so normal questions are
    unaffected.

    [FIX] Round 5 (only when k_override is set, i.e. comprehensive mode):
      - _expand_to_full_section() backfills any chunks missed by ranking
        alone, once a multi-page section is detected.
      - Chunks are then sorted by page number so a multi-page table reads
        in document order instead of jumping around in relevance-rank order.
      - Each chunk is labelled "[PDF Page N]" before being joined into the
        context. The model is instructed (see build_langchain_prompt) to
        cite USING THIS LABEL rather than any page number printed inside
        the document's own header/footer text, which is frequently
        different from the true PDF page index (e.g. front-matter/cover
        pages shift a document's internal page numbering by several
        pages). Without this, the "Sources: Page N" tags shown in the UI
        and the "(Page N)" citations inside the answer text can disagree
        about which page something is on.

    Returns (combined_context_text, sorted_unique_page_numbers).
    """
    restores = _boost_retriever_k(retriever, k_override) if k_override else []
    try:
        docs = retriever.invoke(question)
        if k_override:
            docs = _expand_to_full_section(retriever, docs)
            docs = sorted(docs, key=lambda d: d.metadata.get("page") or 0)
    finally:
        if restores:
            _restore_retriever_k(restores)

    context_parts = []
    pages_seen = set()

    for doc in docs:
        page = doc.metadata.get("page")
        if page:
            pages_seen.add(page)
            context_parts.append(f"[PDF Page {page}]\n{doc.page_content}")
        else:
            context_parts.append(doc.page_content)

    context = "\n\n---\n\n".join(context_parts)
    source_pages = sorted(pages_seen)
    return context, source_pages


# ── [FIX] Round 4: Comprehensive/listing request detection ────────────────────
def _wants_comprehensive_listing(question: str) -> bool:
    """
    True when the user is asking for an exhaustive listing (a full table, a
    complete comparison, "all of X") rather than a quick single-fact lookup.

    These requests need a wider retrieval net: a handful of top-ranked
    chunks is fine for "what's the cook temperature for wings", but a
    request for "the error codes table" implicitly means ALL error codes,
    which can span many chunks across several pages. A bare table or
    comparison request already implies completeness — that's the whole
    point of asking for a table instead of a quick answer — so it counts
    on its own, without needing an extra "all"/"complete" word.
    """
    q = f" {question.lower().strip()} "

    table_keywords = ["table", "tabular", "tabulate", "rows and columns"]
    compare_keywords = [
        "difference between", "differences between", "differ from",
        "compare", "comparison", " vs ", " vs.", " versus ",
        "contrast between", "similarities and differences",
    ]
    completeness_keywords = [
        "all ", "every ", "complete", "entire", "full list", "everything",
        "comprehensive", "list all",
    ]

    has_format_signal = any(kw in q for kw in table_keywords + compare_keywords)
    has_completeness_signal = any(kw in q for kw in completeness_keywords)
    return has_format_signal or has_completeness_signal


# ── #6: [FMT] Format-aware response detection ─────────────────────────────────
def detect_response_format(question: str) -> str:
    """
    [FMT] Inspects the user's question for an explicit formatting request
    (table / comparison-difference / paragraph) and returns an instruction
    string to inject into the prompt so the model actually structures its
    answer that way, instead of defaulting to its own judgement.

    Returns "" when no specific format is requested — the model falls back
    to the general Markdown instructions already in the system prompt.
    """
    q = f" {question.lower().strip()} "

    table_keywords = ["table", "tabular", "tabulate", "rows and columns"]
    compare_keywords = [
        "difference between", "differences between", "differ from",
        "compare", "comparison", " vs ", " vs.", " versus ",
        "contrast between", "similarities and differences",
    ]
    paragraph_keywords = [
        "in paragraph", "as a paragraph", "in prose", "paragraph form",
        "narrative form", "in a paragraph",
    ]

    is_table     = any(kw in q for kw in table_keywords)
    is_compare   = any(kw in q for kw in compare_keywords)
    is_paragraph = any(kw in q for kw in paragraph_keywords)

    # Comparison takes priority over a bare "table" mention (a comparison
    # request implies a table anyway, with a more specific structure).
    if is_compare:
        return (
            "FORMAT REQUIREMENT: The user is asking for a COMPARISON or "
            "DIFFERENCE between two or more items. Present the answer as a "
            "single Markdown table with the items being compared as columns "
            "and the comparison aspects as rows "
            "(e.g. `| Aspect | Item A | Item B |`), so each difference is "
            "clearly partitioned and easy to scan side-by-side. Add a short "
            "summary sentence after the table only if it adds value. Only "
            "fall back to a clearly labelled bullet list per item if a "
            "tabular layout genuinely does not fit the content. This formatting "
            "requirement does NOT override the partial-information rule above: "
            "build the best possible comparison from whatever is available "
            "rather than declining — only say the information is unavailable "
            "if it is truly absent from the retrieved content."
        )

    if is_table:
        return (
            "FORMAT REQUIREMENT: The user explicitly asked for a TABLE. "
            "You MUST respond using a single, well-structured Markdown table "
            "with clear column headers (e.g. `| Column A | Column B |`). "
            "Do not substitute bullet points or prose for the table. "
            "This formatting requirement does NOT override the partial-information "
            "rule above: if the relevant content is fragmented or incomplete, still "
            "build the best possible table from what is available rather than "
            "declining — only say the information is unavailable if it is truly "
            "absent from the retrieved content, not merely because the table is "
            "imperfect."
        )

    if is_paragraph:
        return (
            "FORMAT REQUIREMENT: The user explicitly asked for a "
            "PARAGRAPH-style answer. Respond in flowing prose paragraphs. "
            "Do NOT use bullet points, numbered lists, or tables for this "
            "answer."
        )

    return ""


# ── #6: [FMT] Retrieval-query cleaning ────────────────────────────────────────
_FORMAT_KEYWORD_PATTERNS = [
    r"\btables?\b", r"\btabular\b", r"\btabulate\b", r"\brows and columns\b",
    r"\bdifferences? between\b", r"\bdiffer from\b", r"\bcompare\b", r"\bcomparison\b",
    r"\bvs\.?\b", r"\bversus\b", r"\bcontrast between\b", r"\bsimilarities and differences\b",
    r"\bin paragraphs?(?: form)?\b", r"\bas a paragraph\b", r"\bin prose\b",
    r"\bnarrative form\b", r"\bin a paragraph\b",
]


def _strip_format_keywords(question: str) -> str:
    """
    [FMT] Removes format-instruction words ("table", "compare", "vs",
    "paragraph", ...) from the text used for retrieval, so keyword-based
    (BM25) search isn't polluted by meta-words about HOW to format the
    answer rather than WHAT the answer is about.

    Concrete bug this fixes: "Give me the error codes table" was BM25
    keyword-matching the unrelated "TABLE OF CONTENTS" page purely because
    both contain the literal word "table", diluting the actually-relevant
    Error Codes pages out of the top results and causing the model to
    (incorrectly) report the information as unavailable.

    The ORIGINAL question (with "table" intact) is still used for the final
    answer prompt and for detect_response_format() — only the text handed
    to rewrite_query()/the retriever is cleaned.
    """
    cleaned = question
    for pattern in _FORMAT_KEYWORD_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned if cleaned else question


# ── STEP 5: Build prompt ──────────────────────────────────────────────────────
def build_langchain_prompt() -> ChatPromptTemplate:
    """
    Step 5: Builds the ChatPromptTemplate with all tuned prompt components.

    Round 1 changes:
      - System prompt: partial-info clause added
      - Human prompt: page citation enforcement added

    Round 2 changes:
      [MD]  System prompt: explicit Markdown formatting instruction added
      [MD]  Human prompt footer: "Respond in well-structured Markdown" added
      [HAL] System prompt: uncertainty signalling clause added
            — model must say "I am not certain" instead of guessing confidently
      [HAL] Human prompt: [Low Confidence] prefix instruction added
            — gives the model an explicit escape hatch for borderline answers

    Round 3 changes:
      [FMT] Human prompt: {format_instruction} placeholder added — filled in
            by detect_response_format() per-question (table / comparison /
            paragraph), or left blank when no specific format was requested.
    """

    # ── System Prompt ──────────────────────────────────────────────────────────
    # [MD]  Added: markdown formatting instruction
    # [HAL] Added: uncertainty signalling instruction
    # Round 1: partial-info clause already present
    system_prompt = (
        "You are a precise document assistant. "
        "Your ONLY source of truth is the PDF content provided by the user. "
        "Never use external knowledge or make assumptions beyond "
        "what is explicitly stated in the document. "

        # Round 1: partial info
        "If partial information is available in the document, share what is present "
        "and clearly state what specific details are missing or not covered. "

        # [HAL] Uncertainty signalling — forces model to flag low-confidence answers
        # instead of presenting guesses as facts. This is the single most effective
        # prompt-level hallucination deterrent.
        "When you are uncertain whether information comes from the document or your "
        "own training data, explicitly say: 'I am not certain this is stated in the "
        "document — please verify on the relevant page.' "
        "Never present inferred or assumed information as if it were stated in the document. "

        # [MD] Markdown instruction — tells the model HOW to format output
        "Format all your answers using Markdown. Use ## headings for sections, "
        "**bold** for key terms, bullet lists for enumerated items, and ``` code blocks ``` "
        "for any technical content or direct quotes. Always structure longer answers "
        "with clear sections. When the source material itself is laid out as parallel "
        "columns (for example a Display/Cause/Correction error table, or an "
        "Item/Description/Function table), preserve that structure as a Markdown table "
        "in your answer by default, even if the user did not explicitly ask for a table. "

        # [FMT] Format requirement compliance
        "If a FORMAT REQUIREMENT is given below, it overrides your default formatting "
        "judgement for that specific answer — follow it exactly. A FORMAT REQUIREMENT is "
        "a presentation instruction ONLY: it never changes whether information is "
        "available. First decide whether the document contains the answer by looking at "
        "the PDF CONTENT alone; only after that, apply the requested format to whatever "
        "you found. Reorganize facts into the requested format even if the source page "
        "doesn't present them as a clean table, or if the relevant content is split "
        "across multiple chunks — never respond that the information is unavailable "
        "merely because fitting it into the requested format takes extra reorganizing. "

        # [FIX] Round 6: forbid the "(Same as X)" shorthand in tables — it
        # caused real mislabeling (e.g. claiming two error codes shared a
        # correction when they actually had different wording, because two
        # entries looked similar at a glance). Writing out each row in full
        # costs more tokens but removes that entire failure mode.
        "When listing multiple similar rows in a table (e.g. several rows that share "
        "very similar but not necessarily identical wording), write out the full text "
        "for EVERY row from the source content. Never write a shorthand reference like "
        "'(Same as Error Code X)' or '(See above)' to avoid repeating text — even when "
        "two rows look similar at a glance, their exact wording can differ in a way "
        "that matters, and shorthand references have caused incorrect groupings before."
    )

    # ── Human Prompt ───────────────────────────────────────────────────────────
    # [MD]  Added: "Respond in well-structured Markdown" as final instruction
    # [HAL] Added: [Low Confidence] prefix instruction
    # [FMT] Added: {format_instruction} placeholder
    # Round 1: page citation instruction already present
    human_prompt = """INSTRUCTIONS:
- Answer strictly and only from the PDF content provided below.
- If the document contains NO relevant information at all, respond EXACTLY with:
  "The information is not available in the document." — and nothing else in that case.
- If you found ANY relevant information, even if incomplete, scattered across multiple
  pages, or not laid out as a clean table in the source — do NOT use the sentence above.
  Share what you found and clearly state what specific details are missing instead.
  Never combine the exact "not available" sentence with a partial answer in the same
  response; these two paths are mutually exclusive.
- Do not use external knowledge, prior training data, or assumptions.
- Be concise and accurate. Quote or paraphrase directly from the document.
- Always cite the page number(s) where you found the information, e.g. "(Page 4)" or "(Pages 2, 7)".
  Use the page number from the "[PDF Page N]" label given right before each excerpt below — never use a
  page number printed inside the document's own header/footer text, since a document's internal page
  numbering (e.g. "March 2014 1") often differs from its true position in the PDF file and will not match
  the source page numbers shown elsewhere in this app.
- If you are less than fully confident about an answer, prefix it with **[Low Confidence]** and explain what you are unsure about.
- Respond in well-structured Markdown with headings, bullets, and bold key terms where appropriate.

{format_instruction}

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
    """
    Step 6: Prompt -> Mistral -> StrOutputParser.

    [MD] MAX_TOKENS raised to 2048 — markdown answers need more room than
    plain text (headings, bullets, spacing all consume tokens).
    temperature=0 stays on the answer LLM for maximum factual accuracy.

    [STREAM] This same chain (a standard LangChain RunnableSequence) supports
    both .invoke() (blocking, used by ask_question) and .stream() (used by
    ask_question_stream) — no separate chain needed for streaming.
    """
    llm = ChatMistralAI(
        api_key=MISTRAL_API_KEY,
        model=MODEL,
        temperature=0,         # answer LLM stays deterministic
        max_tokens=MAX_TOKENS, # [MD] 2048 — enough for structured markdown responses
    )
    prompt = build_langchain_prompt()
    parser = StrOutputParser()
    return prompt | llm | parser


# ── #2: Memory helpers ────────────────────────────────────────────────────────
def format_chat_history(chat_history: list, last_k: int = MEMORY_LAST_K) -> str:
    """
    #2: Formats last K chat exchanges into a string for the prompt.

    Round 1: last_k reduced from 5 → 3.
    The caller may pass a lower last_k dynamically when the context budget
    is tight — see [CTX] logic in _prepare_pipeline_inputs().
    """
    recent = chat_history[-(last_k * 2):]

    if not recent:
        return ""

    lines = ["CONVERSATION HISTORY:"]
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")

    return "\n".join(lines) + "\n"


# ── [REFAC] Shared pipeline prep (used by both blocking and streaming paths) ──
def _prepare_pipeline_inputs(
    retriever,
    user_question: str,
    chat_history: list,
    document_type: str,
) -> tuple:
    """
    Runs the full retrieval + context-budget + format-detection pipeline and
    returns (chain_inputs_dict, source_pages) ready to hand to chain.invoke()
    or chain.stream().

      #3      -> Rewrite query (domain-aware, temperature=0.1)
      #1+#5   -> Retrieve relevant chunks via hybrid search
      #4      -> Extract source page numbers
      [CTX]   -> Budget check: trim chunks and/or history if over limit
      #2      -> Format conversation history (dynamic last_k)
      [FMT]   -> Detect requested response format (table / comparison / paragraph)

    [CTX] Context budget logic:
      1. Estimate tokens for retrieved chunks.
      2. If chunks exceed CHUNK_TOKEN_BUDGET, truncate the combined text.
      3. Estimate tokens for full history (last MEMORY_LAST_K exchanges).
      4. If history + chunks together are too large, reduce history to 1 exchange.
      5. If still over budget, drop history entirely.
      This ensures the prompt never silently exceeds Mistral Small's 32k limit.
    """
    # [FMT] Strip format-instruction words ("table", "compare", "vs", ...)
    # before retrieval. These words describe HOW to present the answer, not
    # WHAT the answer is about, and literally matching them via BM25 can
    # pull in unrelated pages (e.g. "table" matching a "TABLE OF CONTENTS"
    # page). detect_response_format() below still sees the ORIGINAL
    # question, so the requested format is never lost — only the text used
    # for search is cleaned.
    retrieval_question = _strip_format_keywords(user_question)

    # [FIX] Round 6: detect comprehensive intent BEFORE rewriting, so we can
    # make the rewrite deterministic for these requests (see rewrite_query
    # docstring) — completeness matters more than wording diversity here.
    comprehensive = _wants_comprehensive_listing(user_question)
    rewrite_temperature = 0.0 if comprehensive else 0.1

    # #3: Rewrite query for better retrieval (domain-aware)
    search_query = rewrite_query(
        retrieval_question, document_type=document_type, temperature=rewrite_temperature
    )

    # [FIX] Round 4: widen retrieval + chunk budget for comprehensive/listing
    # requests (full tables, comparisons, "all of X") — see
    # _wants_comprehensive_listing() and COMPREHENSIVE_TOP_K_CHUNKS above.
    retrieval_k = COMPREHENSIVE_TOP_K_CHUNKS if comprehensive else None
    chunk_budget = CHUNK_TOKEN_BUDGET_COMPREHENSIVE if comprehensive else CHUNK_TOKEN_BUDGET

    # #1 + #5 + #4: Hybrid retrieval with source pages
    pdf_content, source_pages = retrieve_with_sources(retriever, search_query, k_override=retrieval_k)

    # ── [CTX] Step 1: Cap chunk tokens ────────────────────────────────────────
    chunk_tokens = estimate_token_count(pdf_content)
    if chunk_tokens > chunk_budget:
        max_chunk_chars = chunk_budget * 4
        pdf_content = pdf_content[:max_chunk_chars]
        print(f"[CTX] Chunks trimmed: {chunk_tokens} → {chunk_budget} tokens")

    # ── [CTX] Step 2: Dynamic history trim ────────────────────────────────────
    history_list = chat_history or []
    effective_last_k = MEMORY_LAST_K

    full_history_text  = format_chat_history(history_list, last_k=MEMORY_LAST_K)
    short_history_text = format_chat_history(history_list, last_k=1)

    chunk_tok   = estimate_token_count(pdf_content)
    history_tok = estimate_token_count(full_history_text)
    system_tok  = 600   # rough estimate for system + human prompt scaffolding

    total_estimated = chunk_tok + history_tok + system_tok

    if total_estimated > CONTEXT_TOKEN_BUDGET:
        short_tok = estimate_token_count(short_history_text)
        if chunk_tok + short_tok + system_tok <= CONTEXT_TOKEN_BUDGET:
            effective_last_k = 1
            print(f"[CTX] History trimmed to 1 exchange (budget: {total_estimated} > {CONTEXT_TOKEN_BUDGET})")
        else:
            effective_last_k = 0
            print(f"[CTX] History dropped entirely (budget: {total_estimated} > {CONTEXT_TOKEN_BUDGET})")

    history_text = format_chat_history(history_list, last_k=effective_last_k)

    # [FMT] Detect explicit format request (table / comparison / paragraph)
    format_instruction = detect_response_format(user_question)

    chain_inputs = {
        "pdf_content":        pdf_content,
        "user_question":      user_question,
        "chat_history":       history_text,
        "format_instruction": format_instruction,
    }
    return chain_inputs, source_pages


# ── STEP 7: Ask question (full smart pipeline, blocking) ──────────────────────
def ask_question(
    chain,
    retriever,
    user_question: str,
    chat_history: list = None,
    document_type: str = "technical",
) -> tuple:
    """
    Step 7: Full smart pipeline (blocking). Builds chain inputs via
    _prepare_pipeline_inputs(), invokes the chain once, and returns the
    complete answer.

    Returns:
        (answer_string, [source_page_numbers])
    """
    if retriever is None:
        return "Please load a PDF first.", []

    chain_inputs, source_pages = _prepare_pipeline_inputs(
        retriever, user_question, chat_history, document_type
    )

    answer = chain.invoke(chain_inputs)
    return answer, source_pages


# ── STEP 7 (STREAMING): Ask question with live partial output ─────────────────
def ask_question_stream(
    chain,
    retriever,
    user_question: str,
    chat_history: list = None,
    document_type: str = "technical",
) -> tuple:
    """
    [STREAM] Streaming counterpart to ask_question(). Runs the same
    retrieval + context-budget + format-detection pipeline up front
    (fast — no LLM generation yet), then returns immediately with:

        (source_pages, answer_chunk_generator)

    `source_pages` is already known at this point (retrieval has finished),
    so the UI can display "Sources: Page X" right away. The caller should
    iterate `answer_chunk_generator` to receive the answer text as it is
    generated, chunk by chunk, instead of waiting for the full response —
    this is what lets the UI show partial/in-progress output like a real
    chat product, rather than a blank screen until everything is ready.

    Returns:
        (source_pages: list[int], generator yielding str chunks)
    """
    if retriever is None:
        def _no_pdf_gen():
            yield "Please load a PDF first."
        return [], _no_pdf_gen()

    chain_inputs, source_pages = _prepare_pipeline_inputs(
        retriever, user_question, chat_history, document_type
    )

    def _answer_generator():
        for chunk in chain.stream(chain_inputs):
            # StrOutputParser streams plain string deltas.
            yield chunk

    return source_pages, _answer_generator()


# ── CLI Entry Point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "document.pdf"

    # Optional: pass document type as second CLI arg
    # e.g.  python pdf_chatbot.py report.pdf legal
    #        python pdf_chatbot.py notes.pdf medical
    doc_type = sys.argv[2] if len(sys.argv) > 2 else "technical"

    result = load_pdf(pdf_path)
    pdf_store.update(result)

    if pdf_store.get("truncation_warning"):
        print(f"\n⚠  {pdf_store['truncation_warning']}\n")

    chain = build_chain()
    cli_history = []

    print("=" * 55)
    print(" PDF Chatbot ready! (v4 — Format-Aware + Streaming)")
    print(f"   Document type : {doc_type}")
    print(f"   Max tokens    : {MAX_TOKENS}")
    print(f"   Context budget: {CONTEXT_TOKEN_BUDGET:,} tokens")
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

        # CLI streams to stdout too, to exercise the same code path as the UI.
        source_pages, answer_gen = ask_question_stream(
            chain,
            pdf_store["retriever"],
            user_input,
            cli_history,
            document_type=doc_type,
        )

        print("\n Answer:")
        full_answer = ""
        for piece in answer_gen:
            print(piece, end="", flush=True)
            full_answer += piece
        print()

        if source_pages:
            print(f" Sources: Page(s) {', '.join(map(str, source_pages))}")

        cli_history.append({"role": "assistant", "content": full_answer})