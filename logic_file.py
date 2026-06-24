"""
Task 1: PDF Intelligence Chatbot — Core Logic

  #1 → Semantic Chunking + FAISS Vector Retrieval
  #2 → Conversation Memory (last 3 exchanges, dynamic trim)
  #3 → Query Rewriting (vague → specific, domain-aware)
  #4 → Source Page Tracking
  #5 → Hybrid Search (BM25 keyword + Semantic)
  #6 → Format-Aware Answers (table / comparison / paragraph)
  #7 → Streaming Answers (partial results as they're generated)
  #8 → Structured JSON Output (Tag + Response, query classification)
  #9 → Universal Recall Fallback (don't say "not available" too eagerly)

All Changes Log:
  Round 1 — Prompt Engineering Improvements:
    [FIX]  Added max_tokens to both LLM instances (was missing)
    [TUNE] TOP_K_CHUNKS, CHUNK_SIZE, CHUNK_OVERLAP, MEMORY_LAST_K tuned
    [TUNE] System prompt: added partial-info instruction
    [TUNE] Human prompt: added page citation enforcement
    [TUNE] Query rewrite: now domain-aware via document_type param
    [TUNE] load_pdf*: now returns truncation_warning for UI display

  Round 2 — Markdown + Context Window + Hallucination:
    [MD]   MAX_TOKENS raised — markdown answers are longer
    [CTX]  estimate_token_count(), CONTEXT_TOKEN_BUDGET, CHUNK_TOKEN_BUDGET
    [HAL]  Uncertainty signalling + [Low Confidence] prefix instruction
    [HAL]  rewrite_query(): temperature 0 → 0.1 (diverse rewriting)

  Round 3 — Format-Aware Answers + Streaming:
    [FMT]    detect_response_format() — table / comparison / paragraph detection
    [FIX]    _strip_format_keywords() wired into retrieval (was unused)
    [FIX]    System prompt: FORMAT REQUIREMENT decoupled from availability
    [REFAC]  _prepare_pipeline_inputs() shared by blocking + streaming paths
    [STREAM] ask_question_stream() — partial results instead of a blank wait

  Round 4 — Multi-Page Table Completeness:
    [FIX]  _wants_comprehensive_listing() + _boost_retriever_k() — widen
           retrieval for table/comparison/"all of X" requests
    [TUNE] COMPREHENSIVE_TOP_K_CHUNKS, CHUNK_TOKEN_BUDGET_COMPREHENSIVE
    [FIX]  "not available" vs partial-info made mutually exclusive

  Round 5 — Full Multi-Page Section Coverage + Citation Consistency:
    [FIX] _cluster_pages() / _expand_to_full_section() — backfill every
          chunk in a detected contiguous section, not just top-k ranked ones
    [FIX] EnsembleRetriever.all_chunks — full (text, page) index for that
    [FIX] retrieve_with_sources(): page-sorted output + "[PDF Page N]"
          labels so citations match the true PDF index, not header text

  Round 6 — Reliability Hardening for Multi-Page Tables:
    [TUNE] COMPREHENSIVE_TOP_K_CHUNKS: 20 → higher
    [FIX]  rewrite_query() temperature=0 for comprehensive requests
           (determinism over diversity once "give me everything" is implied)
    [FIX]  System prompt forbids "(Same as Error Code X)" shorthand —
           verified to cause real mislabeling between near-duplicate rows

  Round 7 — Structured JSON Output + Universal Recall + Efficiency:
    [PROMPT] Rebuilt the prompt around the user-supplied JSON contract:
             every answer is classified into one of four shapes — Relevant
             ({"Tag": ..., "Response": ...}), Vague/Incomplete, Non-Relevant,
             or Greeting (all three: {"Response": ...}). All Round 1-6
             grounding rules (citation, anti-shorthand, partial-info,
             format-requirement decoupling) now apply specifically to how
             the Response field is composed for Relevant Queries.
    [FIX]    Native JSON mode requested from the API (response_format=
             {"type": "json_object"}) when the installed langchain-mistralai
             supports it, with a defensive fallback to prompt-only JSON
             instructions otherwise — belt-and-suspenders for valid output.
    [FIX]    _parse_json_answer(): three-level fallback (strict json.loads
             → tolerant regex extraction → raw passthrough) so a малformed
             or imperfectly-escaped JSON response never produces a blank or
             crashed answer.
    [FIX]    Streaming redesigned to buffer the full JSON response, parse
             it once it's complete and validated, then reveal the
             guaranteed-correct final text in small increments for a smooth
             "typing" animation. Naively live-streaming raw partial JSON
             was considered and rejected: the UI can only ever APPEND
             yielded text (never replace it), so any approximate live
             unescaping that later turned out wrong would have permanently
             corrupted the stored chat history — not an acceptable trade
             for a cosmetic animation. Time-to-first-byte through the API
             is unaffected; only what we do with the bytes changed.
    [FIX]    New _retrieve_with_recall_fallback(): if the topic is actually
             in the document, a question about it should get an answer
             regardless of exact phrasing. Escalates through up to three
             retrieval attempts — normal k, then a much wider k with the
             same (rewritten) query, then the wider k again with the user's
             RAW original question (bypassing rewriting/keyword-stripping
             entirely) — stopping as soon as one attempt clears a minimum
             usefulness bar, so a single thin/empty retrieval no longer
             leads straight to "not available".
    [FIX]    extract_text_from_bytes/pdf no longer collapse ALL whitespace
             (including newlines) into single spaces. That collapsed every
             page into one undifferentiated blob of text, destroying the
             very table-row/list-item structure later rounds depend on to
             extract clean tables. Now only intra-line whitespace runs are
             collapsed and excessive blank lines are tamed; line breaks
             that mark structure are preserved.
    [PERF]   _get_llm(): ChatMistralAI clients are now created once per
             (model, temperature, max_tokens, json_mode) combination and
             reused, instead of constructing a brand-new client on every
             single rewrite_query() call.
    [PERF]   Debug prints replaced with logging.debug() calls — near-zero
             cost when the host app doesn't enable DEBUG logging, instead
             of unconditionally writing to stdout on every request.

Steps:
  Step 2 → Extract text from PDF
  Step 3 → Chunk + build hybrid retriever
  Step 5 → Build prompt (system + instructions + format + context + history + question)
  Step 6 → Build LLM chain
  Step 7 → Rewrite query → retrieve chunks (with recall fallback) → answer
           with sources (blocking or streamed) → parse structured JSON output
"""

import json
import logging
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

# [PERF] Round 7: module logger instead of unconditional print() calls.
# Silent by default; the host app can opt in with:
#   logging.getLogger("logic_file").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


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
        # [FIX] Round 7.6: stamp the computed fusion score onto each doc's
        # own metadata before returning. Previously this score was computed
        # and then thrown away — callers only ever saw a flat ranked list
        # with no sense of HOW MUCH better the top hits were than the
        # bottom ones. That made a fixed-count cutoff the only lever
        # available, which can't tell a genuinely strong match (the actual
        # Basic Operation section) apart from a much weaker one that still
        # happened to land inside the top N (e.g. an Error Codes chunk that
        # shares generic step-by-step phrasing). Exposing the score lets
        # _filter_by_relevance_gap() drop weak tail matches regardless of
        # their numeric rank position.
        for item in sorted_docs:
            item["doc"].metadata["_relevance_score"] = item["score"]
        return [item["doc"] for item in sorted_docs]


# ── Environment Setup ─────────────────────────────────────────────────────────
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in .env file!")

# Model kept exactly as configured — not changed in this round.
MODEL         = "mistral-small-2506"
CHAR_LIMIT    = 200_000

# Chunking / retrieval tuning (unchanged from the current configuration)
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 250
TOP_K_CHUNKS  = 10
MEMORY_LAST_K = 3

# Markdown-formatted, JSON-wrapped answers need headroom for headings,
# bullets, tables, and the JSON envelope itself.
# [FIX] Round 7.1: 3072 -> 6144. Wrapping the answer in a JSON string means
# every literal double-quote and newline in the Markdown content has to be
# escaped (" -> \" , newline -> \n), which costs MORE characters than the
# same content as raw Markdown ever did. A 37-row error-codes table that
# fit comfortably before was getting cut off mid-table (and mid-word) at
# the old limit once wrapped in JSON — the model ran out of output budget
# partway through, silently dropping every row after the cutoff.
MAX_TOKENS    = 6144

# Mistral Small context window = 32,768 tokens. Reserve 24,000 for input
# (system + history + chunks + instructions), leaving ~6,000+ for output
# (MAX_TOKENS=6144) plus a safety buffer — still comfortably under 32,768
# combined even in the worst case (24,000 + 6,144 = 30,144).
CONTEXT_TOKEN_BUDGET = 24_000

# Max tokens consumed by retrieved PDF chunks in the prompt for a normal
# (non-comprehensive) question.
CHUNK_TOKEN_BUDGET   = 8_000

# [FIX] Round 4: comprehensive/listing requests (full tables, comparisons,
# "all of X") need a much wider retrieval net — a handful of top-ranked
# chunks covers one fact, not a table spanning several pages.
COMPREHENSIVE_TOP_K_CHUNKS       = 50
CHUNK_TOKEN_BUDGET_COMPREHENSIVE = 18_000

# [DEPRECATED] Round 7.2 introduced this as a middle tier for generic
# completeness wording ("all", "every"). Round 7.5 removed its usage —
# widening k to 20 for those generic words genuinely matched scattered
# pages across most of a procedural manual (every section uses similar
# step-by-step phrasing), which then got padded into nearly half the
# document. Left defined in case a future, more targeted use is found;
# _prepare_pipeline_inputs() no longer references it.
MODERATE_TOP_K_CHUNKS = 20

# [FIX] Round 7: minimum amount of retrieved content (in characters) before
# we trust a retrieval pass enough to skip the recall-fallback escalation.
# Deliberately small — this only needs to catch the "basically nothing
# came back" case, not second-guess every modest-but-valid retrieval.
MIN_USEFUL_RETRIEVAL_CHARS = 200

# [FIX] Round 7.5: EnsembleRetriever merges BM25+FAISS results with no
# final cutoff of its own — every unique chunk either sub-retriever
# surfaced within its OWN k gets returned, so even the default k=10 per
# sub-retriever can yield up to ~20 merged candidates. Generic step-by-
# step phrasing ("press the button", "follow the steps") that's common
# across an entire procedural manual let marginal matches from unrelated
# sections slip into that long tail. For normal (non-comprehensive)
# questions, only the top DEFAULT_FINAL_TOP_N merged candidates are kept
# before any padding — comprehensive/expand_sections requests skip this
# cap since they're intentionally meant to be broad.
DEFAULT_FINAL_TOP_N = 5

# [FIX] Round 7.6: a purely STRUCTURAL safety ceiling, independent of any
# ranking score. Generic procedural phrasing ("press the button", "follow
# the steps") repeats throughout a technical manual, and can occasionally
# rank a chunk from a FAR AWAY, unrelated section highly enough to survive
# rank-based filtering (DEFAULT_FINAL_TOP_N, padding's top_n). Related
# content in a well-organized manual is almost always within a handful of
# pages of the single most confident match — anything farther than this
# from that anchor page is dropped for non-comprehensive questions,
# regardless of how it scored. See _restrict_to_anchor_locality().
MAX_PAGE_DISTANCE_FROM_ANCHOR = 8


# ── [PERF] Round 7: cached LLM clients ─────────────────────────────────────────
_llm_cache: dict = {}


def _get_llm(model: str, temperature: float, max_tokens: int, json_mode: bool = False):
    """
    [PERF] Round 7: lazily creates and reuses ChatMistralAI client objects
    instead of constructing a brand-new one on every call — the previous
    rewrite_query() built a fresh client every single time it ran.

    [FIX] Round 7: when json_mode=True (used for the main answer chain),
    requests native JSON-object output from the API via response_format.
    This is a defensive best-effort: if the installed langchain-mistralai
    version doesn't accept that kwarg, we transparently fall back to a
    plain client and rely on the prompt instructions + the tolerant
    _parse_json_answer() fallback chain instead. Either way, output is
    never blocked by this — it only improves the odds of clean JSON.
    """
    key = (model, temperature, max_tokens, json_mode)
    if key in _llm_cache:
        return _llm_cache[key]

    base_kwargs = dict(
        api_key=MISTRAL_API_KEY,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    llm = None
    if json_mode:
        try:
            llm = ChatMistralAI(**base_kwargs, response_format={"type": "json_object"})
        except TypeError:
            logger.debug(
                "Installed langchain-mistralai doesn't accept response_format; "
                "falling back to prompt-only JSON instructions."
            )
    if llm is None:
        llm = ChatMistralAI(**base_kwargs)

    _llm_cache[key] = llm
    return llm


# ── STEP 2: Extract text from PDF ─────────────────────────────────────────────
def _normalize_extracted_text(text: str) -> str:
    """
    [FIX] Round 7: collapses only INTRA-LINE whitespace runs (repeated
    spaces/tabs) and tames excessive blank lines, but preserves real line
    breaks.

    Bug this fixes: an earlier version replaced ALL whitespace (including
    every newline) with a single space, which flattened each page into one
    undifferentiated blob of text. That destroyed the row/list-item
    structure that later rounds (multi-page table detection, clean
    Markdown table reconstruction) depend on — a table that reads as
    distinct rows in the PDF would otherwise become one run-on sentence
    with no boundaries between "DISPLAY", "CAUSE", and "CORRECTION" values.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)      # collapse repeated spaces/tabs
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)  # trim whitespace around line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)   # tame excessive blank lines
    return text.strip()


def extract_text_from_bytes(pdf_bytes: bytes) -> list:
    """
    Extracts text page-by-page from raw PDF bytes, preserving ligatures,
    de-hyphenating wrapped words, and clipping to the visible mediabox —
    while keeping line/row structure intact (see _normalize_extracted_text).
    Returns list of {page, text} dicts for source tracking.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text(
            "text",
            flags=fitz.TEXT_PRESERVE_LIGATURES
            | fitz.TEXT_PRESERVE_WHITESPACE
            | fitz.TEXT_MEDIABOX_CLIP
            | fitz.TEXT_DEHYPHENATE,
        )
        text = _normalize_extracted_text(text)
        if len(text) > 50:  # ignore near-empty pages
            pages.append({"page": page_num, "text": text})
    doc.close()
    if not pages:
        raise ValueError("No readable text found. It may be a scanned/image PDF.")
    return pages


def extract_text_from_pdf(pdf_path: str) -> list:
    """CLI version of extract_text_from_bytes."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text(
            "text",
            flags=fitz.TEXT_PRESERVE_LIGATURES
            | fitz.TEXT_PRESERVE_WHITESPACE
            | fitz.TEXT_MEDIABOX_CLIP
            | fitz.TEXT_DEHYPHENATE,
        )
        text = _normalize_extracted_text(text)
        if len(text) > 50:
            pages.append({"page": page_num, "text": text})
    doc.close()
    if not pages:
        raise ValueError("No readable text found. It may be a scanned/image PDF.")
    return pages


# ── STEP 3: Chunk + Build Hybrid Retriever ────────────────────────────────────
def _is_toc_like_chunk(text: str) -> bool:
    """
    [FIX] Round 7.4: heuristically detects a Table-of-Contents-style chunk
    — dominated by lines like "1-1 Safety .......................... 1"
    (a title followed by dot-leaders/whitespace and a trailing page
    number) or numbered section headers ("2-3 Selecting the Location").

    Concrete bug this fixes: "give me all basic operations in detail"
    retrieved the actual Table of Contents page alongside the real Basic
    Operation content, because the TOC literally contains the words
    "Basic Operation" (it's an entry in it) plus dozens of other section
    titles. The model then tried to exhaustively enumerate every title it
    saw there — generating empty "please refer to the X section" filler
    for sections completely unrelated to what was asked, repeated several
    times over. A TOC entry is a navigation aid (title -> page number), not
    real content, so it's excluded from the index entirely rather than
    relying on the model to ignore it (the model demonstrably didn't).

    Conservative by design (>=50% of a chunk's non-empty lines need to
    match) so a normal numbered-steps procedure — which starts with a
    digit but ends in actual instruction text, not a page number — is not
    mistaken for one.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 5:
        return False
    toc_like = sum(
        1 for l in lines
        if re.search(r"[.\s]{2,}\d{1,4}\s*$", l)                   # "Title .......... 42"
        or re.match(r"^\d+(-\d+)*\.?\s+[A-Z][A-Za-z ]{2,40}$", l)  # "2-3 SELECTING THE LOCATION"
    )
    return (toc_like / len(lines)) >= 0.5


def _pages_to_chunks(pages: list) -> tuple:
    """
    Splits page texts into overlapping chunks.

    [FIX] Round 7.4: chunks that are themselves Table-of-Contents-style
    listings (see _is_toc_like_chunk) are excluded from the index — they
    contain section TITLES, not section CONTENT, and including them
    caused the model to try to summarize every title it saw rather than
    answering the actual question. A page that is ENTIRELY a table of
    contents (the common case — most manuals have one dedicated TOC page)
    is therefore dropped from the index entirely, which is the intended,
    correct outcome here, not a bug: as a navigation aid it was actively
    harmful to retrieval quality and added no answerable content of its
    own. The safety net only applies GLOBALLY — if filtering would leave
    the WHOLE document with zero chunks (a pathological all-TOC PDF),
    every chunk is kept unfiltered rather than building a broken,
    contentless retriever.

    Returns (chunk_texts, chunk_metadatas) — metadata carries page number.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    all_splits = []        # [(text, page), ...] before TOC filtering
    filtered_splits = []   # [(text, page), ...] after TOC filtering

    for page_data in pages:
        for split in splitter.split_text(page_data["text"]):
            all_splits.append((split, page_data["page"]))
            if not _is_toc_like_chunk(split):
                filtered_splits.append((split, page_data["page"]))

    kept = filtered_splits if filtered_splits else all_splits
    chunk_texts  = [text for text, _ in kept]
    chunk_metas  = [{"page": page} for _, page in kept]
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
        logger.debug("Truncation: %s", truncation_warning)

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
    logger.debug("Loading: %s", pdf_path)
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
        logger.debug("Truncation: %s", truncation_warning)

    retriever = build_hybrid_retriever(pages)
    filename = pdf_path.split("/")[-1]
    chars = sum(len(p["text"]) for p in pages)
    logger.debug("Loaded '%s' — %d chars, %d pages", filename, chars, len(pages))

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
def rewrite_query(
    question: str,
    document_type: str = "technical",
    temperature: float = 0.1,
    recent_context: str = None,
) -> str:
    """
    #3: Rewrites vague/short user questions into more specific,
    search-friendly queries before retrieval.

    [HAL] A tiny amount of creativity (temperature=0.1) in rewriting
    produces slightly varied phrasings, improving retrieval diversity for
    normal single-fact lookups.

    [FIX] Round 6: comprehensive/listing requests use temperature=0
    instead (passed in by the caller) — determinism matters more than
    diversity once the request already implies "give me everything".

    [PERF] Round 7: reuses a cached LLM client (see _get_llm) rather than
    constructing a new one on every call.

    [FIX] Round 7.13: new recent_context parameter — a short string (the
    last user/assistant exchange) that lets the rewriter resolve pronouns
    and elliptical follow-ups BEFORE retrieval happens. Concrete bug this
    fixes: a two-turn conversation — "What is the oil capacity?" (answered
    correctly) followed by "And what's the minimum for that?" — failed,
    because this function never saw any conversation history at all. The
    FINAL answer-generation step does receive chat history (via the
    {chat_history} prompt placeholder), but that's too late: if the
    *retrieval* query is just the literal text "what's the minimum for
    that?" with no idea what "that" refers to, the wrong (or no) content
    gets fetched, and no amount of history available to the generator
    afterward can recover content that was never retrieved in the first
    place. Pronoun/ellipsis resolution has to happen before retrieval, not
    just before generation.
    """
    llm = _get_llm(MODEL, temperature, max_tokens=600)

    history_block = ""
    if recent_context:
        history_block = f"""
Recent conversation (use ONLY to resolve pronouns/references like "that",
"it", "this", or an implied subject in a short follow-up question — do not
just repeat this verbatim, and ignore it if the current question is already
self-contained):
{recent_context}
"""

    rewrite_prompt = f"""
You are an expert RAG query re-writer for technical operator manuals and equipment documents.

Your job is to understand the user's real intent and convert it into the most effective search query.

Rules:
- Capture the core topic accurately
- If the current question is a short follow-up that depends on the recent
  conversation to make sense (e.g. "what's the minimum for that?" right
  after a question about oil capacity), resolve the reference using the
  recent conversation below and rewrite it as a fully self-contained query
  — e.g. "minimum for that" after an oil-capacity question becomes
  "minimum oil capacity lower level indicator fill line", not just "minimum".
- If the question already closely matches an actual section title or specific
  procedure name in this kind of document (e.g. "basic operations" closely
  matching a section literally titled "Basic Operation"), preserve that exact
  phrase as the anchor of the rewrite — do not dilute it with generic words.
  Generic terms like "instructions"/"settings"/"controls"/"operation" appear
  in MANY unrelated sections of a technical manual (e.g. "Filter Control",
  "Special Programming"), so padding a specific query with them can drift
  retrieval toward the wrong section entirely.
- Watch for a specific ambiguity: phrases like "restart", "turn back on",
  "re-energize", or "power back up" can describe either (a) the NORMAL
  start-up procedure (turning the unit on for ordinary use), or (b) ERROR
  RECOVERY (power-cycling the control board after a fault, which an Error
  Codes section repeats for almost every entry — e.g. "turn switch to OFF
  position, then back to ON"). If the question does NOT mention an error
  code, a symptom, or something going wrong, assume it means the NORMAL
  start-up procedure and bias the rewrite toward THAT section's specific
  vocabulary instead of generic power-cycling language, since the generic
  phrasing alone tends to retrieve Error Codes content instead.
- Add technical keywords and synonyms that are SPECIFIC and DISTINCTIVE to
  the topic asked about, not generic ones that could match many sections.
- Make it concise but information-rich
- Do not add any explanation or extra words

Examples:
- "give me the error code table" -> "error codes table E- display cause correction troubleshooting"
- "error code table" -> "error codes E- cause correction display list"
- "show all error codes" -> "error codes full list E- troubleshooting"
- "how to filter the oil" -> "filtering instructions filter envelope clean-out mode"
- "what is the capacity" -> "pot capacity specifications"
- "give me the basic operations" -> "basic operation startup procedure POWER switch frypot oil DROP button load product press start cook cycle"
- "give me all basic operations in detail" -> "basic operation startup procedure POWER switch frypot oil DROP button load product press start cook cycle end of cycle"
- "how do I re-energize the system after a power failure?" -> "start-up procedure AUTO-MELT auto-melt mode set point bar graph Mlt Mix Top Pol main cook menu POWER switch ON position"
- "what do I do if the display shows an error after restarting?" -> "error codes troubleshooting display message correction control board"
- (after a question about oil capacity) "and what's the minimum for that?" -> "minimum oil capacity lower level indicator fill line frypot"
{history_block}
User Question: {question}
Rewritten Query (return ONLY the query):"""

    try:
        result = llm.invoke(rewrite_prompt)
        rewritten = result.content.strip()
        logger.debug("Rewrite: %r -> %r (temp=%s)", question, rewritten, temperature)
        return rewritten if rewritten else question
    except Exception:
        logger.debug("Rewrite failed, using original question: %r", question, exc_info=True)
        return question


# ── #4: Retrieve chunks with source pages ─────────────────────────────────────
def _boost_retriever_k(retriever, new_k: int) -> list:
    """
    [FIX] Round 4: temporarily increases how many chunks each sub-retriever
    returns for a single call (mutating .k / .search_kwargs in place is
    cheap — no re-embedding needed since the FAISS vectorstore is unchanged).

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


def _pad_adjacent_pages(retriever, retrieved_docs: list, pad: int = 1, top_n: int = 3) -> list:
    """
    [FIX] Round 7.3: many sections in a technical manual span 2+ consecutive
    pages (a numbered procedure that continues "(CONT.)" on the next page,
    a table split across a page break, etc). A chunk's individual
    relevance score has no notion of this structural continuity — the
    chunk holding the LATER half of a procedure can rank well on its own
    distinctive action words, while the chunk holding the EARLIER half
    (often just setup steps or an intro sentence) scores lower and gets
    left out, even though it's the very next/previous page of a
    confident hit.

    Concrete bug this fixes: "give me basic operations" retrieved the
    chunk covering steps 4-10 of a 10-step procedure, but never the chunk
    covering steps 1-3 on the page before it — even widening k did not
    help, because that chunk genuinely ranked low for the query, not just
    outside an arbitrary cutoff.

    [FIX] Round 7.4: a page can be split into MULTIPLE chunks (with
    CHUNK_SIZE=1000, a ~1,500-char page like this one's "Basic Operation"
    section becomes 2 chunks). The original version of this function only
    backfilled pages with ZERO chunks already retrieved — so if chunk 2 of
    page 24 (steps ~4-7) was retrieved, page 24 already counted as
    "present" and chunk 1 of that SAME page (the intro + steps 1-3) was
    never backfilled, even though it's a completely different, still-
    missing chunk. Now every chunk within the padded page range is
    considered, and only an EXACT text duplicate is skipped — so a
    partially-retrieved page gets its remaining chunks filled in too, not
    just genuinely new neighboring pages.

    [FIX] Round 7.5: top_n bounds WHICH retrieved chunks get padded.
    retrieved_docs is sorted by descending confidence (EnsembleRetriever
    returns it that way) — only the top_n highest-confidence chunks are
    used to decide which pages to pad around. Bug this fixes: "give me
    all basic operations in detail" widened retrieval to k=20 per
    sub-retriever for the generic word "all", which genuinely matched
    pages scattered across most of the manual's procedural sections
    (Filtering, Clean-Out, Programming, Troubleshooting, Error Codes —
    all share similar step-by-step phrasing). Padding around EVERY one of
    those 20-40 candidate pages turned a 2-page answer into nearly half
    the manual, which then got cut off by the token limit anyway. Padding
    only the top few confident hits keeps the answer scoped to what's
    actually relevant, while a wide k is still free to do its job of
    finding the right content in the first place — recall and padding
    are now decoupled.

    Unlike _expand_to_full_section (which can pull in an unboundedly large
    page range once ANY loose cluster is detected, and is therefore only
    enabled for explicit table/comparison/listing requests), this is
    deliberately narrow: only ±`pad` pages around the top `top_n` most
    confident chunks are added, so the worst case is a small, predictable
    amount of extra context — never a runaway expansion across the document.
    """
    all_chunks = getattr(retriever, "all_chunks", None)
    if not all_chunks:
        return retrieved_docs

    top_docs = retrieved_docs[:top_n]
    pages_present = {doc.metadata.get("page") for doc in top_docs if doc.metadata.get("page")}
    if not pages_present:
        return retrieved_docs

    wanted_pages = {p + delta for p in pages_present for delta in range(-pad, pad + 1)}
    already_have = {doc.page_content for doc in retrieved_docs}

    padded = list(retrieved_docs)
    for text, page in all_chunks:
        if page in wanted_pages and text not in already_have:
            padded.append(_SimpleDoc(text, page))
            already_have.add(text)
    return padded


def _restrict_to_anchor_locality(
    docs: list, anchor_page, max_distance: int = MAX_PAGE_DISTANCE_FROM_ANCHOR
) -> list:
    """
    [FIX] Round 7.6: a final, structural safety ceiling for non-comprehensive
    questions — applied AFTER the top-N cutoff and scoped padding, as a
    last line of defense.

    Concrete bug this fixes: "give me all basic operations in detail"
    still pulled in content from Programming, Troubleshooting, and the
    full Error Codes table (40+ pages away from Basic Operation), even
    after rank-based filtering — because BM25's keyword overlap on
    generic procedural phrasing ("press the button", "follow the steps",
    "check the connection") repeats throughout the manual and can rank a
    handful of distant chunks highly enough to survive a rank cutoff.

    This rule is deliberately NOT rank-based: related content in a
    well-organized technical manual is almost always within a handful of
    pages of the single most confident match (`anchor_page`, the page of
    the very first result the retriever returned, before any cutoff or
    padding touched the list). Anything farther than `max_distance` pages
    from that anchor is dropped outright, regardless of its score —
    closing the loophole that purely rank-based filtering can't.
    """
    if anchor_page is None:
        return docs
    return [
        doc for doc in docs
        if doc.metadata.get("page") is None
        or abs(doc.metadata.get("page") - anchor_page) <= max_distance
    ]


def _cluster_pages(pages: list, max_gap: int = 2) -> list:
    """
    [FIX] Round 5: groups a list of page numbers into clusters where
    consecutive pages are within `max_gap` of each other. Lets us tell the
    difference between "these chunks all belong to one contiguous document
    section" versus "these chunks are scattered across unrelated pages".
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
    just whichever ones happened to rank inside the (already widened) top-k.

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


def retrieve_with_sources(
    retriever, question: str, k_override: int = None, expand_sections: bool = False
) -> tuple:
    """
    #4: Retrieves relevant chunks and extracts source page numbers.

    [FIX] Round 4: k_override temporarily widens retrieval for this single
    call — used for comprehensive/listing requests where the default
    top-k is too narrow to cover a multi-page table. Settings are restored
    immediately after, so normal questions are unaffected.

    [FIX] Round 7.3: every retrieval (not just comprehensive ones) is now
    padded with the immediately adjacent page(s) of whatever got retrieved
    — see _pad_adjacent_pages(). This is intentionally small and bounded
    (±1 page per retrieved page), unlike full section expansion, so it's
    safe to apply universally without reintroducing the Round 7.2
    over-broad-expansion regression.

    When expand_sections=True (strong-signal requests only):
      - _expand_to_full_section() additionally backfills any chunks
        missed by ranking alone, once a genuine multi-page section is
        detected (an unbounded range, unlike the padding above).

    When expand_sections=False (normal questions): after padding,
    _restrict_to_anchor_locality() drops anything farther than
    MAX_PAGE_DISTANCE_FROM_ANCHOR pages from the single highest-ranked
    original hit — a structural ceiling, not a rank-based one, that
    closes the gap rank-based filtering alone couldn't (see that
    function's docstring for the concrete bug this fixes).

    Chunks are always sorted by page number before being joined into
    context — reading in document order (rather than relevance-rank
    order) makes it far easier for the model to recognize that two
    excerpts continue the same numbered procedure or table, instead of
    treating them as unrelated fragments.

    Every chunk is labelled "[PDF Page N]" using the TRUE PDF index before
    being joined into context — the model is instructed to cite using
    this label rather than any page number printed inside the document's
    own header/footer text.

    Returns (combined_context_text, sorted_unique_page_numbers).
    """
    restores = _boost_retriever_k(retriever, k_override) if k_override else []
    try:
        docs = retriever.invoke(question)
        anchor_page = docs[0].metadata.get("page") if docs else None

        if not expand_sections:
            # [FIX] Round 7.5: cap to genuinely top-ranked chunks for normal
            # questions — see DEFAULT_FINAL_TOP_N. Comprehensive/expand_sections
            # requests intentionally skip this cap.
            docs = docs[:DEFAULT_FINAL_TOP_N]

        docs = _pad_adjacent_pages(retriever, docs, pad=1)

        if expand_sections:
            docs = _expand_to_full_section(retriever, docs)
        else:
            # [FIX] Round 7.6: structural ceiling, see _restrict_to_anchor_locality.
            docs = _restrict_to_anchor_locality(docs, anchor_page)

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


# ── [FIX] Round 7: Universal recall fallback ───────────────────────────────────
def _retrieve_with_recall_fallback(
    retriever, search_query: str, raw_question: str,
    base_k_override: int = None, base_expand_sections: bool = False,
) -> tuple:
    """
    [FIX] Round 7: if a topic genuinely exists in the document, a question
    about it should get an answer no matter how the question happens to be
    phrased. A single retrieval pass occasionally comes back empty or
    razor-thin — e.g. the rewritten query drifted from the source's exact
    terminology — and previously that went straight to "not available"
    with no second attempt.

    Escalates through up to three retrieval attempts, stopping as soon as
    one clears a minimal usefulness bar (some source pages AND a non-trivial
    amount of text):
      1. Normal retrieval with the given (rewritten/cleaned) search_query
         and whatever k_override/expand_sections the caller already
         decided on.
      2. The SAME search_query with a much wider net — at this point the
         baseline already came back thin, so casting wider is worth the
         risk of some extra noise. Crucially, this does NOT force on full
         section expansion for a normal question: expand_sections stays
         whatever the ORIGINAL caller intended (base_expand_sections).
         [FIX] Round 7.6: this used to hardcode expand_sections=True for
         every escalation, regardless of the original request — which
         meant a normal question whose first attempt landed at, say, 199
         characters (one under the threshold) would escalate straight
         into the unbounded full-section-expansion path, bypassing the
         structural locality ceiling entirely (see
         _restrict_to_anchor_locality). Only a genuine full_expansion
         request (table/comparison/error-codes) should ever get the
         unbounded treatment.
      3. The user's RAW original question — bypassing rewriting and
         format-keyword-stripping entirely — in case either step
         accidentally dropped the one term that actually mattered, again
         with the wider net (same expand_sections behavior as step 2).

    If all three attempts come back thin, returns whichever attempt found
    the most content, so the model still gets the best available context
    to judge from honestly rather than the least.
    """
    attempts = []

    pdf_content, source_pages = retrieve_with_sources(
        retriever, search_query, k_override=base_k_override, expand_sections=base_expand_sections
    )
    attempts.append((pdf_content, source_pages))
    if source_pages and len(pdf_content.strip()) >= MIN_USEFUL_RETRIEVAL_CHARS:
        return pdf_content, source_pages

    logger.debug(
        "Retrieval thin (%d pages, %d chars) for %r — widening the net",
        len(source_pages), len(pdf_content), search_query,
    )
    pdf_content, source_pages = retrieve_with_sources(
        retriever, search_query, k_override=COMPREHENSIVE_TOP_K_CHUNKS, expand_sections=base_expand_sections
    )
    attempts.append((pdf_content, source_pages))
    if source_pages and len(pdf_content.strip()) >= MIN_USEFUL_RETRIEVAL_CHARS:
        return pdf_content, source_pages

    if raw_question.strip().lower() != search_query.strip().lower():
        logger.debug("Still thin — retrying with the raw, un-rewritten question")
        pdf_content, source_pages = retrieve_with_sources(
            retriever, raw_question, k_override=COMPREHENSIVE_TOP_K_CHUNKS, expand_sections=base_expand_sections
        )
        attempts.append((pdf_content, source_pages))
        if source_pages and len(pdf_content.strip()) >= MIN_USEFUL_RETRIEVAL_CHARS:
            return pdf_content, source_pages

    return max(attempts, key=lambda pair: len(pair[0]))


# ── [FIX] Round 4/7.2: Tiered comprehensive/listing request detection ─────────
_TABLE_KEYWORDS = ["table", "tabular", "tabulate", "rows and columns"]
_COMPARE_KEYWORDS = [
    "difference between", "differences between", "differ from",
    "compare", "comparison", " vs ", " vs.", " versus ",
    "contrast between", "similarities and differences",
]
_COMPLETENESS_KEYWORDS = [
    "all ", "every ", "complete", "entire", "full list", "everything",
    "comprehensive", "list all",
]


def _wants_full_section_expansion(question: str) -> bool:
    """
    [FIX] Round 7.2: STRONG signal only — an explicit table/comparison
    request, or this PDF's error-codes special case. This is the tier that
    warrants full cluster expansion (_expand_to_full_section), which
    unconditionally backfills an entire contiguous page range. That's the
    right call for "give me the error codes table" (a genuine multi-page
    table), but it is too blunt an instrument to fire on generic phrasing.

    [FIX] Round 7.10: the error-code special case used to be a bare "e-"
    substring check, which matched ANY word containing that two-character
    sequence — including "re-energize", "re-engage", "re-establish", etc.
    Confirmed bug: "How do I re-energize the system after a power failure?"
    triggered this path, silently switching to COMPREHENSIVE_TOP_K_CHUNKS
    and bypassing the anchor-locality safety net (see retrieve_with_sources
    — anchor-locality only applies when expand_sections=False), with
    nothing in place to stop scope creep if the wide net happened to
    cluster badly. It didn't visibly misfire in that specific test run,
    but the bug was real and the safety net was genuinely disabled by
    accident — not something to leave in place because it got lucky once.
    Now requires the "e" to actually be followed by a digit (with an
    optional hyphen in between), matching real error-code mentions like
    "E-4", "e4", "E-10A" but not ordinary English words.
    """
    q_lower = question.lower().strip()
    if "error code" in q_lower or re.search(r"\be-?\d", q_lower):
        return True
    q = f" {q_lower} "
    return any(kw in q for kw in _TABLE_KEYWORDS + _COMPARE_KEYWORDS)


def _wants_broader_recall(question: str) -> bool:
    """
    [FIX] Round 7.2: WEAK signal — generic completeness words like "all",
    "every", "complete" on their own. These often turn out to be casual
    phrasing ("give me all the basic operations" just means "tell me about
    basic operations", not "this spans many pages") rather than a genuine
    multi-page-table request.

    Bug this fixes: "give me all basic operations" used to trigger the
    SAME full cluster expansion as an explicit table request, which
    backfilled an entire nearby page range and pulled in unrelated
    content (e.g. the Error Codes section) alongside the actually-relevant
    Basic Operation steps. Now this tier only earns a modest k boost
    (MODERATE_TOP_K_CHUNKS) — more candidates considered, but no blind
    whole-section backfill.
    """
    if _wants_full_section_expansion(question):
        return True
    q = f" {question.lower().strip()} "
    return any(kw in q for kw in _COMPLETENESS_KEYWORDS)


def _wants_comprehensive_listing(question: str) -> bool:
    """Backward-compatible alias — True for either tier. Used only where a
    single yes/no signal is needed (e.g. choosing the rewrite temperature)."""
    return _wants_broader_recall(question)


# ── #6: [FMT] Format-aware response detection ─────────────────────────────────
def detect_response_format(question: str) -> str:
    """
    [FMT] Inspects the user's question for an explicit formatting request
    (table / comparison-difference / paragraph) and returns an instruction
    string to inject into the prompt, telling the model how to format the
    content of the JSON "Response" field for this answer.

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

    if is_compare:
        return (
            "FORMAT REQUIREMENT: The user is asking for a COMPARISON or "
            "DIFFERENCE between two or more items. Inside the Response "
            "field, present the answer as a single Markdown table with the "
            "items being compared as columns and the comparison aspects as "
            "rows (e.g. `| Aspect | Item A | Item B |`), so each difference "
            "is clearly partitioned and easy to scan side-by-side. Only "
            "fall back to a clearly labelled bullet list per item if a "
            "tabular layout genuinely does not fit. This requirement does "
            "NOT override the partial-information rule: build the best "
            "possible comparison from whatever is available rather than "
            "declining — only state the information is unavailable if it "
            "is truly absent from the retrieved content."
        )

    if is_table:
        return (
            "FORMAT REQUIREMENT: The user explicitly asked for a TABLE. "
            "Inside the Response field, you MUST present the answer using "
            "a single, well-structured Markdown table with clear column "
            "headers (e.g. `| Column A | Column B |`). Do not substitute "
            "bullet points or prose for the table. This requirement does "
            "NOT override the partial-information rule: if the relevant "
            "content is fragmented, still build the best possible table "
            "from what is available — only state the information is "
            "unavailable if it is truly absent, not merely because the "
            "table is imperfect."
        )

    if is_paragraph:
        return (
            "FORMAT REQUIREMENT: The user explicitly asked for a "
            "PARAGRAPH-style answer. Inside the Response field, respond in "
            "flowing prose paragraphs. Do NOT use bullet points, numbered "
            "lists, or tables for this answer."
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
    Step 5: Builds the ChatPromptTemplate.

    [PROMPT] Round 7: rebuilt around the user-supplied JSON contract — every
    answer is one of four shapes (Relevant / Vague-Incomplete / Non-Relevant
    / Greeting). All grounding rules accumulated through Rounds 1-6
    (strict-context-only, partial-info vs not-available mutual exclusivity,
    page citation via the "[PDF Page N]" label, no "(Same as X)" shorthand,
    FORMAT REQUIREMENT decoupling) now apply specifically to how the
    Response field is composed for Relevant Queries — they're folded in
    rather than dropped, since none of them conflict with the JSON shape.
    """

    system_prompt = (
        "You are an advanced AI assistant responsible for providing responses to "
        "user questions based ONLY on the given PDF context. Never use external "
        "knowledge or information not present in the provided context — if "
        "something isn't in the context, treat it as not available. Think step "
        "by step before answering, and generate all response content in Markdown "
        "format. "

        "Classify every question into exactly one of these four categories, and "
        "reply with a single raw JSON object — no markdown code fences, no text "
        "before or after the JSON — in the exact shape shown for that category:\n\n"

        "1. RELEVANT QUERY: the question relates to the document and the context "
        "below contains relevant information, even if only partial. Determine a "
        "concise, Title Case Tag that categorizes the question (for example: "
        "\"Warranty\", \"Installation\", \"General Inquiry\", \"Troubleshooting\", "
        "\"Error Codes\", \"Cleaning Instructions\", \"Unpacking Instructions\", "
        "\"Specifications\"). Provide a detailed, professional Response built "
        "strictly from the context — e.g. if asked for a table of contents, error "
        "codes, or any other listing, reproduce it in full detail rather than "
        "summarizing it away. Shape: "
        '{{"Tag": "<Title Case category>", "Response": "<detailed Markdown answer>"}}\n\n'

        "2. INCOMPLETE OR VAGUE QUERY: the question is unclear, too short, or "
        "missing detail needed to answer from the document. Shape: "
        '{{"Response": "<a clarification request, with guidance on how to ask a '
        'more specific question>"}}\n\n'

        "3. NON-RELEVANT QUERY: the question doesn't relate to the document's "
        "subject matter, even if some context text was retrieved alongside it. "
        "Shape: "
        '{{"Response": "<a brief, honest reply noting this isn\'t covered by the '
        'document, plus relevant follow-up questions>"}}\n\n'

        "4. GENERAL GREETING: simple greetings like \"Hi\" or \"Hello\" with no "
        "real question. Shape: "
        '{{"Response": "<a polite greeting, plus a few example questions the user '
        'could ask>"}}\n\n'

        "Any follow-up questions you suggest (categories 3 and 4) must be "
        "questions the provided PDF context can actually answer — never invent "
        "generic questions unrelated to this specific document. "

        "Rules that apply specifically to RELEVANT QUERY responses: "
        "If you found ANY relevant information, even if incomplete or scattered "
        "across multiple pages, share what you found in the Response and clearly "
        "state what specific details are missing — never claim nothing was found "
        "if you actually have partial information; stating 'not available' and "
        "giving a partial answer are mutually exclusive within the same Response. "
        "Always cite page numbers, e.g. \"(Page 4)\", using the page number given "
        "in the \"[PDF Page N]\" label right before each excerpt in the context — "
        "never a page number printed in the document's own header or footer text, "
        "since that internal numbering frequently differs from the page's true "
        "position in the PDF file. "
        "If you are less than fully confident about a specific detail, prefix that "
        "part with **[Low Confidence]** and say what you're unsure about — but "
        "scope this to the SPECIFIC missing or uncertain piece only. In a table "
        "row, if one column's value is genuinely absent from the source (e.g. no "
        "distinct Cause is given) but another column's value IS clearly present "
        "(e.g. the Correction text is right there), still include what's "
        "available — never blank out an entire row as \"not available\" just "
        "because one of several columns is thin or missing; say only that "
        "specific column wasn't specified, while still reporting the rest. "

        "COMPLETENESS: if the context contains a list or table that is DIRECTLY "
        "about what the user asked, the Response must include all of its items — "
        "never silently stop partway through or skip an item to save space, even "
        "for a long table. If everything genuinely can't fit, say so explicitly "
        "rather than quietly omitting items. This applies ONLY to content that "
        "answers the actual question — it does not mean enumerating every section "
        "title, list, or table that merely happens to appear nearby in the "
        "context. "

        "STAY ON TOPIC: some excerpts may be a Table of Contents / section index "
        "— a list of section TITLES with page numbers and little else. Such an "
        "excerpt only tells you where to look; never use it as a basis for "
        "summarizing, listing, or writing filler like \"for information on X, "
        "see the X section\" about sections the user did not ask about, even if "
        "their titles appear in the context. The context may also contain "
        "complete, substantive excerpts from OTHER sections that aren't a TOC "
        "(e.g. Filtering, Programming, Error Codes alongside a question about "
        "Basic Operation) because they happened to share some wording with the "
        "question — when that happens, address ONLY the section(s) the "
        "question is actually about and ignore the rest entirely, even though "
        "real content is available for them. Answer only the topic actually "
        "asked about, and never repeat the same heading, list, or paragraph more "
        "than once within a single Response. "

        "NUMBERING: when the source presents a sequence of ordered steps where "
        "doing them in order matters, reproduce them as ONE numbered Markdown "
        "list (1. 2. 3. ...) in that same order — never convert ordered steps "
        "into unordered bullets, and never renumber or reorder them. The "
        "context below may contain the same procedure split across multiple "
        "excerpts (e.g. one excerpt ends at step 7 and a later excerpt, "
        "possibly marked \"(CONT.)\", begins at step 8) — when that happens, "
        "treat them as ONE continuous procedure and continue the numbering "
        "from the source's own step numbers; never restart a later portion at "
        "1 or present it as a separate \"Additional Steps\" list. Use bullet "
        "points only for items that have no inherent order. "

        "TABLE COLUMN FIDELITY: when the source is laid out as parallel columns "
        "(for example a Display/Cause/Correction error table), preserve that "
        "structure as a Markdown table by default, even without an explicit "
        "request for a table. Identify EVERY distinct column before writing any "
        "row, and map each value to the SAME column it came from — never shift "
        "values over by one column, merge two columns into one, or drop a "
        "column's content (most often the longest one, e.g. the correction/"
        "action text) just because it is long. Example: the source row "
        '"E-4" "CPU TOO HOT" Control board overheating Turn switch to OFF '
        "position, then back to ON; if display still shows E04, replace the "
        'control. — has exactly THREE columns: Display = \'"E-4" "CPU TOO '
        "HOT\"', Cause = 'Control board overheating', Correction = 'Turn switch "
        "to OFF position, then back to ON; if display still shows E04, replace "
        "the control.' — the short phrase right after the quoted display text "
        "is the Cause; everything after that is the Correction. Never collapse "
        "Cause and Correction into one cell or omit the Correction text. "
        "EVERY row must have EXACTLY the same number of `|`-separated cells as "
        "the header, with no exceptions — a row with fewer cells than the "
        "header (e.g. only 2 cells in a 3-column table) renders with every "
        "value shifted into the wrong column. If a particular row's source "
        "entry is missing a value for one column entirely (not merely thin, "
        "but truly absent — e.g. an error code with no separate Cause beyond "
        "its code, only a Correction), still write that row with the SAME "
        "column count, putting a placeholder like \"Not specified\" in the "
        "empty cell rather than skipping it. "
        "PAIRED VARIANTS: when two adjacent rows are clearly two variants of "
        "the same underlying issue (e.g. one row says \"(Open Circuit)\" and "
        "the very next says \"(Shorted)\" for the same component, or similar "
        "obviously-paired labels), the raw extracted text sometimes interleaves "
        "their bullet points out of order — e.g. \"replace probe\" or a second "
        "row's label appearing in the middle of the first row's text. Before "
        "concluding a value is \"Not specified\", check whether the surrounding "
        "text for the PAIRED row contains content that was clearly meant for "
        "this row but got extracted out of sequence, and assign it to the "
        "correct row. Only fall back to \"Not specified\" if no such "
        "content exists anywhere nearby for that specific row. "
        "When listing multiple similar "
        "rows, write out the full text for EVERY row — never use a shorthand "
        "like '(Same as Error Code X)' to skip repeating text, since similar-"
        "looking rows can differ in ways that matter and shorthand references "
        "have caused incorrect groupings before. "

        "If a FORMAT REQUIREMENT is given below, it overrides your default "
        "formatting judgement for that Response — follow it exactly. A FORMAT "
        "REQUIREMENT is a presentation instruction ONLY: it never changes whether "
        "information is available. Decide whether the context contains the answer "
        "first; only then apply the requested format to whatever you found."
    )

    human_prompt = """{format_instruction}

{chat_history}

Context:
---
{pdf_content}
---

Question: {user_question}

Answer (a single raw JSON object only, matching exactly one of the four shapes above — no markdown code fences, no commentary before or after it):"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template(human_prompt),
    ])

    return prompt


# ── [FIX] Round 7: Structured JSON output parsing ─────────────────────────────
def _strip_code_fences(text: str) -> str:
    """Removes a ```json ... ``` or ``` ... ``` wrapper if the model added
    one despite being told not to."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


_RESPONSE_FIELD_RE = re.compile(r'"Response"\s*:\s*"(.*)"\s*\}\s*$', re.DOTALL)
_TAG_FIELD_RE = re.compile(r'"Tag"\s*:\s*"([^"]*)"')


def _unescape_json_string(s: str) -> str:
    """Best-effort unescape of common JSON string escapes — tolerant of
    literal (unescaped) newlines/quotes a model sometimes leaves in despite
    instructions, which strict json.loads() would reject outright."""
    s = s.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    s = s.replace('\\"', '"')
    s = s.replace("\\\\", "\\")
    return s


def _parse_json_answer(raw_text: str) -> dict:
    """
    [FIX] Round 7: parses the model's JSON-shaped output with three
    fallback levels, so a malformed or imperfectly-escaped response never
    crashes the app or produces a blank answer:

      1. Strict json.loads() — works whenever the model followed the
         schema correctly (most of the time, especially with native JSON
         mode enabled — see _get_llm).
      2. Tolerant regex extraction — recovers "Tag"/"Response" even when
         the model left a literal unescaped newline or quote inside the
         Response string, which breaks strict JSON parsing but is
         trivially recoverable for this known, fixed shape.
      3. Raw passthrough — if the output isn't JSON-shaped at all, treat
         the entire raw text as the answer rather than showing nothing.

    Returns a dict with a "response" key (str), and a "tag" key (str)
    when one was present.
    """
    text = _strip_code_fences(raw_text).strip()

    # Level 1: strict parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "Response" in parsed:
            result = {"response": str(parsed["Response"])}
            if parsed.get("Tag"):
                result["tag"] = str(parsed["Tag"])
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    # Level 2: tolerant regex extraction
    match = _RESPONSE_FIELD_RE.search(text)
    if match:
        result = {"response": _unescape_json_string(match.group(1))}
        tag_match = _TAG_FIELD_RE.search(text)
        if tag_match:
            result["tag"] = tag_match.group(1)
        return result

    # Level 3: raw passthrough — never show a blank answer
    logger.debug("JSON parse fully failed, falling back to raw text passthrough")
    return {"response": text}


def _render_final_answer(parsed: dict) -> str:
    """
    Builds the final Markdown text shown to the user: a small category
    line when a Tag was returned (Relevant Queries only), followed by the
    Response content. Vague/Non-Relevant/Greeting answers have no Tag and
    are returned exactly as their Response text.
    """
    response = parsed.get("response", "").strip()
    tag = parsed.get("tag")
    if tag:
        return f"**Category:** {tag}\n\n{response}"
    return response


# ── [FIX] Round 7.11: post-generation retry when retrieval found the wrong topic ──
_NOT_FOUND_PHRASES = [
    "does not contain", "not covered in the document", "not covered in this document",
    "not available in the document", "context does not contain",
    "context provided does not contain", "is not covered", "no information about",
    "not mentioned in the document", "not contain information about",
]


def _looks_like_not_found(parsed: dict) -> bool:
    """
    [FIX] Round 7.11: True when an answer reads like the model couldn't
    actually locate the requested content in the retrieved context.

    Concrete bug this targets: "Clean-Out Mode" content exists in the
    document, but a retrieval pass anchored on the wrong nearby section
    (Drain Pan Assembly / Display Options, which share some "drain"
    vocabulary) and the model reported "the provided context does not
    contain information about the Clean-Out Mode process" — a retrieval
    miss, not a hallucination, but one the existing recall fallback
    couldn't catch because it only measures whether retrieval came back
    *thin* (few chars/pages), not whether it came back about the *wrong
    topic*. A non-thin-but-wrong-topic result and a genuinely thorough
    "not found" both look identical from a pure character-count
    perspective — this checks the model's own verdict instead.

    [FIX] Round 7.12: dropped the requirement that a Tag be present. A
    question that's genuinely relevant to the document (Clean-Out Mode IS
    a real section here) but retrieved the wrong section's content can get
    misclassified as a NON-RELEVANT QUERY — which has no Tag by schema —
    rather than a Relevant Query with missing info, since from the model's
    point of view the retrieved excerpts (about Drain Pan Assembly) didn't
    seem to relate to "Clean-Out Mode" at all. Gating on Tag let that exact
    case slip through with no retry. The phrase match alone is the real
    signal; a genuinely off-topic question (e.g. "what's the weather")
    triggering one extra, ultimately-discarded retry call is an acceptable,
    bounded cost — the retry will just come back "not found" again and the
    original honest answer is kept (see ask_question/ask_question_stream).
    """
    response_lower = parsed.get("response", "").lower()
    return any(phrase in response_lower for phrase in _NOT_FOUND_PHRASES)


def _prepare_escalated_retry_inputs(
    retriever, user_question: str, document_type: str
) -> tuple:
    """
    [FIX] Round 7.11: builds chain inputs for a single retry pass after the
    first attempt's answer looked like a retrieval miss (see
    _looks_like_not_found). Deliberately forces the widest, least-clever
    settings — bypassing whatever rewriting/tiering choice led the first
    attempt to anchor on the wrong section:
      - the user's RAW question, with no rewriting and no format-keyword
        stripping, in case either step drifted away from the document's
        own terminology;
      - COMPREHENSIVE_TOP_K_CHUNKS + full section expansion, so a genuine
        multi-page section gets every chunk in its range rather than
        whatever a narrower pass happened to rank highest;
      - conversation history dropped, to maximize the token budget
        available for content on this one bounded retry.
    This only ever runs once per question (no further escalation beyond
    this), so the worst-case cost is one extra retrieval + generation call,
    paid only when the first pass actually needed it.
    """
    pdf_content, source_pages = retrieve_with_sources(
        retriever, user_question, k_override=COMPREHENSIVE_TOP_K_CHUNKS, expand_sections=True
    )
    chunk_budget = CHUNK_TOKEN_BUDGET_COMPREHENSIVE
    chunk_tokens = estimate_token_count(pdf_content)
    if chunk_tokens > chunk_budget:
        pdf_content = pdf_content[: chunk_budget * 4]

    chain_inputs = {
        "pdf_content": pdf_content,
        "user_question": user_question,
        "chat_history": "",
        "format_instruction": detect_response_format(user_question),
    }
    return chain_inputs, source_pages


# ── STEP 6: Build chain ───────────────────────────────────────────────────────
def build_chain():
    """
    Step 6: Prompt -> Mistral -> StrOutputParser.

    temperature=0 on the answer LLM for maximum factual accuracy and the
    most reliable JSON-schema compliance.

    [FIX] Round 7: requests native JSON-object mode where supported (see
    _get_llm) so the API itself enforces syntactically valid JSON, on top
    of the prompt instructions and the tolerant parser fallback.

    [STREAM] This same chain supports both .invoke() (blocking, used by
    ask_question) and .stream() (used by ask_question_stream).
    """
    llm = _get_llm(MODEL, temperature=0, max_tokens=MAX_TOKENS, json_mode=True)
    prompt = build_langchain_prompt()
    parser = StrOutputParser()
    return prompt | llm | parser


# ── #2: Memory helpers ────────────────────────────────────────────────────────
def format_chat_history(chat_history: list, last_k: int = MEMORY_LAST_K) -> str:
    """
    #2: Formats last K chat exchanges into a string for the prompt.
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

      #3      -> Rewrite query (domain-aware, deterministic for comprehensive)
      #1+#5   -> Retrieve relevant chunks via hybrid search, with the
                 [FIX] Round 7 recall-fallback escalation if the first pass
                 comes back thin
      #4      -> Extract source page numbers
      [CTX]   -> Budget check: trim chunks and/or history if over limit
      #2      -> Format conversation history (dynamic last_k)
      [FMT]   -> Detect requested response format (table / comparison / paragraph)
    """
    # [FMT] Strip format-instruction words before retrieval — see
    # _strip_format_keywords docstring for the concrete bug this avoids.
    retrieval_question = _strip_format_keywords(user_question)

    # [FIX] Round 7.13: a short, lightweight slice of the LAST exchange only
    # (not the full MEMORY_LAST_K history used for the final answer) — just
    # enough for rewrite_query() to resolve a pronoun/ellipsis in a terse
    # follow-up like "what's the minimum for that?" before retrieval runs.
    recent_context = None
    if chat_history:
        last_pair = chat_history[-2:]
        if last_pair:
            recent_context = "\n".join(
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in last_pair
            )

    # [FIX] Round 7.2: tiered intent detection — see _wants_full_section_expansion
    # / _wants_broader_recall docstrings. Computed once, reused for the rewrite
    # temperature and the retrieval-width decisions below.
    full_expansion = _wants_full_section_expansion(user_question)
    broader_recall = full_expansion or _wants_broader_recall(user_question)
    rewrite_temperature = 0.0 if broader_recall else 0.1

    search_query = rewrite_query(
        retrieval_question,
        document_type=document_type,
        temperature=rewrite_temperature,
        recent_context=recent_context,
    )
    logger.debug(
        "Question=%r Rewritten=%r FullExpansion=%s BroaderRecall=%s",
        user_question, search_query, full_expansion, broader_recall,
    )

    # [FIX] Round 7.5: dropped the MODERATE_TOP_K_CHUNKS boost for the
    # broader_recall-but-not-full_expansion tier. Bug: widening k to 20 for
    # generic completeness wording ("all", "every") genuinely matched pages
    # scattered across most of a procedural manual (every section uses
    # similar step-by-step phrasing), turning a 2-page answer into nearly
    # half the document. The default top-k is already enough to surface
    # genuinely relevant chunks for a reasonably specific question; the
    # top-N-scoped page padding (see _pad_adjacent_pages) and the recall
    # fallback below remain as universal safety nets regardless of this
    # tier, so completeness for a real multi-page SECTION (not a sprawling
    # match across unrelated sections) is still covered without needing a
    # wider net here.
    if full_expansion:
        retrieval_k, chunk_budget = COMPREHENSIVE_TOP_K_CHUNKS, CHUNK_TOKEN_BUDGET_COMPREHENSIVE
    else:
        retrieval_k, chunk_budget = None, CHUNK_TOKEN_BUDGET

    # [FIX] Round 7: never settle for a single thin retrieval pass — escalate
    # automatically before ever telling the model (and the user) "not found".
    pdf_content, source_pages = _retrieve_with_recall_fallback(
        retriever, search_query, user_question,
        base_k_override=retrieval_k, base_expand_sections=full_expansion,
    )

    # ── [CTX] Step 1: Cap chunk tokens ────────────────────────────────────────
    chunk_tokens = estimate_token_count(pdf_content)
    if chunk_tokens > chunk_budget:
        max_chunk_chars = chunk_budget * 4
        pdf_content = pdf_content[:max_chunk_chars]
        logger.debug("Chunks trimmed: %d -> %d tokens", chunk_tokens, chunk_budget)

    # ── [CTX] Step 2: Dynamic history trim ────────────────────────────────────
    history_list = chat_history or []
    effective_last_k = MEMORY_LAST_K

    full_history_text  = format_chat_history(history_list, last_k=MEMORY_LAST_K)
    short_history_text = format_chat_history(history_list, last_k=1)

    chunk_tok   = estimate_token_count(pdf_content)
    history_tok = estimate_token_count(full_history_text)
    # [FIX] Round 7.9: 2000 -> 2200. The system + human prompt scaffolding
    # now measures ~2,140 tokens after the paired-variant (Open Circuit/
    # Shorted style) extraction-order guidance was folded in.
    system_tok  = 2200

    total_estimated = chunk_tok + history_tok + system_tok

    if total_estimated > CONTEXT_TOKEN_BUDGET:
        short_tok = estimate_token_count(short_history_text)
        if chunk_tok + short_tok + system_tok <= CONTEXT_TOKEN_BUDGET:
            effective_last_k = 1
            logger.debug("History trimmed to 1 exchange (budget %d > %d)", total_estimated, CONTEXT_TOKEN_BUDGET)
        else:
            effective_last_k = 0
            logger.debug("History dropped entirely (budget %d > %d)", total_estimated, CONTEXT_TOKEN_BUDGET)

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


# ── [CTX] Token estimation helper ────────────────────────────────────────────
def estimate_token_count(text: str) -> int:
    """Rough token estimator: 1 token ≈ 4 characters."""
    return max(1, len(text) // 4)


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
    _prepare_pipeline_inputs(), invokes the chain once, parses the
    structured JSON the model returned, and returns the rendered answer.

    [FIX] Round 7.11: if the first pass's answer looks like a retrieval
    miss (see _looks_like_not_found), automatically retries once with
    escalated, rewriting-bypassing retrieval settings (see
    _prepare_escalated_retry_inputs) before giving up — see that
    function's docstring for the concrete bug this fixes.

    Returns:
        (answer_string, [source_page_numbers])
    """
    if retriever is None:
        return "Please load a PDF first.", []

    chain_inputs, source_pages = _prepare_pipeline_inputs(
        retriever, user_question, chat_history, document_type
    )

    raw_output = chain.invoke(chain_inputs)
    parsed = _parse_json_answer(raw_output)

    if _looks_like_not_found(parsed):
        retry_inputs, retry_pages = _prepare_escalated_retry_inputs(
            retriever, user_question, document_type
        )
        retry_raw = chain.invoke(retry_inputs)
        retry_parsed = _parse_json_answer(retry_raw)
        if not _looks_like_not_found(retry_parsed):
            parsed, source_pages = retry_parsed, retry_pages

    answer = _render_final_answer(parsed)
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
    [STREAM] Streaming counterpart to ask_question(). Runs retrieval,
    generation, and (if needed) the Round 7.11 retry-on-not-found pass
    before returning:

        (source_pages, answer_chunk_generator)

    `source_pages` reflects whichever pass's answer is actually being
    shown — if a retry occurred, it's the retry's pages, not the first
    pass's. The returned generator is a pure "reveal" animation over
    already-finalized text; no further LLM calls happen once this function
    returns.

    [FIX] Round 7 — streaming design note: the generator below buffers the
    full raw JSON response from chain.stream() internally before yielding
    anything, then reveals the GUARANTEED-correct, fully-parsed final text
    in small increments for a smooth "typing" animation, rather than
    live-streaming raw partial JSON characters.

    This was a deliberate choice, not an oversight: the UI consuming this
    generator can only ever APPEND each yielded chunk to what it has
    already shown (it has no way to retract or correct earlier text). Raw
    JSON streamed mid-flight is frequently mid-escape-sequence, mid-Tag-
    field, or about to be followed by closing `"}` syntax — any of which
    would have to "leak" into the visible answer or be guessed at
    approximately. A guess that's later proven wrong could no longer be
    un-shown, and would permanently corrupt the stored chat history (it's
    re-rendered from that accumulated text on every future page rerun).
    Buffering costs nothing in practice — the API call itself still
    streams under the hood, and total wall-clock time to a finished answer
    is unchanged; only the in-between display behavior is different.

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

    raw_output = "".join(chain.stream(chain_inputs))
    parsed = _parse_json_answer(raw_output)

    # [FIX] Round 7.11: retry-on-not-found (see _looks_like_not_found /
    # _prepare_escalated_retry_inputs) runs HERE — synchronously, before
    # returning — rather than lazily inside the generator. This is what
    # makes it safe: source_pages below is decided only once, after we
    # already know which pass's answer will actually be shown, instead of
    # being locked in to the first pass's pages before a later retry could
    # change them. The user-visible latency characteristic is unchanged
    # from before this fix — the generator already buffered the entire
    # first pass before revealing anything, so doing the retry check at
    # this point (rather than after returning) costs nothing extra except
    # the bounded, rare cost of the retry call itself.
    if _looks_like_not_found(parsed):
        retry_inputs, retry_pages = _prepare_escalated_retry_inputs(
            retriever, user_question, document_type
        )
        retry_raw = "".join(chain.stream(retry_inputs))
        retry_parsed = _parse_json_answer(retry_raw)
        if not _looks_like_not_found(retry_parsed):
            parsed, source_pages = retry_parsed, retry_pages

    final_text = _render_final_answer(parsed)

    def _answer_generator():
        # Reveal the already-final, validated text in small increments —
        # purely a pacing/animation choice, not a guess about content that
        # could later change (see design note above).
        reveal_chunk_size = 24
        for i in range(0, len(final_text), reveal_chunk_size):
            yield final_text[i:i + reveal_chunk_size]

    return source_pages, _answer_generator()


# ── CLI Entry Point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "document.pdf"
    doc_type = sys.argv[2] if len(sys.argv) > 2 else "technical"

    result = load_pdf(pdf_path)
    pdf_store.update(result)

    if pdf_store.get("truncation_warning"):
        print(f"\n⚠  {pdf_store['truncation_warning']}\n")

    chain = build_chain()
    cli_history = []

    print("=" * 55)
    print(" PDF Chatbot ready! (v7 — Structured JSON + Recall Fallback)")
    print(f"   Model         : {MODEL}")
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