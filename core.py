import os
import re
import json
from typing import List, Dict, Tuple, Optional
from collections import Counter

import pdfplumber
import tiktoken
import numpy as np
from scipy.spatial.distance import cosine

# Gemini (Google GenAI SDK)
# pip install -U google-genai
from google import genai
from google.genai import types

from config import (
    # Step 3
    INPUT_PDF_DIR,
    OUTPUT_CHUNKS_PATH,
    TOKEN_CHUNK_SIZE,
    TOKEN_CHUNK_OVERLAP,
    TOKEN_ENCODING_NAME,
    MIN_CHUNK_CHARS,
    # Step 4
    EMBEDDING_MODEL,
    RETRIEVAL_TOP_K,
    # Step 5
    CHAT_MODEL,
    RAG_TOP_K,
    MAX_CONTEXT_CHARS,
    RAG_DISTANCE_THRESHOLD,
    # Step 6
    INTENT_MODEL,
    ENABLE_INTENT_ROUTER,
)

# =========================
# Shared & Local Vector Store
# =========================

_PAGE_TAG_RE = re.compile(r"\[PAGE\s+(\d+)\]", re.IGNORECASE)
VECTOR_STORE_PATH = "storage/vector_store.json"  # 本地向量存储路径


# =========================
# Gemini client (keep old function name for compatibility)
# =========================

def _openai_client(api_key: Optional[str] = None) -> genai.Client:
    """
    Compatibility shim: keep the old name `_openai_client` so other modules (app.py/prod_*.py)
    don't need to change. Internally returns a Gemini client.

    The google-genai SDK will pick up GEMINI_API_KEY or GOOGLE_API_KEY automatically
    if api_key is None. :contentReference[oaicite:3]{index=3}
    """
    if api_key:
        return genai.Client(api_key=api_key)
    return genai.Client()


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove excessive blank lines."""
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_line(line: str) -> str:
    """Normalize a line for robust header/footer repetition detection."""
    line = (line or "").strip()
    line = re.sub(r"\bpage\s*\d+\b", "page", line, flags=re.IGNORECASE)
    line = re.sub(r"\b\d+\s*/\s*\d+\b", "x/y", line)
    line = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "date", line)
    line = re.sub(r"\b\d+\b", "num", line)
    line = re.sub(r"\s+", " ", line)
    return line.lower().strip()


# =========================
# Sources-used enforcement
# =========================

def _format_sources_used(n_sources: int) -> str:
    if n_sources <= 0:
        return "[]"
    return "[" + ", ".join(str(i) for i in range(n_sources)) + "]"


def _enforce_sources_used_line(answer: str, sources: List[Dict]) -> str:
    if answer is None:
        answer = ""
    correct = _format_sources_used(len(sources))
    pattern = re.compile(r"(?im)^(?P<prefix>\s*(?:\d+\)\s*)?Sources used:\s*).*$")
    if pattern.search(answer):
        answer = pattern.sub(rf"\g<prefix>{correct}", answer)
        return answer.strip()
    return (answer.rstrip() + "\n\nSources used: " + correct).strip()


# =========================
# Step 3: Parse & Chunk
# =========================

def detect_repeated_headers_footers(pages_text, top_n_lines=2, bottom_n_lines=2, min_repeat_ratio=0.6):
    total_pages = len(pages_text)
    if total_pages == 0:
        return set(), set()
    h_cnt, f_cnt = Counter(), Counter()
    for _, text in pages_text:
        lines = [ln.strip() for ln in (text or "").split("\n") if ln.strip()]
        if not lines:
            continue
        for ln in lines[:top_n_lines]:
            h_cnt[_normalize_line(ln)] += 1
        for ln in lines[-bottom_n_lines:]:
            f_cnt[_normalize_line(ln)] += 1
    thresh = max(2, int(total_pages * min_repeat_ratio))
    return {ln for ln, c in h_cnt.items() if c >= thresh}, {ln for ln, c in f_cnt.items() if c >= thresh}


def remove_detected_headers_footers(text, h_norms, f_norms):
    if not text:
        return ""
    lines = text.split("\n")
    cleaned = [ln for ln in lines if _normalize_line(ln) not in h_norms and _normalize_line(ln) not in f_norms]
    return _clean_text("\n".join(cleaned))


def parse_pdf_to_pages(pdf_path: str) -> List[Tuple[int, str]]:
    raw_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw_pages.append((i, _clean_text(page.extract_text() or "")))
    h, f = detect_repeated_headers_footers(raw_pages)
    return [(i, remove_detected_headers_footers(txt, h, f)) for i, txt in raw_pages if txt]


def chunk_text_by_tokens(full_text: str) -> List[str]:
    enc = tiktoken.get_encoding(TOKEN_ENCODING_NAME)
    tokens = enc.encode(full_text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + TOKEN_CHUNK_SIZE, len(tokens))
        chunk_text = enc.decode(tokens[start:end]).strip()
        if len(chunk_text) >= MIN_CHUNK_CHARS:
            chunks.append(chunk_text)
        if end == len(tokens):
            break
        start = end - TOKEN_CHUNK_OVERLAP
    return chunks


def build_chunks_from_pdf(pdf_path: str) -> List[Dict]:
    doc_name = os.path.basename(pdf_path)
    pages = parse_pdf_to_pages(pdf_path)
    joined = "\n\n".join([f"[PAGE {p}]\n{txt}" for p, txt in pages])
    chunks = chunk_text_by_tokens(joined)
    return [
        {"doc_name": doc_name, "chunk_id": f"{doc_name}::chunk_{idx:04d}", "text": c}
        for idx, c in enumerate(chunks)
    ]


def step3_ingest_to_json(input_dir=INPUT_PDF_DIR, output_path=OUTPUT_CHUNKS_PATH):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Dir not found: {input_dir}")
    pdfs = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(".pdf")])
    all_chunks = []
    for p in pdfs:
        all_chunks.extend(build_chunks_from_pdf(p))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {"num_pdfs": len(pdfs), "num_chunks": len(all_chunks), "chunks": all_chunks}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


# =========================
# Step 4: Embeddings & Manual Vector Search (Gemini)
# =========================

def load_parsed_chunks(path=OUTPUT_CHUNKS_PATH) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("chunks", [])


def extract_page_range(text: str) -> Tuple[Optional[int], Optional[int]]:
    pages = [int(x) for x in _PAGE_TAG_RE.findall(text or "")]
    return (min(pages), max(pages)) if pages else (None, None)


def _embedding_values(emb_obj) -> List[float]:
    """
    google-genai returns embedding objects where values live in `.values`.
    This helper keeps us robust if response shape evolves slightly.
    :contentReference[oaicite:4]{index=4}
    """
    if emb_obj is None:
        return []
    if hasattr(emb_obj, "values"):
        return list(emb_obj.values)
    # fallback: sometimes could already be list[float]
    if isinstance(emb_obj, (list, tuple)):
        return [float(x) for x in emb_obj]
    return [float(x) for x in list(emb_obj)]


def embed_texts_openai(
    texts: List[str],
    api_key: Optional[str] = None,
    batch_size: int = 96,
) -> List[List[float]]:
    """
    Compatibility shim: keep function name `embed_texts_openai` so other files don't change.
    Internally uses Gemini embedding API: client.models.embed_content(...). :contentReference[oaicite:5]{index=5}

    Task type:
    - For single-text calls (often query embedding): RETRIEVAL_QUERY
    - For batch calls (often document chunks): RETRIEVAL_DOCUMENT
    """
    client = _openai_client(api_key)

    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        task_type = "RETRIEVAL_QUERY" if len(batch) == 1 else "RETRIEVAL_DOCUMENT"
        cfg = types.EmbedContentConfig(task_type=task_type)

        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch if len(batch) > 1 else batch[0],
            config=cfg,
        )

        # result.embeddings is a list; each embedding has .values
        for emb in getattr(result, "embeddings", []) or []:
            vectors.append(_embedding_values(emb))

    return vectors


def step4_index_to_chroma(api_key: Optional[str] = None):
    """现更名为本地索引：将向量和元数据存入 JSON 模拟 ChromaDB"""
    chunks = load_parsed_chunks()
    if not chunks:
        print("❌ No chunks found.")
        return

    ids = [c["chunk_id"] for c in chunks]
    docs = [c["text"] for c in chunks]
    metadatas = []
    for c in chunks:
        p1, p2 = extract_page_range(c.get("text", ""))
        metadatas.append(
            {
                "doc_name": str(c.get("doc_name", "")),
                "page_start": int(p1) if p1 is not None else -1,
                "page_end": int(p2) if p2 is not None else -1,
            }
        )

    print(f"Indexing {len(docs)} chunks locally (using Manual Vector Store + Gemini embeddings)...")
    embeddings = embed_texts_openai(docs, api_key=api_key)

    store = {
        "ids": ids,
        "documents": docs,
        "metadatas": metadatas,
        "embeddings": embeddings,
    }

    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
    with open(VECTOR_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f)

    print(f"✅ Step4 index complete. Saved to: {VECTOR_STORE_PATH}")


def step4_query(query: str, top_k: int = RETRIEVAL_TOP_K, api_key: Optional[str] = None) -> Dict:
    """使用余弦相似度手动检索最相似的 chunk"""
    if not query.strip():
        raise ValueError("Query is empty")
    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError("Vector store not found. Please run index first.")

    with open(VECTOR_STORE_PATH, "r", encoding="utf-8") as f:
        store = json.load(f)

    # Query embedding via same shim (single -> RETRIEVAL_QUERY)
    q_emb = embed_texts_openai([query], api_key=api_key)[0]

    dists = [float(cosine(q_emb, emb)) for emb in store["embeddings"]]
    sorted_indices = np.argsort(dists)[:top_k]

    return {
        "ids": [[store["ids"][i] for i in sorted_indices]],
        "documents": [[store["documents"][i] for i in sorted_indices]],
        "metadatas": [[store["metadatas"][i] for i in sorted_indices]],
        "distances": [[dists[i] for i in sorted_indices]],
    }


# =========================
# Declarations Page facts (Scenario helper)
# =========================

def load_declarations_facts(parsed_json_path: str):
    if not os.path.exists(parsed_json_path):
        return {}, []
    with open(parsed_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    facts, evidence = {}, []
    for ch in data.get("chunks", []):
        doc, text = (ch.get("doc_name") or "").lower(), ch.get("text", "")
        if "declarations" not in doc and "decp" not in doc:
            continue
        if re.search(r"Uninsured/Underinsured\s+Motorist\s+Rejected", text, re.I):
            facts["um_uim"] = "Rejected"
            evidence.append("Declarations: UM/UIM = Rejected")
        m = re.search(
            r"Bodily Injury Liability\s+\$?([\d,]+)\s+each person/\$?([\d,]+)\s+each accident", text, re.I
        )
        if m:
            facts["bi_limit"] = f"${m.group(1)}/${m.group(2)}"
            evidence.append(f"Declarations: BI Liability limit = {facts['bi_limit']}")
    return facts, evidence


def _get_declarations_chunks_from_step3(parsed_json_path=OUTPUT_CHUNKS_PATH):
    if not os.path.exists(parsed_json_path):
        return []
    with open(parsed_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return [
        ch
        for ch in payload.get("chunks", [])
        if "declarations" in (ch.get("doc_name") or "").lower() or "decp" in (ch.get("doc_name") or "").lower()
    ]


def _build_declarations_block(intent: str) -> str:
    if intent != "Scenario":
        return ""
    try:
        facts, evidence = load_declarations_facts(OUTPUT_CHUNKS_PATH)
        if not facts and not evidence:
            return ""
        return (
            "Declarations Page facts:\n"
            + json.dumps(facts, indent=2)
            + "\nEvidence:\n- "
            + "\n- ".join(evidence)
            + "\n\n"
        )
    except Exception:
        return ""


# =========================
# Step 6: Intent Classification (Gemini)
# =========================

def classify_intent(question: str, api_key: Optional[str] = None) -> str:
    client = _openai_client(api_key)

    system_instruction = (
        "You are an intent classifier.\n"
        "Classify the user's question into exactly one label:\n"
        "- Informational\n"
        "- Clarification\n"
        "- Scenario\n"
        "Output ONLY the label."
    )

    # Keep your few-shot examples (same semantics as before)
    examples = [
        ("What is the deductible?", "Informational"),
        ("I went to the ER. Covered?", "Scenario"),
    ]

    prompt_lines = ["Examples:"]
    for q, a in examples:
        prompt_lines.append(f"Q: {q}\nA: {a}")
    prompt_lines.append("\nNow classify this question (output only the label):")
    prompt_lines.append(question)
    prompt = "\n".join(prompt_lines)

    resp = client.models.generate_content(
        model=INTENT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,
        ),
    )

    label = (getattr(resp, "text", "") or "").strip()
    return label if label in {"Informational", "Clarification", "Scenario"} else "Informational"


def build_answer_instruction(intent: str) -> str:
    if intent == "Informational":
        return (
            "Answer: \n"
            "- Provide the best possible answer using ONLY the provided context.\n\n"
            "What is unclear: \n"
            "- State any missing/unknown details that are not explicitly in the documents.\n\n"
            "Recommended Questions: \n"
            "- Provide 2–3 helpful follow-up questions.\n\n"
            "Sources used: [#]\n"
        )
    if intent == "Clarification":
        return (
            "Answer: \n"
            "- Provide the best possible answer using ONLY the provided context.\n\n"
            "What is unclear: \n"
            "- Explain what part is unclear or not explicitly stated in the documents.\n\n"
            "Recommended Questions: \n"
            "- Provide 2–3 follow-up questions that would resolve the ambiguity.\n\n"
            "Sources used: [#]\n"
        )
    return (
        "Answer: \n"
        "- Provide the best possible scenario-based answer using ONLY the provided context.\n\n"
        "What is unclear: \n"
        "- List any assumptions or missing policy details required to be certain.\n\n"
        "Recommended Questions: \n"
        "- Provide 2–3 follow-up questions that would clarify the scenario.\n\n"
        "Sources used: [#]\n"
    )


# =========================
# Step 5: RAG Answer (Gemini)
# =========================

def _build_context_from_retrieval(res: Dict, max_chars: int = MAX_CONTEXT_CHARS):
    ids, docs, metas, dists = res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    pieces, sources, evidence, total = [], [], [], 0
    for i in range(len(ids)):
        snippet = docs[i].strip()
        src = {
            "rank": i + 1,
            "chunk_id": ids[i],
            "doc_name": metas[i].get("doc_name"),
            "page_start": metas[i].get("page_start"),
            "page_end": metas[i].get("page_end"),
            "distance": dists[i],
        }
        sources.append(src)
        evidence.append(src.copy())
        evidence[-1]["text"] = snippet
        block = (
            f"[SOURCE {i + 1}] doc={src['doc_name']} pages={src['page_start']}-{src['page_end']} "
            f"dist={dists[i]:.3f}\n{snippet}\n"
        )
        if total + len(block) > max_chars:
            break
        pieces.append(block)
        total += len(block)
    return "\n---\n".join(pieces).strip(), sources, evidence


def rag_answer(question: str, api_key: Optional[str] = None, top_k: int = RAG_TOP_K) -> Dict:
    if not question.strip():
        raise ValueError("Empty question")
    retrieval = step4_query(question, top_k=top_k, api_key=api_key)
    if not retrieval.get("ids") or not retrieval["ids"][0]:
        return {"intent": "Informational", "answer": "No relevant text found.", "evidence": [], "sources": []}

    intent = classify_intent(question, api_key=api_key) if ENABLE_INTENT_ROUTER else "Informational"
    context_text, sources, evidence = _build_context_from_retrieval(retrieval)

    if intent == "Scenario":
        decl = _get_declarations_chunks_from_step3()
        if decl:
            d = decl[0]
            p1, p2 = extract_page_range(d["text"])
            context_text = (
                f"[SOURCE 0] doc={d['doc_name']} pages={p1}-{p2} dist=0.0\n{d['text']}\n---\n" + context_text
            )
            sources.insert(0, {"rank": 0, "doc_name": d["doc_name"], "page_start": p1, "page_end": p2, "distance": 0.0})

    instruction = build_answer_instruction(intent)

    system_instruction = "You are PolicyPal. Use provided context only."
    user_msg = (
        f"{_build_declarations_block(intent)}"
        f"Question: {question}\n\n"
        f"Context: {context_text}\n\n"
        f"Rules: Use ONLY context. Follow structure:\n{instruction}"
    )

    client = _openai_client(api_key)
    resp = client.models.generate_content(
        model=CHAT_MODEL,
        contents=user_msg,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
        ),
    )

    answer_text = (getattr(resp, "text", "") or "").strip()
    answer = _enforce_sources_used_line(answer_text, sources[:top_k])
    return {"intent": intent, "answer": answer, "evidence": evidence[:top_k], "sources": sources[:top_k]}


if __name__ == "__main__":
    import sys

    if "--step3" in sys.argv:
        step3_ingest_to_json()
    elif "--index" in sys.argv:
        step4_index_to_chroma()
    elif "--ask" in sys.argv:
        q = sys.argv[sys.argv.index("--ask") + 1]
        out = rag_answer(q)
        print(f"\nINTENT: {out['intent']}\n\nANSWER:\n{out['answer']}")