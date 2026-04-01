import os
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import core  # 复用你的：parse_pdf_to_pages/chunk_text_by_tokens/embed_texts_openai (Gemini shim)
from policy_paths import COMPARE_DIR

# Optional: BM25 (keep behavior consistent; install rank-bm25)
try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except Exception:
    BM25Okapi = None
    _HAS_BM25 = False


def _safe(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", (name or "policy").strip())


def _tokenize(text: str) -> List[str]:
    # Simple tokenizer; production can swap in more advanced tokenization if needed
    return re.findall(r"[A-Za-z0-9]+", (text or "").lower())


@dataclass
class PolicyIndexPaths:
    chunks_path: str
    store_path: str


def build_policy_index(
    policy_folder: str,
    policy_name: str,
    api_key: Optional[str] = None,
    out_dir: str = str(COMPARE_DIR),
) -> PolicyIndexPaths:
    """
    Production-ish index builder:
    - chunk (Step3)
    - dense embeddings
    - optional BM25 index

    NOTE:
    - We keep calling core.embed_texts_openai(...) for compatibility.
      In the Gemini-migrated core.py, this is a shim that uses Gemini embeddings.
    """
    if not os.path.isdir(policy_folder):
        raise FileNotFoundError(f"Folder not found: {policy_folder}")

    safe = _safe(policy_name)
    os.makedirs(out_dir, exist_ok=True)

    chunks_path = os.path.join(out_dir, f"{safe}__chunks.json")
    store_path = os.path.join(out_dir, f"{safe}__store.json")

    # Step 3: folder -> chunks json (core supports custom input_dir/output_path)
    payload = core.step3_ingest_to_json(input_dir=policy_folder, output_path=chunks_path)
    chunks: List[Dict[str, Any]] = payload.get("chunks", [])

    if not chunks:
        raise RuntimeError(f"No chunks built for policy '{policy_name}'")

    docs = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]

    metadatas = []
    for c in chunks:
        p1, p2 = core.extract_page_range(c.get("text", ""))
        metadatas.append(
            {
                "policy_name": policy_name,
                "doc_name": c.get("doc_name", ""),
                "page_start": int(p1) if p1 is not None else -1,
                "page_end": int(p2) if p2 is not None else -1,
                # optional hints (kept for forward compatibility)
                "section_hint": c.get("section_hint", ""),
                "extract_method": c.get("extract_method", "text"),
            }
        )

    # Dense embeddings (Gemini via core shim)
    embeddings = core.embed_texts_openai(docs, api_key=api_key)

    # Optional BM25
    if _HAS_BM25:
        tokenized = [_tokenize(t) for t in docs]
        bm25 = {
            "enabled": True,
            "tokenized": tokenized,  # simplified storage
        }
    else:
        bm25 = {"enabled": False}

    store = {
        "schema_version": 2,
        "policy_name": policy_name,
        "ids": ids,
        "documents": docs,
        "metadatas": metadatas,
        "embeddings": embeddings,
        "bm25": bm25,
        "embedding_model": core.EMBEDDING_MODEL,
    }

    with open(store_path, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)

    return PolicyIndexPaths(chunks_path=chunks_path, store_path=store_path)