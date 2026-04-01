import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy.spatial.distance import cosine

import core

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except Exception:
    BM25Okapi = None
    _HAS_BM25 = False


def _load_store(store_path: str) -> Dict[str, Any]:
    with open(store_path, "r", encoding="utf-8") as f:
        return json.load(f)


def dense_search(store: Dict[str, Any], query: str, api_key: Optional[str], top_k: int) -> List[int]:
    """
    Dense retrieval using query embedding.
    NOTE: We keep calling core.embed_texts_openai(...) for compatibility.
    In Gemini-migrated core.py, this function is a shim that uses Gemini embeddings.
    """
    docs = store["documents"]
    embs = store["embeddings"]

    # Query embedding (Gemini via core shim)
    q_vec = np.array(core.embed_texts_openai([query], api_key=api_key)[0], dtype=float)

    dist: List[Tuple[int, float]] = []
    for i, v in enumerate(embs):
        d = cosine(q_vec, np.array(v, dtype=float))
        dist.append((i, float(d)))
    dist.sort(key=lambda x: x[1])
    return [i for i, _ in dist[:top_k]]


def bm25_search(store: Dict[str, Any], query: str, top_k: int) -> List[int]:
    bm25_info = store.get("bm25", {}) or {}
    if not _HAS_BM25 or not bm25_info.get("enabled"):
        return []

    tokenized = bm25_info["tokenized"]
    bm25 = BM25Okapi(tokenized)
    q_tokens = [t.lower() for t in query.split() if t.strip()]
    scores = bm25.get_scores(q_tokens)
    idx = np.argsort(-scores)[:top_k]
    return [int(i) for i in idx]


def retrieve_evidence(
    store_path: str,
    queries: List[str],
    api_key: Optional[str],
    dense_top_k: int = 12,
    bm25_top_k: int = 8,
    final_k: int = 12,
) -> List[Dict[str, Any]]:
    """
    Production retrieval:
    - multi-query
    - dense + bm25
    - merge + dedupe
    """
    store = _load_store(store_path)
    picked = []
    seen = set()

    for q in queries:
        for i in dense_search(store, q, api_key, dense_top_k):
            if i not in seen:
                seen.add(i)
                picked.append(i)
        for i in bm25_search(store, q, bm25_top_k):
            if i not in seen:
                seen.add(i)
                picked.append(i)

    # truncate to final_k
    picked = picked[:final_k]

    out = []
    for i in picked:
        out.append(
            {
                "text": store["documents"][i],
                "metadata": store["metadatas"][i],
                "chunk_id": store["ids"][i],
            }
        )
    return out