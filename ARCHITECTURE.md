# ARCHITECTURE.md

## 1. System Overview

**Gemini-PolicyPal** is a Retrieval-Augmented Generation (RAG) system designed to analyze and compare insurance policy documents.

The system integrates:

* Large Language Models (Gemini)
* Vector-based retrieval
* PDF document processing
* Interactive UI (Streamlit)

It supports three main functionalities:

1. Policy analysis
2. Question answering (RAG)
3. Policy comparison

---

## 2. High-Level Architecture

The system follows a modular architecture:

```
User (Streamlit UI)
        ↓
Application Layer (app.py)
        ↓
Core Processing Layer (RAG Pipeline)
        ↓
Storage Layer (Chunks + Embeddings)
        ↓
LLM Layer (Gemini API)
```

---

## 3. Data Flow

### Step 1: Document Ingestion

* Input: Insurance policy PDFs
* Tool: `pdfplumber`
* Output: Raw extracted text

Processing includes:

* Removing repeated headers/footers
* Cleaning formatting issues

---

### Step 2: Chunking

* Implemented in: `core.py`
* Text is split into manageable chunks using token-based segmentation
* Overlapping chunks are used to preserve context

Output:

* JSON file containing text chunks with metadata

---

### Step 3: Embedding

* Model: `gemini-embedding-001`
* Each chunk is converted into a dense vector

Stored as:

* Vector embeddings
* Associated metadata (page range, document name)

---

### Step 4: Indexing

* Implemented in: `prod_index.py`
* Builds vector store from chunks
* Optional BM25 index for keyword-based retrieval

Storage:

```
storage/
└── __store.json
```

---

### Step 5: Retrieval

* Implemented in: `prod_retriever.py`
* Hybrid retrieval:

  * Dense vector similarity
  * BM25 keyword search

Process:

1. Convert user query into embedding
2. Retrieve top-k relevant chunks
3. Merge and deduplicate results

---

### Step 6: Context Construction

* Selected chunks are combined into a context window
* Context length controlled by token limits

---

### Step 7: LLM Generation

* Model: `gemini-2.5-flash`
* Prompt includes:

  * Retrieved context
  * User query

Output:

* Grounded answer (QA mode)
* Structured JSON (analysis mode)
* Comparison results (compare mode)

---

## 4. Key Modules

### 4.1 app.py (Application Layer)

* Streamlit frontend
* Handles:

  * User interaction
  * File selection
  * Routing between modes:

    * Dashboard
    * Ask Pal
    * Compare

---

### 4.2 core.py (RAG Engine)

* PDF parsing
* Chunking
* Embedding
* Retrieval helpers
* Context construction
* LLM interaction

---

### 4.3 auto_analysis.py

* Extracts structured policy insights
* Uses constrained prompting to generate JSON output

---

### 4.4 prod_index.py

* Builds production-ready vector store
* Adds metadata for retrieval

---

### 4.5 prod_retriever.py

* Hybrid retrieval engine
* Combines:

  * Semantic similarity
  * Keyword matching

---

### 4.6 prod_compare.py

* Policy comparison logic

* Retrieves evidence for:

  * Coverage
  * Deductibles
  * Exclusions
  * Premiums

* Generates:

  * Structured comparison
  * Winner decision
  * Trade-offs

---

### 4.7 compare_policies.py

* Visualization module
* Generates radar chart using Plotly

---

## 5. Storage Design

### Input Data

```
data/
├── qa_policies/
└── compare_prod/
```

### Processed Data

```
storage/
└── __store.json
```

### Metadata Includes:

* Document name
* Page range
* Section hints
* Extraction method

---

## 6. RAG Design Choices

### 6.1 Hybrid Retrieval

* Dense retrieval → semantic understanding
* BM25 → keyword precision

→ Improves accuracy and robustness

---

### 6.2 Chunking Strategy

* Token-based splitting
* Overlapping chunks

→ Preserves context across sections

---

### 6.3 Anti-Hallucination Design

* Retrieval before generation
* Evidence-based answering
* Placeholder detection (e.g., `$000`, `TBD`)

---

### 6.4 Structured Output

* JSON-based responses
* Enables:

  * Visualization
  * Comparison
  * Consistent outputs

---

## 7. UI Architecture

* Framework: Streamlit
* Pages:

  * Dashboard → Policy analysis
  * Ask Pal → Q&A interface
  * Compare → Policy comparison

Features:

* Interactive charts (Plotly)
* Real-time responses
* Multi-document support

---

## 8. Limitations

* Dependent on PDF text quality
* No OCR support for scanned documents
* LLM responses are not fully deterministic

---

## 9. Future Improvements

* Add OCR support
* Improve retrieval ranking
* Introduce evaluation metrics
* Extend to more insurance types

---

## 10. Summary

Gemini-PolicyPal uses a modular RAG architecture to transform unstructured insurance documents into structured, explainable insights through:

* Document processing
* Hybrid retrieval
* LLM reasoning

This design ensures scalability, interpretability, and real-world applicability.
