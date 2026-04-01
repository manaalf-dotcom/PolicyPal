"""
PolicyPal — Interactive ChatGPT-Style Q&A with Optimized Dashboard
Focus: Real-time user feedback and clear data visualization
"""
import os
import sys
import json
import pathlib
import numpy as np
from scipy.spatial.distance import cosine
import pdfplumber
import re
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv
import base64

# ── 1. SETUP & CSS ────────────────────────────────────────────────────────────
load_dotenv()
st.set_page_config(
    page_title="PolicyPal — Your Insurance, Simplified",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

try:
    current_dir = pathlib.Path(__file__).parent
    css_path = current_dir / "styles.css"
    _css = css_path.read_text(encoding="utf-8")

    ui_fixes = """
        [data-testid="stBottom"] {
            padding-bottom: 2.5rem !important;
            background: transparent !important;
        }
        [data-testid="stBottom"] > div {
            background: transparent !important;
        }
        [data-testid="stChatInput"] {
            border-radius: 50px !important;
            max-width: 850px !important;
            margin: 0 auto !important;
        }
        .stMainBlockContainer {
            padding-bottom: 150px !important;
        }
        * {
            overflow-anchor: none !important;
        }
    """
    _fonts = '<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">'
    st.markdown(_fonts, unsafe_allow_html=True)
    st.markdown(f"<style>{_css}\n{ui_fixes}</style>", unsafe_allow_html=True)
except Exception as e:
    st.warning(f"styles.css loading error: {e}")


def _get_api_key():
    k = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    if k:
        return k
    st.error("⚠️ Add GEMINI_API_KEY to your Streamlit secrets or .env file.")
    st.stop()


API_KEY = _get_api_key()

# ── 2. BACKEND IMPORTS ────────────────────────────────────────────────────────
import core
from policy_paths import PROJECT_ROOT, POLICY_A_DIR, POLICY_B_DIR, COMPARE_DIR
from prod_index import build_policy_index
from auto_analysis import analyze_policy_document
from prod_compare import build_policy_summary, compare_policies_prod
from compare_policies import build_radar_chart, compare_policies_llm

# ── 3. STATE & UTILS ──────────────────────────────────────────────────────────
def open_folder(path: pathlib.Path):
    """
    Cloud-safe replacement for opening folders.
    Streamlit Cloud has no GUI, so we only show the folder path.
    """
    path.mkdir(parents=True, exist_ok=True)
    st.info(f"Folder path: {path.resolve()}")


for k, v in {
    "page": "dashboard",
    "analysis": None,
    "policy_text": None,
    "chat_history": [],
    "comparison": None,
    "cmp_name_a": "Policy A",
    "cmp_name_b": "Policy B",
    "compare_last_answer": "",
    "a_store": "",
    "b_store": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

QA_PDF_DIR = PROJECT_ROOT / "data" / "qa_policies"
QA_CHUNKS_PATH = PROJECT_ROOT / "storage" / "qa_parsed_chunks.json"
QA_VECTOR_STORE_PATH = PROJECT_ROOT / "storage" / "qa_vector_store.json"

for p in [QA_PDF_DIR, QA_CHUNKS_PATH.parent, POLICY_A_DIR, POLICY_B_DIR, COMPARE_DIR]:
    p.mkdir(parents=True, exist_ok=True)


import tempfile

def save_uploaded_files(uploaded_files, target_dir: pathlib.Path):
    target_dir.mkdir(parents=True, exist_ok=True)

    # 先清空旧文件，避免混淆
    for old_file in target_dir.glob("*"):
        if old_file.is_file():
            old_file.unlink()

    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = target_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)

    return saved_paths


def clear_folder(folder: pathlib.Path):
    folder.mkdir(parents=True, exist_ok=True)
    for f in folder.glob("*"):
        if f.is_file():
            f.unlink()

def safe_list_pdfs(folder: pathlib.Path):
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(folder.glob("*.pdf"))


def extract_text_from_pdf(pdf_path: pathlib.Path, max_chars: int = 20000) -> str:
    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_parts.append(page_text.strip())
    except Exception:
        return ""
    return "\n".join(text_parts)[:max_chars].strip()


def extract_text_from_folder(folder_path: str, max_chars: int = 40000) -> str:
    text_parts = []
    folder = pathlib.Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return ""

    for pdf_file in folder.glob("*.pdf"):
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_parts.append(page_text.strip())
        except Exception:
            pass

    return "\n".join(text_parts)[:max_chars].strip()


def folder_has_text_pdf(folder: pathlib.Path) -> bool:
    """
    Return True if at least one PDF in folder yields non-empty text.
    """
    for pdf_file in safe_list_pdfs(folder):
        if extract_text_from_pdf(pdf_file):
            return True
    return False


def build_qa_index_from_folder(folder_path: str):
    payload = core.step3_ingest_to_json(
        input_dir=str(folder_path),
        output_path=str(QA_CHUNKS_PATH)
    )

    chunks = payload.get("chunks", [])
    cleaned_chunks = []
    for c in chunks:
        txt = c.get("text", "")
        if isinstance(txt, str) and txt.strip():
            cleaned_chunks.append(c)

    if not cleaned_chunks:
        raise ValueError("No valid text chunks extracted from QA folder.")

    ids = [c["chunk_id"] for c in cleaned_chunks]
    docs = [c["text"].strip() for c in cleaned_chunks]
    metadatas = [
        {
            "doc_name": c.get("doc_name", ""),
            "page_start": core.extract_page_range(c.get("text", ""))[0] or -1,
        }
        for c in cleaned_chunks
    ]

    embeddings = core.embed_texts_openai(docs, api_key=API_KEY)
    if not embeddings:
        raise ValueError("Embedding generation returned no results for QA folder.")

    store = {
        "ids": ids,
        "documents": docs,
        "metadatas": metadatas,
        "embeddings": embeddings,
    }
    with open(QA_VECTOR_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f)


def query_rag(question: str):
    if not QA_VECTOR_STORE_PATH.exists():
        return "Index not found. Please analyze and index documents first.", []

    with open(QA_VECTOR_STORE_PATH, "r", encoding="utf-8") as f:
        store = json.load(f)

    if not store.get("embeddings") or not store.get("documents"):
        return "The QA index is empty. Please rebuild the index.", []

    q_embs = core.embed_texts_openai([question], api_key=API_KEY)
    if not q_embs:
        return "Failed to create embedding for the question.", []

    q_emb = q_embs[0]
    dists = [float(cosine(q_emb, emb)) for emb in store["embeddings"]]
    idx = np.argsort(dists)[:3]

    retrieval = {
        "ids": [[store["ids"][i] for i in idx]],
        "documents": [[store["documents"][i] for i in idx]],
        "metadatas": [[store["metadatas"][i] for i in idx]],
        "distances": [[dists[i] for i in idx]],
    }

    context_text, sources, _ = core._build_context_from_retrieval(retrieval)

    from google.genai import types
    client = core._openai_client(API_KEY)

    resp = client.models.generate_content(
        model=core.CHAT_MODEL,
        contents=f"Question: {question}\n\nContext: {context_text}",
        config=types.GenerateContentConfig(
            system_instruction="""
Use the context only.
Answer naturally and conversationally.
Do not mention sources, citations, document labels, or phrases like
'Source 1', 'Source 2', 'according to the document', or 'the policy states'.
Do not quote source numbers in the answer.
""",
            temperature=0.2,
        ),
    )

    answer = getattr(resp, "text", "").strip()
    answer = re.sub(r"\(Source\s*\d+\)", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"Source\s*\d+", "", answer, flags=re.IGNORECASE)

    return answer.strip(), sources


# ── 4. VISUAL COMPONENTS ──────────────────────────────────────────────────────
def pal_svg(size=44, state="default"):
    img_path = "logo.png"
    if os.path.exists(img_path):
        try:
            with open(img_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
            return f'<img src="data:image/png;base64,{encoded_string}" width="{size}" height="{size}" style="object-fit: contain;">'
        except Exception:
            pass

    s = size
    h = int(s * 1.2)
    grad = f'<linearGradient id="pg{s}{state}" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#4F46E5"/><stop offset="100%" stop-color="#7C3AED"/></linearGradient>'
    shield = f'<path d="M{s*.5} {s*.04} C{s*.5} {s*.04} {s*.94} {s*.15} {s*.97} {s*.22} L{s*.97} {s*.5} C{s*.97} {s*.78} {s*.5} {s*1.15} {s*.5} {s*1.15} C{s*.5} {s*1.15} {s*.03} {s*.78} {s*.03} {s*.5} L{s*.03} {s*.22} C{s*.06} {s*.15} {s*.5} {s*.04} {s*.5} {s*.04}Z" fill="url(#pg{s}{state})"/>'
    return f'<svg width="{s}" height="{h}" viewBox="0 0 {s} {h}" xmlns="http://www.w3.org/2000/svg"><defs>{grad}</defs>{shield}</svg>'


def donut_chart(areas):
    labels = list(areas.keys())
    values = list(areas.values())
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            textinfo="label+percent",
            marker=dict(colors=["#6366F1", "#8B5CF6", "#06B6D4", "#10B981"]),
        )
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(color="#A89FCC"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=450,
        margin=dict(t=20, b=100, l=20, r=20),
    )
    return fig


# ── 5. NAVIGATION ─────────────────────────────────────────────────────────────
def render_nav():
    logo_col, spacer, c1, c2, c3 = st.columns([1.5, 3, 1, 1, 1])

    with logo_col:
        st.markdown(
            f'<div class="pp-logo" style="height:44px;display:flex;align-items:center;gap:10px;padding-left:8px">{pal_svg(28)} PolicyPal</div>',
            unsafe_allow_html=True,
        )
    with spacer:
        st.empty()

    with c1:
        if st.button(
            "Dashboard",
            key="n1",
            use_container_width=True,
            type="primary" if st.session_state.page == "dashboard" else "secondary",
        ):
            st.session_state.page = "dashboard"
            st.rerun()

    with c2:
        if st.button(
            "Compare",
            key="n2",
            use_container_width=True,
            type="primary" if st.session_state.page == "compare" else "secondary",
        ):
            st.session_state.page = "compare"
            st.rerun()

    with c3:
        if st.button(
            "Ask Pal",
            key="n3",
            use_container_width=True,
            type="primary" if st.session_state.page == "ask" else "secondary",
        ):
            st.session_state.page = "ask"
            st.rerun()


# ── 6. PAGES ──────────────────────────────────────────────────────────────────
def page_dashboard():
    an = st.session_state.analysis
    st.markdown('<div class="pp-page"><div class="orb-tr"></div><div class="orb-bl"></div>', unsafe_allow_html=True)

    if an is None:
        st.markdown(
            '<div class="hero-wrap"><div class="hero-h">Your insurance,<br><span class="hero-grad">simplified.</span></div></div>',
            unsafe_allow_html=True,
        )

        _, center_col, _ = st.columns([1, 2.5, 1])

        with center_col:
            st.markdown(
                f'''
                <div class="upload-zone-wrapper">
                    <div class="upload-zone-inner" style="flex-direction: column; justify-content: center; text-align: center; gap: 0.8rem; padding: 2.5rem 2rem;">
                        <div>{pal_svg(64)}</div>
                        <div class="upload-card-text">
                            <h3 style="margin-bottom: 0.4rem;">Upload Policy PDFs</h3>
                            <p>You can upload one or multiple PDF files for analysis.</p>
                        </div>
                    </div>
                </div>''',
                unsafe_allow_html=True,
            )

            uploaded_qa_files = st.file_uploader(
                "Upload policy PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                key="qa_uploader"
            )

            if uploaded_qa_files:
                st.success(f"{len(uploaded_qa_files)} file(s) ready for analysis.")
                

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Clear Uploaded QA Files", use_container_width=True):
                    clear_folder(QA_PDF_DIR)
                    st.session_state.analysis = None
                    st.session_state.policy_text = None
                    st.success("QA folder cleared.")
                    st.rerun()

            start_analysis = False
            with c2:
                if st.button("Analyze & Index", type="primary", use_container_width=True):
                    start_analysis = True

            if start_analysis:
                if not uploaded_qa_files:
                    st.error("Please upload at least one PDF file.")
                    st.stop()

                save_uploaded_files(uploaded_qa_files, QA_PDF_DIR)

                with st.spinner("Extracting & Indexing..."):
                    try:
                        text = extract_text_from_folder(str(QA_PDF_DIR))
                        if not text:
                            st.error("No readable text extracted from uploaded PDFs.")
                            st.stop()

                        st.session_state.policy_text = text
                        st.session_state.analysis = analyze_policy_document(text, API_KEY)
                        build_qa_index_from_folder(str(QA_PDF_DIR))
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to analyze and index documents: {e}")
                        st.stop()

            qa_files = safe_list_pdfs(QA_PDF_DIR)
            if qa_files:
                st.caption(f"Detected {len(qa_files)} PDF file(s) in QA folder.")
            else:
                st.caption("No PDF files detected in QA folder yet.")

            if start_analysis:
                if not qa_files:
                    st.error("No PDF files found in data/qa_policies.")
                    st.stop()

                if not folder_has_text_pdf(QA_PDF_DIR):
                    st.error("PDF files were found, but no readable text could be extracted.")
                    st.stop()

                with st.spinner("Extracting & Indexing..."):
                    try:
                        text = extract_text_from_folder(str(QA_PDF_DIR))
                        if not text:
                            st.error("No readable text extracted from QA folder.")
                            st.stop()

                        st.session_state.policy_text = text
                        st.session_state.analysis = analyze_policy_document(text, API_KEY)
                        build_qa_index_from_folder(str(QA_PDF_DIR))
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to analyze and index documents: {e}")
                        st.stop()

        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown(
        f'''<div class="policy-banner">
          <div class="banner-gradient-bar"></div>
          {pal_svg(52)}
          <div style="flex-grow:1">
            <div class="policy-name">{an.get("insurer", "Your Policy")}</div>
            <div class="policy-meta">Indexed from folder &nbsp;·&nbsp; <span class="active">Active</span></div>
          </div>
        </div>''',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="stat-card sc-1"><div class="stat-label">Deductible</div><div class="stat-value">{an.get("deductible","—")}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="stat-card sc-2"><div class="stat-label">Premium</div><div class="stat-value">{an.get("monthly_premium") or an.get("annual_premium") or "—"}</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="stat-card sc-3"><div class="stat-label">Out-of-Pocket</div><div class="stat-value">{an.get("out_of_pocket_max","—")}</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="stat-card sc-4"><div class="stat-label">Risk Level</div><div class="stat-value">{an.get("risk_score",5)}/10</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="gap-md"></div>', unsafe_allow_html=True)

    st.markdown(
        f'''<div class="summary-card">
      <div class="sum-header"><div><h3>Summary</h3></div></div>
      <div class="sum-text">{an.get("plain_summary","")}</div>
      <div class="ideal-for"> <strong>Ideal for:</strong> {an.get("who_its_good_for","")}</div>
    </div>''',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="gap-lg"></div>', unsafe_allow_html=True)

    st.markdown('<div class="cc"><div class="cc-title">Coverage Composition</div>', unsafe_allow_html=True)
    areas = an.get("coverage_areas", {})
    if areas:
        st.plotly_chart(donut_chart(areas), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="gap-lg"></div>', unsafe_allow_html=True)
    if st.button("Reset and Analyze New Policy Content", use_container_width=True):
        st.session_state.analysis = None
        st.session_state.policy_text = None
        if QA_VECTOR_STORE_PATH.exists():
            try:
                QA_VECTOR_STORE_PATH.unlink()
            except Exception:
                pass
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def page_compare():
    st.markdown('<div class="pp-page"><div class="orb-tr"></div><div class="orb-bl"></div>', unsafe_allow_html=True)
    st.markdown('<div class="cmp-headline">Policy <span>Comparison</span></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cmp-sub">Compare two insurance policies side-by-side with AI-powered insights</div>',
        unsafe_allow_html=True,
    )

    c_left, c_right = st.columns([1.2, 2.5], gap="large")

    with c_left:
        st.markdown(
            '''
            <div style="background:rgba(0,0,0,0.25); border:1px solid rgba(167,139,250,0.2); border-radius:12px; padding:10px 14px; margin-bottom:1.5rem; display:flex; align-items:center; gap:8px; font-family:'Courier New', monospace; font-size:0.85rem; color:#A78BFA; box-shadow:inset 0 2px 4px rgba(0,0,0,0.2);">
                <span style="color:#6B5F8A;">Mode</span> Upload multiple PDFs for each policy
            </div>
            ''',
            unsafe_allow_html=True,
        )

        st.markdown(
            '''
            <div style="font-weight:700; color:#EEE8FF; margin-bottom:1.5rem; font-size:1.05rem;">
                 Upload Policy Files
            </div>
            ''',
            unsafe_allow_html=True,
        )

        # Policy A
        st.markdown(
            '<div style="font-size:0.85rem; font-weight:600; color:#EEE8FF; margin-bottom:8px;">Policy A</div>',
            unsafe_allow_html=True,
        )

        uploaded_a_files = st.file_uploader(
            "Upload PDF(s) for Policy A",
            type=["pdf"],
            accept_multiple_files=True,
            key="policy_a_uploader",
        )

        na = st.text_input("Label A", value=st.session_state.cmp_name_a)

        if uploaded_a_files:
            st.caption(f"Policy A uploaded files: {len(uploaded_a_files)}")


        st.markdown('<div class="gap-md"></div>', unsafe_allow_html=True)

        # Policy B
        st.markdown(
            '<div style="font-size:0.85rem; font-weight:600; color:#EEE8FF; margin-bottom:8px;">Policy B</div>',
            unsafe_allow_html=True,
        )

        uploaded_b_files = st.file_uploader(
            "Upload PDF(s) for Policy B",
            type=["pdf"],
            accept_multiple_files=True,
            key="policy_b_uploader",
        )

        nb = st.text_input("Label B", value=st.session_state.cmp_name_b)

        if uploaded_b_files:
            st.caption(f"Policy B uploaded files: {len(uploaded_b_files)}")


        st.markdown('<div class="gap-lg"></div>', unsafe_allow_html=True)

        c_btn1, c_btn2 = st.columns(2)

        with c_btn1:
            if st.button("Clear Policy Files", use_container_width=True):
                clear_folder(POLICY_A_DIR)
                clear_folder(POLICY_B_DIR)
                st.session_state.comparison = None
                st.session_state.compare_last_answer = ""
                st.session_state.a_store = ""
                st.session_state.b_store = ""
                st.success("Uploaded policy files cleared.")
                st.rerun()

        with c_btn2:
            run_compare = st.button("Run Comparison", type="primary", use_container_width=True)

        if run_compare:
            if not uploaded_a_files:
                st.error("Please upload at least one PDF for Policy A.")
                st.stop()

            if not uploaded_b_files:
                st.error("Please upload at least one PDF for Policy B.")
                st.stop()

            save_uploaded_files(uploaded_a_files, POLICY_A_DIR)
            save_uploaded_files(uploaded_b_files, POLICY_B_DIR)

            if not folder_has_text_pdf(POLICY_A_DIR):
                st.error("Policy A PDFs were uploaded, but no readable text could be extracted.")
                st.stop()

            if not folder_has_text_pdf(POLICY_B_DIR):
                st.error("Policy B PDFs were uploaded, but no readable text could be extracted.")
                st.stop()

            with st.spinner("Analyzing..."):
                try:
                    idx_a = build_policy_index(str(POLICY_A_DIR), na, API_KEY, str(COMPARE_DIR))
                    idx_b = build_policy_index(str(POLICY_B_DIR), nb, API_KEY, str(COMPARE_DIR))

                    st.session_state.a_store = idx_a.store_path
                    st.session_state.b_store = idx_b.store_path

                    st.session_state.comparison = compare_policies_llm(
                        build_policy_summary(na, idx_a.store_path, API_KEY),
                        build_policy_summary(nb, idx_b.store_path, API_KEY),
                        API_KEY,
                    )

                    st.session_state.cmp_name_a = na
                    st.session_state.cmp_name_b = nb
                    st.session_state.compare_last_answer = ""
                except Exception as e:
                    st.error(f"Comparison failed: {e}")
                    st.stop()

            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with c_right:
        cmp = st.session_state.comparison
        if cmp:
            st.plotly_chart(
                build_radar_chart(cmp, st.session_state.cmp_name_a, st.session_state.cmp_name_b),
                use_container_width=True,
            )

            q = st.text_area("", placeholder="Ask a cross-policy question...")

            if st.button("💬 Retrieve & Compare", type="primary"):
                if not q.strip():
                    st.warning("Please enter a question.")
                elif not st.session_state.a_store or not st.session_state.b_store:
                    st.error("Please run comparison first.")
                else:
                    with st.spinner("Searching..."):
                        try:
                            st.session_state.compare_last_answer = compare_policies_prod(
                                st.session_state.cmp_name_a,
                                st.session_state.a_store,
                                st.session_state.cmp_name_b,
                                st.session_state.b_store,
                                q,
                                API_KEY,
                            )
                        except Exception as e:
                            st.error(f"Detailed comparison failed: {e}")

            if st.session_state.compare_last_answer:
                st.markdown(st.session_state.compare_last_answer, unsafe_allow_html=True)

        else:
            st.markdown(
                '''
                <div class="ready-placeholder">
                    <div class="rp-glow"></div>
                    <div class="rp-icon">✨</div>
                    <h3>Ready to Compare</h3>
                    <p>Upload PDF files for Policy A and Policy B, then click "Run Comparison".</p>
                </div>
                ''',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


def page_ask():
    st.markdown('<div class="pp-page"><div class="orb-tr"></div><div class="orb-bl"></div>', unsafe_allow_html=True)

    head_col, btn_col = st.columns([3, 1])
    with head_col:
        st.markdown(
            '<div class="hero-h" style="font-size:2.5rem; margin-bottom:1rem;">Ask <span class="hero-grad">Pal</span></div>',
            unsafe_allow_html=True
        )


    if not st.session_state.policy_text:
        st.info("Please index your documents in the Dashboard first.")
        return

    user_q = st.chat_input("Ask about your policy documents...")
    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        st.rerun()

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            _, c2 = st.columns([1, 4])
            with c2:
                st.markdown(
                    f'<div style="display:flex;gap:10px;justify-content:flex-end;margin-bottom:14px"><div class="bubble-user">{msg["content"]}</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            c1, _ = st.columns([4, 1])
            with c1:
                content = msg["content"]
                st.markdown(
                    f'<div style="display:flex;gap:12px;margin-bottom:14px">{pal_svg(44)}<div class="bubble-pal">{content}</div></div>',
                    unsafe_allow_html=True,
                )

    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        c1, _ = st.columns([4, 1])
        with c1:
            placeholder = st.empty()
            placeholder.markdown(
                f'''
                <div style="display:flex;gap:12px;align-items:flex-start;margin-bottom:14px">
                  {pal_svg(44)}
                  <div class="bubble-pal">
                    <div class="typing-bubble">
                      <div class="t-dot"></div><div class="t-dot"></div><div class="t-dot"></div>
                    </div>
                  </div>
                </div>
                ''',
                unsafe_allow_html=True,
            )

            last_q = st.session_state.chat_history[-1]["content"]
            try:
                ans, sources = query_rag(last_q)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": ans, "sources": sources}
                )
            except Exception as e:
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": f"Sorry, something went wrong: {e}", "sources": []}
                )
            st.rerun()

    st.markdown('<div style="height:100px"></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── 7. EXECUTE ────────────────────────────────────────────────────────────────
render_nav()
if st.session_state.page == "dashboard":
    page_dashboard()
elif st.session_state.page == "compare":
    page_compare()
elif st.session_state.page == "ask":
    page_ask()