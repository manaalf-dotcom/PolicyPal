# prod_compare.py
# Production compare: cached structured summaries (per policy) + compare summaries.
# Upgrade: Detect placeholder values (e.g., $000, $XXX, TBD, "see schedule") and treat as Missing.
#
# Output format (per request): matches the UI-friendly "Comparison Result" style:
# - "## Comparison Result"
# - 1-paragraph direct answer
# - "### Here is a comparison of the available information:" + fixed-width HTML table
# - "## Key Differences" bullets
# - "## Who Should Choose A vs B" bullets
# - "## Missing Info Checklist"

import json
import os
import re
from typing import Dict, Any, Optional, List

import core
from policy_paths import COMPARE_DIR
from prod_retriever import retrieve_evidence

# Gemini types (core.py provides Gemini client via _openai_client shim)
from google.genai import types

# Fixed K (NOT shown in UI)
COMPARE_TOP_K = 12

# ---------- field routing / query templates ----------
FIELD_QUERIES: Dict[str, List[str]] = {
    "coverage_limits": [
        "Limits of Liability",
        "liability limits",
        "Bodily Injury Liability",
        "Property Damage Liability",
        "each person",
        "each accident",
        "per accident",
        "limit of liability",
        "Uninsured Motorist",
        "Underinsured Motorist",
        "UM/UIM",
    ],
    "deductibles": [
        "Deductible",
        "collision deductible",
        "comprehensive deductible",
        "deductible applies",
        "Collision",
        "Comprehensive",
        "Other Than Collision",
    ],
    "exclusions": [
        "Exclusions",
        "We do not provide coverage",
        "This coverage does not apply",
        "is not covered",
        "not cover",
        "not pay",
    ],
    "claim_conditions": [
        "Duties after an accident",
        "Duties after loss",
        "Notice",
        "promptly notify",
        "cooperate",
        "proof of loss",
        "assist and cooperate",
        "claim reporting",
        "report the accident",
    ],
    "premium": [
        "Premium",
        "Total premium",
        "policy premium",
        "fees",
        "payment",
        "POLICY PREMIUM",
        "Declarations",
    ],
}


# ---------- helpers ----------
def _safe_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        name = "policy"
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", name)


def _summary_path(policy_name: str) -> str:
    safe = _safe_name(policy_name)
    return os.path.join(str(COMPARE_DIR), f"{safe}__summary.json")


def _ensure_field_obj(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"value": None, "status": "missing", "evidence": []}
    value = obj.get("value", None)
    status = (obj.get("status") or "missing").lower()
    evidence = obj.get("evidence") or []
    if status not in ("found", "inferred", "missing"):
        status = "missing"
    if not isinstance(evidence, list):
        evidence = []
    return {"value": value, "status": status, "evidence": evidence}


# ---------- placeholder detection ----------
_PLACEHOLDER_PATTERNS = [
    r"^\$?\s*0{2,}\s*$",  # "000" / "$000" / " 000 "
    r"^\$?\s*x{2,}\s*$",  # "XXX" / "$XXX"
    r"\bTBD\b",
    r"to be determined",
    r"not provided",
    r"not specified",
    r"see (the )?(declarations|schedule)",
    r"refer to (the )?(declarations|schedule|endorsement)",
    r"shown on (the )?(declarations|schedule)",
]


def _is_placeholder_value(value: Any) -> bool:
    """
    Detect template / placeholder / deferred references that should be treated as Missing.
    IMPORTANT: We do NOT treat 'N/A' or 'Not Applicable' as placeholder;
    those can be a valid "not applicable" value (not missing).
    """
    if value is None:
        return False
    txt = str(value).strip()
    if not txt:
        return False

    low = txt.lower()

    if low in {"n/a", "na", "not applicable"}:
        return False

    # "$0" can be a real deductible; do NOT blanket-mark as placeholder.
    if low in {"$0", "0", "$0.00", "0.00"}:
        return False

    for pat in _PLACEHOLDER_PATTERNS:
        if re.search(pat, txt, flags=re.IGNORECASE):
            return True

    if "___" in txt or "____" in txt:
        return True

    return False


def _normalize_placeholders_in_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-process LLM summary:
    if a field value looks like placeholder, force it to missing.
    """
    if not isinstance(summary, dict):
        return {"policy_name": "", "fields": {}}

    fields = summary.get("fields", {})
    if not isinstance(fields, dict):
        fields = {}

    for k in ["coverage_limits", "deductibles", "exclusions", "claim_conditions", "premium"]:
        fo = _ensure_field_obj(fields.get(k, {}))
        val = fo.get("value", None)

        if _is_placeholder_value(val):
            fo["value"] = None
            fo["status"] = "missing"
            ev = fo.get("evidence") or []
            ev = ev if isinstance(ev, list) else []
            ev.insert(0, "Detected placeholder / template value; treated as Missing.")
            fo["evidence"] = ev[:5]
            fields[k] = fo
        else:
            fields[k] = fo

    return {"policy_name": summary.get("policy_name", ""), "fields": fields}


def _render_value(field_obj: Dict[str, Any]) -> str:
    """
    For comparison table cell. Always return a string.
    Also enforce placeholder->Missing at render time (extra safety).
    """
    status = (field_obj.get("status") or "missing").lower()
    value = field_obj.get("value", None)

    if status == "missing" or value is None or str(value).strip() == "":
        return "Missing"

    if _is_placeholder_value(value):
        return "Missing"

    return str(value).strip()


def _missing_fields(summary: Dict[str, Any]) -> List[str]:
    fields = (summary.get("fields") or {}) if isinstance(summary, dict) else {}
    missing = []
    for k in ["coverage_limits", "deductibles", "exclusions", "claim_conditions", "premium"]:
        fo = _ensure_field_obj(fields.get(k, {}))
        if fo["status"] == "missing" or _is_placeholder_value(fo.get("value")):
            missing.append(k)
    return missing


def _build_fixed_width_table_html(policy_a_name: str, policy_b_name: str, rows: List[tuple]) -> str:
    """
    Fixed-width HTML table to prevent huge stretching from long text.
    Render with st.markdown(..., unsafe_allow_html=True) in app.py.
    """
    table_html = f"""
<table style="width:100%; table-layout:fixed; border-collapse:collapse;">
  <tr>
    <th style="width:18%; text-align:left; border-bottom:1px solid #444; padding:8px;">Feature</th>
    <th style="width:41%; text-align:left; border-bottom:1px solid #444; padding:8px;">{policy_a_name}</th>
    <th style="width:41%; text-align:left; border-bottom:1px solid #444; padding:8px;">{policy_b_name}</th>
  </tr>
"""
    for feat, va, vb in rows:
        table_html += f"""
  <tr>
    <td style="vertical-align:top; border-bottom:1px solid #333; padding:8px; white-space:normal; word-break:break-word;">{feat}</td>
    <td style="vertical-align:top; border-bottom:1px solid #333; padding:8px; white-space:normal; word-break:break-word;">{va}</td>
    <td style="vertical-align:top; border-bottom:1px solid #333; padding:8px; white-space:normal; word-break:break-word;">{vb}</td>
  </tr>
"""
    table_html += "</table>"
    return table_html


def _gemini_generate_json(client, model: str, system: str, user_obj: Dict[str, Any], temperature: float) -> str:
    resp = client.models.generate_content(
        model=model,
        contents=json.dumps(user_obj, ensure_ascii=False),
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
        ),
    )
    return (getattr(resp, "text", "") or "").strip()


def _gemini_generate_text(client, model: str, system: str, user_obj: Dict[str, Any], temperature: float) -> str:
    resp = client.models.generate_content(
        model=model,
        contents=json.dumps(user_obj, ensure_ascii=False),
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
        ),
    )
    return (getattr(resp, "text", "") or "").strip()


# ---------- core: build structured summary (cached) ----------
def build_policy_summary(
    policy_name: str,
    store_path: str,
    api_key: Optional[str],
    force: bool = False,
) -> Dict[str, Any]:
    """
    Build (or load cached) structured summary for a policy.
    The summary is extracted ONLY from retrieved evidence.

    Production rules:
    - If value is not explicit -> missing
    - Never guess numeric values
    - Placeholder/template values -> missing
    """
    path = _summary_path(policy_name)
    if (not force) and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        return _normalize_placeholders_in_summary(loaded)

    # retrieve evidence per field (multi-query + hybrid search)
    evidence_by_field: Dict[str, List[Dict[str, Any]]] = {}
    for field, qs in FIELD_QUERIES.items():
        ev = retrieve_evidence(
            store_path=store_path,
            queries=qs,
            api_key=api_key,
            dense_top_k=12,
            bm25_top_k=8,
            final_k=COMPARE_TOP_K,
        )
        evidence_by_field[field] = ev

    # Compact evidence to control prompt size
    compact_evidence_by_field: Dict[str, List[Dict[str, Any]]] = {}
    for field, ev_list in evidence_by_field.items():
        compact_evidence_by_field[field] = []
        for ev in ev_list:
            meta = ev.get("metadata", {}) or {}
            compact_evidence_by_field[field].append(
                {
                    "doc_name": meta.get("doc_name", "unknown"),
                    "page_start": meta.get("page_start", -1),
                    "page_end": meta.get("page_end", -1),
                    "text": (ev.get("text", "") or "")[:1800],
                }
            )

    system = (
        "You are an expert insurance policy analyst.\n"
        "Extract a structured policy summary.\n\n"
        "STRICT RULES:\n"
        "1) Use ONLY the provided evidence.\n"
        "2) If a value is not explicitly stated, set value=null and status='missing'.\n"
        "3) Do NOT guess or infer ANY numeric values (limits, deductibles, premiums).\n"
        "4) IMPORTANT: If the value appears as a placeholder/template (e.g., '$000', '000', '$XXX', 'TBD', "
        "'see declarations', 'refer to schedule'), treat it as missing.\n"
        "5) You MAY set status='inferred' only for NON-numeric qualitative points (e.g., 'requires prompt notice'), "
        "and only if evidence clearly implies it.\n"
        "6) For each field, include up to 3 evidence snippets (doc+pages+short quote) in evidence[].\n"
        "7) Output MUST be valid JSON ONLY.\n"
    )

    user_obj = {
        "task": "build_policy_summary",
        "policy_name": policy_name,
        "fields": ["coverage_limits", "deductibles", "exclusions", "claim_conditions", "premium"],
        "evidence_by_field": compact_evidence_by_field,
        "output_schema": {
            "policy_name": "string",
            "fields": {
                "coverage_limits": {"value": "string|null", "status": "found|inferred|missing", "evidence": "string[]"},
                "deductibles": {"value": "string|null", "status": "found|inferred|missing", "evidence": "string[]"},
                "exclusions": {"value": "string|null", "status": "found|inferred|missing", "evidence": "string[]"},
                "claim_conditions": {"value": "string|null", "status": "found|inferred|missing", "evidence": "string[]"},
                "premium": {"value": "string|null", "status": "found|inferred|missing", "evidence": "string[]"},
            },
        },
        "formatting_guidance": {
            "coverage_limits": "Summarize major limits in 1-4 lines. If multiple coverages, separate with semicolons.",
            "deductibles": "List deductible amounts and which coverage they apply to. If not explicit or placeholder -> missing.",
            "exclusions": "List key exclusions as bullets in a single string OR semicolon-separated.",
            "claim_conditions": "Summarize key duties: prompt notice, cooperate, documentation, etc.",
            "premium": "If premium numbers are not explicit or placeholder -> missing. Do not guess.",
        },
    }

    client = core._openai_client(api_key)
    text = _gemini_generate_json(
        client=client,
        model=core.CHAT_MODEL,
        system=system,
        user_obj=user_obj,
        temperature=0.1,
    )

    # Parse JSON with minimal salvage
    try:
        summary = json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            summary = json.loads(text[start : end + 1])
        else:
            summary = {"policy_name": policy_name, "fields": {}, "raw": text}

    if not isinstance(summary, dict):
        summary = {"policy_name": policy_name, "fields": {}}

    summary.setdefault("policy_name", policy_name)
    fields = summary.get("fields", {})
    if not isinstance(fields, dict):
        fields = {}

    norm_fields = {}
    for k in ["coverage_limits", "deductibles", "exclusions", "claim_conditions", "premium"]:
        norm_fields[k] = _ensure_field_obj(fields.get(k, {}))

    summary = {"policy_name": policy_name, "fields": norm_fields}
    summary = _normalize_placeholders_in_summary(summary)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


# ---------- compare summaries ----------
def compare_policies_prod(
    policy_a_name: str,
    policy_a_store: str,
    policy_b_name: str,
    policy_b_store: str,
    question: str,
    api_key: Optional[str],
    force_refresh_summaries: bool = False,
) -> str:
    sum_a = build_policy_summary(policy_a_name, policy_a_store, api_key=api_key, force=force_refresh_summaries)
    sum_b = build_policy_summary(policy_b_name, policy_b_store, api_key=api_key, force=force_refresh_summaries)

    fa = sum_a.get("fields", {}) or {}
    fb = sum_b.get("fields", {}) or {}

    rows = [
        ("Coverage limits", _render_value(_ensure_field_obj(fa.get("coverage_limits"))), _render_value(_ensure_field_obj(fb.get("coverage_limits")))),
        ("Deductibles", _render_value(_ensure_field_obj(fa.get("deductibles"))), _render_value(_ensure_field_obj(fb.get("deductibles")))),
        ("Key exclusions", _render_value(_ensure_field_obj(fa.get("exclusions"))), _render_value(_ensure_field_obj(fb.get("exclusions")))),
        ("Claim conditions", _render_value(_ensure_field_obj(fa.get("claim_conditions"))), _render_value(_ensure_field_obj(fb.get("claim_conditions")))),
        ("Premium", _render_value(_ensure_field_obj(fa.get("premium"))), _render_value(_ensure_field_obj(fb.get("premium")))),
    ]

    table_html = _build_fixed_width_table_html(policy_a_name, policy_b_name, rows)

    missing_a = _missing_fields(sum_a)
    missing_b = _missing_fields(sum_b)

    system = (
        "You are an expert insurance analyst.\n\n"
        "STRICT RULES:\n"
        "1) Use ONLY the provided summaries and the provided HTML comparison table.\n"
        "2) Do NOT invent any missing values. If a value is 'Missing' in the table, treat it as unknown.\n"
        "3) If any key fields are missing, explicitly say a definitive recommendation cannot be made.\n"
        "4) If a value appears as '$000', '000', '$XXX', 'TBD', or refers to 'see declarations'/'see schedule', "
        "explain that this likely indicates a template placeholder and the real value is not specified.\n"
        "5) Follow the OUTPUT FORMAT TEMPLATE exactly. Do not remove or reorder headings.\n"
        "6) Keep the direct answer to ONE paragraph.\n"
        "7) Key Differences must be exactly 3 bullets.\n"
        "8) Missing Info Checklist must mention missing coverage limits/deductibles/premium if missing.\n"
        "9) Return the HTML table exactly as provided.\n"
    )

    output_format_template = f"""
## Comparison Result

{{DIRECT_ANSWER_ONE_PARAGRAPH}}

### Here is a comparison of the available information:
{table_html}

## Key Differences
- {{DIFF_1}}
- {{DIFF_2}}
- {{DIFF_3}}

## Missing Info Checklist
- {policy_a_name}: {{MISSING_A}}
- {policy_b_name}: {{MISSING_B}}
""".strip()

    user_obj = {
        "task": "compare_two_policies",
        "question": question,
        "policy_a_name": policy_a_name,
        "policy_b_name": policy_b_name,
        "comparison_table_html": table_html,
        "summaries": {"policy_a": sum_a, "policy_b": sum_b},
        "missing": {"policy_a": missing_a, "policy_b": missing_b},
        "output_format_template": output_format_template,
        "notes": {
            "missing_a": missing_a,
            "missing_b": missing_b,
            "table_values": rows,
        },
    }

    client = core._openai_client(api_key)
    text = _gemini_generate_text(
        client=client,
        model=core.CHAT_MODEL,
        system=system,
        user_obj=user_obj,
        temperature=0.2,
    )

    if "## Comparison Result" not in text or "### Here is a comparison of the available information:" not in text:
        missing_a_str = ", ".join(missing_a) if missing_a else "None"
        missing_b_str = ", ".join(missing_b) if missing_b else "None"
        text = (
            "## Comparison Result\n\n"
            "To determine which policy you should get, key information (e.g., coverage limits, deductibles, and/or premium) "
            "is missing for one or both policies, so a definitive recommendation cannot be made from the available evidence.\n\n"
            "### Here is a comparison of the available information:\n"
            f"{table_html}\n\n"
            "## Key Differences\n"
            "- Differences cannot be fully determined because key fields are Missing.\n"
            "- Provide Declarations/Coverages pages to confirm actual limits, deductibles, and premium.\n"
            "- Exclusions and claim conditions may still differ; confirm with complete policy wording.\n\n"
            "## Missing Info Checklist\n"
            f"- {policy_a_name}: Missing fields: {missing_a_str}\n"
            f"- {policy_b_name}: Missing fields: {missing_b_str}\n"
        )

    return text