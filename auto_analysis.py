"""
auto_analysis.py — PolicyPal v3
Uses Google Gemini via google-genai SDK.
"""
import json
import io
import pdfplumber
from google import genai
from google.genai import types  # <--- 确保加上这一行

GEMINI_MODEL = "gemini-2.5-flash"


def _client(api_key: str):
    return genai.Client(
        api_key=api_key,
        http_options={"api_version": "v1"}
    )


def extract_pdf_text(uploaded_file, max_chars: int = 20000) -> str:
    text_parts = []
    try:
        data = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as e:
        return f"[PDF extraction error: {e}]"
    return "\n".join(text_parts)[:max_chars]


def analyze_policy_document(text: str, api_key: str) -> dict:
    client = _client(api_key)

    prompt = f"""You are a licensed insurance advisor. Analyze this insurance policy and extract key information.

    Return ONLY a valid JSON object — no markdown fences, no preamble, no extra text.

    Schema:
    {{
      "policy_type": "Health | Auto | Home | Life | Renters | Other",
      "insurer": "Company name or Unknown",
      "deductible": "e.g. $1,500 or Not found",
      "annual_premium": "e.g. $2,400/yr or Not found",
      "monthly_premium": "e.g. $200/mo or Not found",
      "out_of_pocket_max": "e.g. $7,000 or Not found",
      "coverage_limit": "e.g. $500,000 or Not found",
      "coverage_areas": {{"AreaName": integer_percentage}},
      "key_benefits": ["Up to 5 specific benefits"],
      "exclusions": ["Up to 6 exclusions"],
      "risk_flags": ["Up to 3 serious gaps"],
      "risk_score": integer_1_to_10,
      "risk_explanation": "1-2 sentences",
      "plain_summary": "2-3 sentences plain English",
      "who_its_good_for": "1 sentence",
      "potential_savings": "Specific tip or None identified"
    }}

    Coverage areas must sum to 100.

    SCORING RUBRIC FOR 'risk_score' (1-10, where 10 is the BEST/SAFEST policy):
    Start at a base score of 10, then strictly apply these deductions based on the text:
    - Deduct 1 point if the deductible is unusually high or vaguely defined.
    - Deduct 1 point if "out_of_pocket_max" is missing or very high.
    - Deduct 2 points if there are 3 or more strict exclusions ("risk_flags").
    - Deduct 1 point if coverage limits are ambiguous or "TBD".
    (Minimum score is 1. If it's a solid, transparent policy, it should score 8-10).

    POLICY TEXT:
    {text}"""

    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,  # 强制设为 0，消除模型回答的随机性
            )
        )
        raw = resp.text.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {type(e).__name__}: {e}") from e


def ask_policy_question(
    question: str,
    policy_text: str,
    api_key: str,
    chat_history: list = None,
) -> str:
    client = _client(api_key)

    history_text = ""
    if chat_history:
        for msg in chat_history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n\n"

    full_prompt = f"""You are PolicyPal, a friendly insurance assistant. Answer ONLY based on the policy document below.
Rules:
1. Only state facts from the policy. Say so clearly if not mentioned.
2. Cite sections when possible.
3. Use plain English.
4. For scenario questions, reason step by step.
5. End scenario answers with: I recommend confirming directly with your insurer.

POLICY DOCUMENT:
{policy_text[:12000]}

{history_text}User: {question}
Assistant:"""

    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt,
        )
        return resp.text
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {type(e).__name__}: {e}") from e

