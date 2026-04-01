import json
import core
from google.genai import types
from prod_compare import build_policy_summary

DIMENSIONS = [
    "Coverage Breadth",
    "Cost & Deductibles",
    "Exclusions",
    "Claim Flexibility",
    "Overall Value"
]


def build_radar_chart():
    # Dummy function to satisfy the import from app_1.py
    # The actual radar chart rendering is handled by plotly in the main UI file.
    pass


def compare_policies_rag(policy_a_name, policy_a_store, policy_b_name, policy_b_store, api_key):
    """
    Uses the existing production RAG pipeline to build summaries,
    then extracts the specific JSON structure needed for the UI dashboard.
    """
    # 1. Generate or load cached structured summaries from your existing backend
    sum_a = build_policy_summary(policy_a_name, policy_a_store, api_key=api_key)
    sum_b = build_policy_summary(policy_b_name, policy_b_store, api_key=api_key)

    # 2. Extract into UI-friendly JSON format
    system = (
        "You are an expert insurance analyst. Compare the two policy summaries provided.\n"
        "You MUST output ONLY valid JSON matching this exact schema:\n"
        "{\n"
        '  "overall_score_a": float (1.0 to 10.0),\n'
        '  "overall_score_b": float (1.0 to 10.0),\n'
        '  "overall_winner": "A", "B", or "Tie",\n'
        '  "overall_winner_reason": "string (1-2 sentences explaining why)",\n'
        '  "dimension_scores": {\n'
        '    "Coverage Breadth": {"a": int, "b": int},\n'
        '    "Cost & Deductibles": {"a": int, "b": int},\n'
        '    "Exclusions": {"a": int, "b": int},\n'
        '    "Claim Flexibility": {"a": int, "b": int},\n'
        '    "Overall Value": {"a": int, "b": int}\n'
        '  },\n'
        '  "category_winners": {\n'
        '     "Coverage Breadth": "A" or "B" or "Tie",\n'
        '     "Cost & Deductibles": "A" or "B" or "Tie",\n'
        '     "Exclusions": "A" or "B" or "Tie",\n'
        '     "Claim Flexibility": "A" or "B" or "Tie",\n'
        '     "Overall Value": "A" or "B" or "Tie"\n'
        '  },\n'
        '  "best_for": {\n'
        '     "Budget-conscious users": "A" or "B",\n'
        '     "Comprehensive coverage seekers": "A" or "B"\n'
        '  },\n'
        '  "a_advantages": ["string", "string"],\n'
        '  "b_advantages": ["string", "string"]\n'
        "}\n"
    )
    user_msg = {"policy_a": sum_a, "policy_b": sum_b}

    client = core._openai_client(api_key)
    resp = client.models.generate_content(
        model=core.CHAT_MODEL,
        contents=json.dumps(user_msg, ensure_ascii=False),
        config=types.GenerateContentConfig(system_instruction=system, temperature=0.1)
    )

    text = (getattr(resp, "text", "") or "").strip()
    try:
        start = text.find("{")
        end = text.rfind("}")
        return json.loads(text[start:end + 1])
    except Exception:
        # Fallback if parsing fails
        return {
            "overall_score_a": 5, "overall_score_b": 5, "overall_winner": "Tie",
            "overall_winner_reason": "Failed to parse comparison data.",
            "dimension_scores": {d: {"a": 5, "b": 5} for d in DIMENSIONS},
            "category_winners": {d: "Tie" for d in DIMENSIONS},
            "best_for": {"General Use": "Tie"}, "a_advantages": [], "b_advantages": []
        }