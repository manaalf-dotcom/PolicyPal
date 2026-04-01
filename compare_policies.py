"""
compare_policies.py — PolicyPal v3
Uses Google Gemini via the official google-genai SDK.
"""
import json
from google import genai
import plotly.graph_objects as go

GEMINI_MODEL = "gemini-2.5-flash"

DIMENSIONS = [
    "Coverage Completeness",
    "Affordability",
    "Flexibility",
    "Exclusion Risk",
    "Ease of Claims",
    "Overall Value",
]


def _client(api_key: str):
    return genai.Client(api_key=api_key, http_options={"api_version": "v1"})


def compare_policies_llm(analysis_a: dict, analysis_b: dict, api_key: str) -> dict:
    client = _client(api_key)

    prompt = f"""You are an expert, impartial insurance advisor comparing two policies side by side.

POLICY A SUMMARY:
{json.dumps(analysis_a, indent=2)}

POLICY B SUMMARY:
{json.dumps(analysis_b, indent=2)}

CRITICAL RULES FOR ALIGNMENT:
1. Read the 'status' and 'evidence' for key fields (coverage_limits, deductibles, premium) in both summaries carefully.
2. STRICT RULE: If ANY key financial field has status="missing" or contains placeholder values (like $000, XXX, TBD, "see declarations") in EITHER policy, you MUST NOT declare A or B as the overall winner.
3. If crucial information is missing:
   - You MUST set "overall_winner" to "Tie".
   - You MUST set "overall_score_a" and "overall_score_b" to the exact same conservative number (e.g., 5.0).
   - In "overall_winner_reason", you MUST explain that "A definitive recommendation cannot be made due to missing or placeholder financial information (limits, deductibles, or premiums). The actual Declarations page is required for a true comparison."
4. You can still evaluate the 'dimension_scores' (Coverage Completeness, Exclusion Risk, etc.) based on the text that IS available, but the overall result must remain a Tie if limits/premiums are unknown.

Return ONLY a valid JSON object — no markdown fences, no extra text — using this exact schema:
{{
  "dimension_scores": {{
    "Coverage Completeness": {{"a": 1_to_10, "b": 1_to_10}},
    "Affordability":          {{"a": 1_to_10, "b": 1_to_10}},
    "Flexibility":            {{"a": 1_to_10, "b": 1_to_10}},
    "Exclusion Risk":         {{"a": 1_to_10, "b": 1_to_10}},
    "Ease of Claims":         {{"a": 1_to_10, "b": 1_to_10}},
    "Overall Value":          {{"a": 1_to_10, "b": 1_to_10}}
  }},
  "category_winners": {{
    "Coverage Completeness": "A or B or Tie",
    "Affordability": "A or B or Tie",
    "Flexibility": "A or B or Tie",
    "Exclusion Risk": "A or B or Tie",
    "Ease of Claims": "A or B or Tie",
    "Overall Value": "A or B or Tie"
  }},
  "overall_winner": "A or B or Tie",
  "overall_score_a": 1_to_10,
  "overall_score_b": 1_to_10,
  "overall_winner_reason": "2-3 plain-English sentences explaining the decision OR explaining why it's a Tie due to missing info.",
  "best_for": {{
    "Young and healthy": "A or B or Tie",
    "Families with children": "A or B or Tie",
    "Chronic conditions": "A or B or Tie",
    "Budget-conscious": "A or B or Tie"
  }},
  "key_tradeoffs": ["3 specific tradeoffs between the two plans based on available text"],
  "a_advantages": ["3 specific advantages of Policy A"],
  "b_advantages": ["3 specific advantages of Policy B"],
  "red_flag_a": "Single most important concern about Policy A, or null",
  "red_flag_b": "Single most important concern about Policy B, or null"
}}"""

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    raw = resp.text.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


def build_radar_chart(comparison: dict, name_a: str = "Policy A", name_b: str = "Policy B") -> go.Figure:
    dims = DIMENSIONS
    scores_a = [comparison["dimension_scores"][d]["a"] for d in dims]
    scores_b = [comparison["dimension_scores"][d]["b"] for d in dims]
    sa = scores_a + [scores_a[0]]
    sb = scores_b + [scores_b[0]]
    dc = dims + [dims[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=sa, theta=dc, fill="toself", name=name_a,
        line=dict(color="#6366F1", width=3), fillcolor="rgba(99,102,241,0.25)",
        marker=dict(size=7, color="#6366F1")))
    fig.add_trace(go.Scatterpolar(r=sb, theta=dc, fill="toself", name=name_b,
        line=dict(color="#06B6D4", width=3), fillcolor="rgba(6,182,212,0.2)",
        marker=dict(size=7, color="#06B6D4")))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], tickvals=[2,4,6,8,10],
                            tickfont=dict(size=10, color="#7B6FA0"),
                            gridcolor="rgba(255,255,255,0.08)",
                            linecolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(tickfont=dict(size=13, color="#C4B5FD")),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5,
                    font=dict(size=13, color="#C4B5FD"), bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=40, t=20, b=80), height=450,
    )
    return fig