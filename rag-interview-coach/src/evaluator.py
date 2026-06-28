import json

from openai import OpenAI


EVALUATION_PROMPT = """
다음 면접 답변을 평가하세요.

평가 기준:
- specificity: 구체성
- star_structure: STAR 구조
- job_relevance: 직무 연관성
- conciseness: 간결성

각 항목은 1~5점으로 평가하고, 반드시 아래 JSON 형식만 반환하세요.

{
  "before": {
    "specificity": 1,
    "star_structure": 1,
    "job_relevance": 1,
    "conciseness": 1
  },
  "after": {
    "specificity": 1,
    "star_structure": 1,
    "job_relevance": 1,
    "conciseness": 1
  },
  "comment": "짧은 총평"
}
"""


LABELS = {
    "specificity": "구체성",
    "star_structure": "STAR 구조",
    "job_relevance": "직무 연관성",
    "conciseness": "간결성",
}


def parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)


def evaluate_answer(
    client: OpenAI,
    model: str,
    question: str,
    original_answer: str,
    improved_answer: str,
) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EVALUATION_PROMPT},
            {
                "role": "user",
                "content": f"""
면접 질문:
{question}

개선 전 답변:
{original_answer}

개선 후 답변:
{improved_answer}
""",
            },
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content.strip()
    return parse_json_response(content)


def format_evaluation_markdown(evaluation: dict) -> str:
    before = evaluation.get("before", {})
    after = evaluation.get("after", {})
    lines = ["| 평가 항목 | 개선 전 | 개선 후 |", "|---|---:|---:|"]

    for key, label in LABELS.items():
        lines.append(f"| {label} | {before.get(key, '-')} | {after.get(key, '-')} |")

    comment = evaluation.get("comment", "")
    if comment:
        lines.append("")
        lines.append(f"총평: {comment}")

    return "\n".join(lines)
