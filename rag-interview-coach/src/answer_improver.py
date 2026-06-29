from openai import OpenAI

from src.personal_context import PersonalContext, format_personal_context
from src.retriever import RetrievedExample, format_retrieved_examples


SYSTEM_PROMPT = """
당신은 STAR 구조와 채용 면접 커뮤니케이션에 능숙한 면접 코치입니다.
사용자가 실제로 말한 경험의 범위를 벗어나지 않고 답변을 구체화해야 합니다.
"""


def improve_answer_star(
    client: OpenAI,
    model: str,
    question: str,
    answer: str,
    examples: list[RetrievedExample],
    personal_contexts: list[PersonalContext] | None = None,
) -> str:
    rag_context = format_retrieved_examples(examples)
    personal_context = format_personal_context(personal_contexts or [])
    user_prompt = f"""
면접 질문:
{question}

사용자 답변:
{answer}

[지원자 개인 프로젝트/직무 정보]
{personal_context}

[일반 STAR 참고 사례]
{rag_context}

아래 조건에 맞게 면접 답변을 STAR 구조로 자연스럽게 개선하세요.

조건:
- 경험 창작 금지
- 사용자 답변과 개인 프로젝트/직무 정보에 없는 회사명, 수치, 기술을 임의로 추가하지 말 것
- 개인 프로젝트/직무 정보가 답변과 관련 있으면 일반 사례보다 우선 반영할 것
- 상황(Situation), 과제(Task), 행동(Action), 결과(Result)의 흐름이 드러날 것
- 6~10문장으로 작성
- 면접에서 실제로 말하기 좋은 자연스러운 문장으로 작성
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()
