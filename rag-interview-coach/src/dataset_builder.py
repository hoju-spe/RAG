import argparse
import json
from pathlib import Path

from openai import OpenAI

from src.data_generator import parse_json_response


PROMPT_TEMPLATE = """
아래 면접 질문을 기반으로 RAG 학습용 샘플 데이터를 하나 생성하세요.

면접 질문:
{question}

반환 형식(JSON):
{{
  "instruction": "면접 질문",
  "input": "부족한 원본 답변",
  "output": "STAR 기반 개선 답변"
}}

조건:
- instruction은 제공된 면접 질문을 그대로 사용하세요.
- input은 실제 지원자가 말할 법한 짧고 구체성이 부족한 답변으로 작성하세요.
- output은 input에 없는 구체적인 회사명, 수치, 기술명을 임의로 창작하지 말고 STAR 흐름으로 자연스럽게 확장하세요.
- 반드시 JSON만 반환하세요.
"""


def load_questions(path: str) -> list[str]:
    source = Path(path)
    questions = []
    for line in source.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            questions.append(text)
    return questions


def generate_sample_from_question(client: OpenAI, model: str, question: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(question=question),
            }
        ],
        temperature=0.5,
    )
    return parse_json_response(response.choices[0].message.content)


def build_dataset(
    input_path: str = "data/interview_questions_top100.txt",
    output_path: str = "data/training_data_generated.jsonl",
    model: str = "gpt-4o-mini",
    limit: int | None = None,
) -> int:
    client = OpenAI()
    questions = load_questions(input_path)
    if limit is not None:
        questions = questions[:limit]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    created = 0
    with output.open("w", encoding="utf-8") as file:
        for index, question in enumerate(questions, start=1):
            print(f"{index}/{len(questions)} generating: {question}")
            try:
                item = generate_sample_from_question(client, model, question)
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
                created += 1
            except Exception as exc:
                print(f"skip: {question} ({exc})")

    return created


def main():
    parser = argparse.ArgumentParser(description="Build interview RAG training data from question list.")
    parser.add_argument("--input", default="data/interview_questions_top100.txt")
    parser.add_argument("--output", default="data/training_data_generated.jsonl")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    count = build_dataset(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        limit=args.limit,
    )
    print(f"{count} samples written to {args.output}")


if __name__ == "__main__":
    main()
