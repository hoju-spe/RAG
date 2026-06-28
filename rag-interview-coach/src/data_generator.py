import json
from pathlib import Path

from openai import OpenAI


GENERATION_PROMPT = """
다음 형식의 면접 학습 데이터를 하나 생성하세요.

형식(JSON):
{
  "instruction": "면접 질문",
  "input": "원본 답변",
  "output": "STAR 기반 개선 답변"
}

조건:
- 질문은 실제 면접에서 자주 나오는 질문일 것
- 원본 답변은 부족하거나 짧거나 구체적이지 않은 답변일 것
- output은 원본 답변만을 기반으로 STAR 구조로 확장할 것
- output에 새로운 경험을 창작하지 말 것
- 반드시 JSON만 반환할 것
"""


def parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)


def generate_sample(client: OpenAI, model: str = "gpt-4o-mini") -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": GENERATION_PROMPT}],
        temperature=0.6,
    )
    text = response.choices[0].message.content.strip()
    return parse_json_response(text)


def generate_dataset(
    output_path: str = "data/training_data.jsonl",
    count: int = 100,
    model: str = "gpt-4o-mini",
) -> list[dict]:
    client = OpenAI()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    dataset: list[dict] = []
    for index in range(count):
        print(f"{index + 1}/{count} generating...")
        try:
            item = generate_sample(client, model=model)
            dataset.append(item)
        except Exception as exc:
            print(f"skip: {exc}")

    with output.open("w", encoding="utf-8") as file:
        for item in dataset:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    return dataset


if __name__ == "__main__":
    generate_dataset()
