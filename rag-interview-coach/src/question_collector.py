import argparse
import re
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


QUESTION_PATTERNS = (
    "무엇인가요",
    "무엇입니까",
    "말해주세요",
    "설명해주세요",
    "있나요",
    "했나요",
    "하나요",
    "합니까",
    "?",
)


def normalize_question(text: str) -> str:
    text = re.sub(r"^\s*[\d]+[.)]\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_question_candidate(text: str) -> bool:
    if len(text) < 8 or len(text) > 120:
        return False
    return any(pattern in text for pattern in QUESTION_PATTERNS)


def collect_questions_from_url(url: str) -> list[str]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https URLs are supported.")

    response = requests.get(
        url,
        timeout=10,
        headers={"User-Agent": "PICO-RAG-Interview-Coach/1.0"},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    candidates: list[str] = []
    for raw in soup.get_text("\n").splitlines():
        question = normalize_question(raw)
        if is_question_candidate(question):
            candidates.append(question)

    return dedupe(candidates)


def dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def save_questions(questions: list[str], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(questions) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Collect interview question candidates from allowed web pages.")
    parser.add_argument("urls", nargs="+")
    parser.add_argument("--output", default="data/crawled_questions.txt")
    args = parser.parse_args()

    all_questions: list[str] = []
    for url in args.urls:
        print(f"collecting: {url}")
        all_questions.extend(collect_questions_from_url(url))

    questions = dedupe(all_questions)
    save_questions(questions, args.output)
    print(f"{len(questions)} questions written to {args.output}")


if __name__ == "__main__":
    main()
