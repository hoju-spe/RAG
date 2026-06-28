from dataclasses import dataclass


@dataclass
class RetrievedExample:
    document: str
    score: float | None = None
    metadata: dict | None = None


def build_query(question: str, answer: str) -> str:
    return f"""
면접 질문: {question}
사용자 답변: {answer}
""".strip()


def retrieve_similar_examples(collection, question: str, answer: str, top_k: int = 3) -> list[RetrievedExample]:
    query = build_query(question, answer)
    results = collection.query(query_texts=[query], n_results=top_k)

    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    examples: list[RetrievedExample] = []
    for index, document in enumerate(documents):
        distance = distances[index] if index < len(distances) else None
        metadata = metadatas[index] if index < len(metadatas) else None
        score = None if distance is None else round(1 - distance, 4)
        examples.append(RetrievedExample(document=document, score=score, metadata=metadata))

    return examples


def format_retrieved_examples(examples: list[RetrievedExample]) -> str:
    if not examples:
        return "검색된 참고 사례가 없습니다."

    formatted = []
    for index, example in enumerate(examples, start=1):
        score_text = "" if example.score is None else f"유사도: {example.score}"
        formatted.append(f"[Top {index}] {score_text}\n{example.document}")

    return "\n\n---\n\n".join(formatted)
