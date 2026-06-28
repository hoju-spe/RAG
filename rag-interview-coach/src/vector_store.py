import json
from pathlib import Path

import chromadb


def get_collection(
    chroma_path: str = "./chroma_db",
    collection_name: str = "interview_data",
):
    client = chromadb.PersistentClient(path=chroma_path)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def load_jsonl(path: str) -> list[dict]:
    data_path = Path(path)
    with data_path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def build_documents(data: list[dict]) -> tuple[list[str], list[dict], list[str]]:
    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for index, item in enumerate(data):
        question = item.get("instruction", "")
        weak_answer = item.get("input", "")
        star_answer = item.get("output", "")

        documents.append(
            f"""
[면접 질문] {question}
[짧은 답변] {weak_answer}
[STAR 확장 답변] {star_answer}
""".strip()
        )
        metadatas.append(
            {
                "category": "interview",
                "question": question,
                "weak_answer": weak_answer,
            }
        )
        ids.append(str(index))

    return documents, metadatas, ids


def seed_vector_store(
    jsonl_path: str = "data/sample_training_data.jsonl",
    chroma_path: str = "./chroma_db",
    collection_name: str = "interview_data",
) -> int:
    collection = get_collection(chroma_path, collection_name)
    data = load_jsonl(jsonl_path)
    documents, metadatas, ids = build_documents(data)

    if ids:
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

    return len(ids)


if __name__ == "__main__":
    count = seed_vector_store()
    print(f"{count} examples saved to ChromaDB.")
