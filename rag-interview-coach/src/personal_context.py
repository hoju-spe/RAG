from dataclasses import dataclass

import chromadb


PERSONAL_COLLECTION_NAME = "personal_context"


@dataclass
class PersonalContext:
    document: str
    score: float | None = None
    metadata: dict | None = None


def get_personal_collection(chroma_path: str = "./chroma_db"):
    client = chromadb.PersistentClient(path=chroma_path)
    return client.get_or_create_collection(
        name=PERSONAL_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build_personal_documents(project_experience: str, job_info: str) -> tuple[list[str], list[dict], list[str]]:
    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    if project_experience.strip():
        documents.append(f"[지원자 프로젝트 경험]\n{project_experience.strip()}")
        metadatas.append({"source": "user_input", "type": "project_experience"})
        ids.append("project_experience")

    if job_info.strip():
        documents.append(f"[지원 직무 정보]\n{job_info.strip()}")
        metadatas.append({"source": "user_input", "type": "job_info"})
        ids.append("job_info")

    return documents, metadatas, ids


def save_personal_context(collection, project_experience: str, job_info: str) -> int:
    documents, metadatas, ids = build_personal_documents(project_experience, job_info)
    if not ids:
        return 0

    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    return len(ids)


def retrieve_personal_context(collection, question: str, answer: str, top_k: int = 3) -> list[PersonalContext]:
    query = f"""
면접 질문: {question}
사용자 답변: {answer}
""".strip()
    results = collection.query(query_texts=[query], n_results=top_k)

    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    contexts: list[PersonalContext] = []
    for index, document in enumerate(documents):
        distance = distances[index] if index < len(distances) else None
        metadata = metadatas[index] if index < len(metadatas) else None
        score = None if distance is None else round(1 - distance, 4)
        contexts.append(PersonalContext(document=document, score=score, metadata=metadata))

    return contexts


def format_personal_context(contexts: list[PersonalContext]) -> str:
    if not contexts:
        return "저장된 개인 프로젝트 경험/직무 정보가 없습니다."

    formatted = []
    for index, context in enumerate(contexts, start=1):
        score_text = "" if context.score is None else f"유사도: {context.score}"
        formatted.append(f"[개인 컨텍스트 {index}] {score_text}\n{context.document}")

    return "\n\n---\n\n".join(formatted)
