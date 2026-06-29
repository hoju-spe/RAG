import os

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

from src.answer_improver import improve_answer_star
from src.evaluator import evaluate_answer, format_evaluation_markdown
from src.personal_context import (
    format_personal_context,
    get_personal_collection,
    retrieve_personal_context,
    save_personal_context,
)
from src.retriever import format_retrieved_examples, retrieve_similar_examples
from src.sentiment_analyzer import analyze_tone_with_openai
from src.vector_store import get_collection, seed_vector_store


load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "interview_data")
SAMPLE_DATA_PATH = os.getenv("TRAINING_DATA_PATH", "data/sample_training_data_50.jsonl")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
example_collection = get_collection(CHROMA_PATH, COLLECTION_NAME)
personal_collection = get_personal_collection(CHROMA_PATH)


def ensure_sample_data() -> str:
    count = seed_vector_store(
        jsonl_path=SAMPLE_DATA_PATH,
        chroma_path=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
    )
    return f"{count}개의 기본 면접 사례를 ChromaDB에 저장했습니다."


def save_user_context(project_experience: str, job_info: str) -> str:
    count = save_personal_context(personal_collection, project_experience, job_info)
    if count == 0:
        return "저장할 프로젝트 경험 또는 직무 정보를 입력해주세요."
    return f"{count}개의 개인 컨텍스트를 ChromaDB에 저장했습니다."


def run_coach(
    project_experience: str,
    job_info: str,
    question: str,
    answer: str,
    top_k: int,
):
    if project_experience.strip() or job_info.strip():
        save_personal_context(personal_collection, project_experience, job_info)

    if not question.strip() or not answer.strip():
        return (
            "질문과 답변을 입력해주세요.",
            "평가 대기 중",
            "검색 대기 중",
            "개인 컨텍스트 대기 중",
            "분석 대기 중",
        )

    examples = retrieve_similar_examples(example_collection, question, answer, top_k=top_k)
    personal_contexts = retrieve_personal_context(personal_collection, question, answer, top_k=top_k)

    retrieved_context = format_retrieved_examples(examples)
    personal_context_text = format_personal_context(personal_contexts)

    improved_answer = improve_answer_star(
        client=client,
        model=OPENAI_MODEL,
        question=question,
        answer=answer,
        examples=examples,
        personal_contexts=personal_contexts,
    )

    evaluation = evaluate_answer(
        client=client,
        model=OPENAI_MODEL,
        question=question,
        original_answer=answer,
        improved_answer=improved_answer,
    )
    evaluation_markdown = format_evaluation_markdown(evaluation)
    tone_analysis = analyze_tone_with_openai(client, OPENAI_MODEL, answer)

    return improved_answer, evaluation_markdown, retrieved_context, personal_context_text, tone_analysis


with gr.Blocks(title="PICO - 개인화 RAG 기반 AI 면접 답변 코치") as demo:
    gr.Markdown("# PICO - 개인화 RAG 기반 AI 면접 답변 코치")
    gr.Markdown(
        "지원자의 프로젝트 경험과 지원 직무 정보를 먼저 저장한 뒤, "
        "면접 질문에 맞는 개인 근거와 STAR 참고 사례를 함께 검색해 답변을 개선합니다."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 1. 개인 컨텍스트 입력")
            project_input = gr.Textbox(
                label="프로젝트 경험",
                lines=8,
                placeholder=(
                    "예: Maily 프로젝트에서 Gmail API 연동, JWT/OAuth2 인증, "
                    "RabbitMQ 기반 비동기 처리, 관리자 모니터링 기능을 구현했습니다."
                ),
            )
            job_info_input = gr.Textbox(
                label="지원 직무 정보",
                lines=6,
                placeholder=(
                    "예: Java/Spring Boot 기반 백엔드 개발, SQL/RDBMS, "
                    "Docker, AWS, 운영/개선 경험을 요구하는 직무입니다."
                ),
            )
            save_context_button = gr.Button("프로젝트/직무 정보 저장")
            context_status = gr.Textbox(label="개인 컨텍스트 상태", interactive=False)

            gr.Markdown("## 2. 기본 RAG 데이터")
            seed_button = gr.Button("기본 샘플 사례 ChromaDB 저장")
            seed_status = gr.Textbox(label="샘플 데이터 상태", interactive=False)

            gr.Markdown("## 3. 면접 답변 첨삭")
            question_input = gr.Textbox(
                label="면접 질문",
                lines=2,
                placeholder="예: Docker를 사용해본 경험이 있나요?",
            )
            answer_input = gr.Textbox(
                label="나의 답변",
                lines=8,
                placeholder="현재 답변을 입력하세요.",
            )
            top_k_input = gr.Slider(
                label="참고 근거 개수",
                minimum=1,
                maximum=5,
                value=3,
                step=1,
            )
            run_button = gr.Button("분석 및 첨삭 받기", variant="primary")

        with gr.Column(scale=1):
            improved_output = gr.Textbox(label="AI 수정 답변 (STAR)", lines=12)
            evaluation_output = gr.Markdown(label="답변 평가")

    with gr.Row():
        retrieved_output = gr.Textbox(label="일반 STAR 참고 사례 Top K", lines=12)
        personal_output = gr.Textbox(label="검색된 개인 프로젝트/직무 근거", lines=12)
        tone_output = gr.Textbox(label="면접 답변 톤 분석", lines=12)

    save_context_button.click(
        fn=save_user_context,
        inputs=[project_input, job_info_input],
        outputs=context_status,
    )
    seed_button.click(fn=ensure_sample_data, outputs=seed_status)
    run_button.click(
        fn=run_coach,
        inputs=[project_input, job_info_input, question_input, answer_input, top_k_input],
        outputs=[
            improved_output,
            evaluation_output,
            retrieved_output,
            personal_output,
            tone_output,
        ],
    )


if __name__ == "__main__":
    demo.launch()
