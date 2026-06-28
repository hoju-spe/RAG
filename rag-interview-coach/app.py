import os

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

from src.answer_improver import improve_answer_star
from src.evaluator import evaluate_answer, format_evaluation_markdown
from src.retriever import format_retrieved_examples, retrieve_similar_examples
from src.sentiment_analyzer import analyze_tone_with_openai
from src.vector_store import get_collection, seed_vector_store


load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "interview_data")
SAMPLE_DATA_PATH = "data/sample_training_data.jsonl"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
collection = get_collection(CHROMA_PATH, COLLECTION_NAME)


def ensure_sample_data() -> str:
    count = seed_vector_store(
        jsonl_path=SAMPLE_DATA_PATH,
        chroma_path=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
    )
    return f"{count}개의 샘플 면접 사례를 ChromaDB에 저장했습니다."


def run_coach(question: str, answer: str, top_k: int):
    if not question.strip() or not answer.strip():
        return (
            "질문과 답변을 입력해주세요.",
            "평가 대기 중",
            "검색 대기 중",
            "분석 대기 중",
        )

    examples = retrieve_similar_examples(collection, question, answer, top_k=top_k)
    retrieved_context = format_retrieved_examples(examples)

    improved_answer = improve_answer_star(
        client=client,
        model=OPENAI_MODEL,
        question=question,
        answer=answer,
        examples=examples,
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

    return improved_answer, evaluation_markdown, retrieved_context, tone_analysis


with gr.Blocks(title="PICO - RAG 기반 AI 면접 답변 코치") as demo:
    gr.Markdown("# PICO - RAG 기반 AI 면접 답변 코치")
    gr.Markdown(
        "사용자의 면접 질문과 답변을 입력받아 유사한 STAR 답변 사례를 검색하고, "
        "검색 결과를 기반으로 답변을 개선합니다."
    )

    with gr.Row():
        with gr.Column(scale=1):
            seed_button = gr.Button("샘플 데이터 ChromaDB 저장")
            seed_status = gr.Textbox(label="데이터 상태", interactive=False)
            question_input = gr.Textbox(
                label="면접 질문",
                lines=2,
                placeholder="예: 본인의 강점은 무엇인가요?",
            )
            answer_input = gr.Textbox(
                label="나의 답변",
                lines=8,
                placeholder="현재 답변을 입력하세요.",
            )
            top_k_input = gr.Slider(
                label="참고 사례 개수",
                minimum=1,
                maximum=5,
                value=3,
                step=1,
            )
            run_button = gr.Button("분석 및 첨삭 받기", variant="primary")

        with gr.Column(scale=1):
            improved_output = gr.Textbox(label="AI 수정 답변 (STAR)", lines=10)
            evaluation_output = gr.Markdown(label="답변 평가")

    with gr.Row():
        retrieved_output = gr.Textbox(label="참고한 유사 답변 사례 Top K", lines=14)
        tone_output = gr.Textbox(label="면접 답변 톤 분석", lines=14)

    seed_button.click(fn=ensure_sample_data, outputs=seed_status)
    run_button.click(
        fn=run_coach,
        inputs=[question_input, answer_input, top_k_input],
        outputs=[improved_output, evaluation_output, retrieved_output, tone_output],
    )


if __name__ == "__main__":
    demo.launch()
