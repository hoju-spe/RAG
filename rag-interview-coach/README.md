# PICO - RAG 기반 AI 면접 답변 코치

PICO는 사용자의 면접 답변을 입력받아 유사한 STAR 답변 사례를 검색하고, GPT 모델을 활용해 답변을 STAR 구조로 개선하는 RAG 기반 면접 코칭 서비스입니다.

기존 `RAG_fixed.ipynb`에 들어 있던 데이터 생성, ChromaDB 저장, 검색, GPT 호출, 톤 분석, Gradio UI를 프로젝트 구조로 분리했습니다.

## 주요 기능

- 면접 질문과 사용자 답변 입력
- ChromaDB 기반 유사 STAR 답변 사례 검색
- 검색 결과 Top K를 UI에 표시
- RAG 참고 사례를 프롬프트에 주입해 STAR 구조 답변 생성
- 개선 전/후 답변 평가 점수 출력
- 면접 답변 톤 분석
- Gradio 기반 웹 UI 제공
- Qwen 로컬 LLM 톤 분석 모듈 선택 제공

## 기술 스택

- Python
- OpenAI API
- ChromaDB
- Gradio
- python-dotenv
- Optional: Qwen2.5, Transformers, BitsAndBytes, PyTorch

## 시스템 구조

```txt
rag-interview-coach/
  app.py
  requirements.txt
  .env.example
  README.md
  src/
    data_generator.py
    vector_store.py
    retriever.py
    answer_improver.py
    evaluator.py
    sentiment_analyzer.py
  data/
    sample_training_data.jsonl
```

## RAG 처리 흐름

```txt
사용자 질문/답변 입력
→ 질문 + 사용자 답변을 검색 쿼리로 구성
→ ChromaDB에서 유사 STAR 답변 사례 Top K 검색
→ 검색 결과를 GPT 프롬프트에 주입
→ STAR 구조 답변 생성
→ 개선 전/후 답변 평가 점수 생성
→ 면접 답변 톤 분석
→ Gradio UI 출력
```

## 실행 방법

### 1. 가상환경 생성

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. 환경변수 설정

`.env.example`을 참고해 `.env` 파일을 생성합니다.

```txt
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
CHROMA_PATH=./chroma_db
COLLECTION_NAME=interview_data
```

### 4. 앱 실행

```bash
python app.py
```

실행 후 브라우저에서 Gradio 로컬 주소에 접속합니다.

### 5. 샘플 데이터 저장

앱 화면에서 `샘플 데이터 ChromaDB 저장` 버튼을 먼저 누르면 `data/sample_training_data.jsonl`의 예시 데이터가 ChromaDB에 저장됩니다.

## Qwen 로컬 톤 분석

기본 앱은 실행 편의성을 위해 OpenAI API 기반 톤 분석을 사용합니다.

`src/sentiment_analyzer.py`에는 Qwen2.5-3B-Instruct 기반 로컬 톤 분석 클래스도 분리해두었습니다. 이 기능은 일반 로컬 PC보다 Colab T4 같은 GPU 환경에서 사용하는 것을 권장합니다.

Optional dependencies:

```bash
pip install transformers accelerate bitsandbytes torch
```

## 기존 노트북 대비 개선점

- API 키를 코드에서 제거하고 환경변수로 관리
- 하나의 노트북에 섞여 있던 기능을 모듈별로 분리
- 질문뿐 아니라 사용자 답변까지 포함해 RAG 검색 쿼리 개선
- 검색된 참고 사례 Top K를 UI에 표시
- 개선 전/후 평가 점수 기능 추가
- Qwen 로컬 모델을 기본 실행 경로에서 분리
- Gradio 앱을 독립 실행 가능한 `app.py`로 구성

## 개선 방향

- 실제 면접 답변 데이터셋 확장
- 검색 품질 평가 지표 추가
- 답변 개선 결과 저장 기능 추가
- 사용자별 답변 히스토리 관리
- FastAPI 백엔드와 React 프론트엔드로 확장
- 배포 환경에서 ChromaDB 영속성 관리
