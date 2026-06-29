# PICO - 개인화 RAG 기반 AI 면접 답변 코치

PICO는 사용자의 프로젝트 경험과 지원 직무 정보를 벡터 DB에 저장하고, 면접 질문에 맞는 개인 근거를 검색해 답변을 STAR 구조로 개선하는 개인화 RAG 기반 면접 코칭 서비스입니다.

기존 `RAG_fixed.ipynb`에 들어 있던 데이터 생성, ChromaDB 저장, 검색, GPT 호출, 톤 분석, Gradio UI를 프로젝트 구조로 분리했습니다.

## 주요 기능

- 프로젝트 경험과 지원 직무 정보 입력
- 입력된 개인 컨텍스트를 ChromaDB에 저장
- 면접 질문과 사용자 답변 입력
- ChromaDB 기반 개인 프로젝트/직무 근거 검색
- ChromaDB 기반 유사 STAR 답변 사례 검색
- 검색 결과 Top K를 UI에 표시
- 개인 근거와 STAR 참고 사례를 프롬프트에 주입해 STAR 구조 답변 생성
- 개선 전/후 답변 평가 점수 출력
- 면접 답변 톤 분석
- Gradio 기반 웹 UI 제공
- 면접 질문 TOP100 기반 보조 학습 데이터 생성 스크립트 제공
- 허용된 웹 페이지에서 질문 후보를 수집하는 크롤링 유틸 제공
- Qwen 로컬 LLM 톤 분석 모듈 선택 제공

## 기술 스택

- Python
- OpenAI API
- ChromaDB
- Gradio
- python-dotenv
- Requests
- BeautifulSoup4
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
    personal_context.py
    answer_improver.py
    evaluator.py
    dataset_builder.py
    question_collector.py
    sentiment_analyzer.py
  data/
    sample_training_data.jsonl
    interview_questions_top100.txt
```

## RAG 처리 흐름

```txt
프로젝트 경험/지원 직무 정보 입력
→ 개인 컨텍스트를 ChromaDB에 저장
→ 사용자 질문/답변 입력
→ 질문 + 사용자 답변을 검색 쿼리로 구성
→ ChromaDB에서 개인 프로젝트/직무 근거 검색
→ ChromaDB에서 유사 STAR 답변 사례 Top K 검색
→ 개인 근거와 검색 사례를 GPT 프롬프트에 주입
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
TRAINING_DATA_PATH=data/sample_training_data.jsonl
```

### 4. 앱 실행

```bash
python app.py
```

실행 후 브라우저에서 Gradio 로컬 주소에 접속합니다.

### 5. 개인 컨텍스트 저장

앱 화면에서 `프로젝트 경험`과 `지원 직무 정보`를 입력한 뒤 `프로젝트/직무 정보 저장` 버튼을 누르면 해당 내용이 `personal_context` 컬렉션에 저장됩니다.

저장된 개인 컨텍스트는 면접 답변 첨삭 시 일반 STAR 참고 사례보다 우선적으로 반영됩니다.

### 6. 샘플 데이터 저장

앱 화면에서 `샘플 데이터 ChromaDB 저장` 버튼을 먼저 누르면 `data/sample_training_data.jsonl`의 예시 데이터가 ChromaDB에 저장됩니다.

## 면접 질문 TOP100 기반 데이터 생성

`data/interview_questions_top100.txt`에는 일반적인 백엔드/개발자 면접 질문 100개를 정리해두었습니다.

이 기능은 메인 기능이 아니라, 추가적인 일반 STAR 참고 사례를 늘리기 위한 보조 데이터 생성 파이프라인입니다. PICO의 핵심은 사용자의 실제 프로젝트 경험과 지원 직무 정보를 검색 근거로 활용하는 개인화 RAG입니다.

아래 명령어를 실행하면 질문 목록을 기반으로 OpenAI API가 `instruction`, `input`, `output` 형식의 RAG 학습 데이터를 생성합니다.

```bash
python -m src.dataset_builder --input data/interview_questions_top100.txt --output data/training_data_generated.jsonl
```

테스트용으로 10개만 생성하려면:

```bash
python -m src.dataset_builder --limit 10
```

생성된 데이터를 ChromaDB에 저장하려면:

```bash
python -m src.vector_store data/training_data_generated.jsonl
```

현재 `app.py`의 기본 버튼은 API 비용 방지를 위해 고정 샘플 5개만 저장합니다. 100개 데이터를 사용하려면 `data/training_data_generated.jsonl`을 생성한 뒤 `.env`의 `TRAINING_DATA_PATH`를 아래처럼 변경하면 됩니다.

```txt
TRAINING_DATA_PATH=data/training_data_generated.jsonl
```

## 질문 후보 크롤링

`src/question_collector.py`는 허용된 웹 페이지에서 질문 후보를 수집해 텍스트 파일로 저장하는 유틸입니다.

```bash
python -m src.question_collector https://example.com/interview-questions --output data/crawled_questions.txt
```

주의:

- 크롤링 전 대상 사이트의 이용약관과 robots.txt를 확인하세요.
- 공개 포트폴리오에는 원문을 그대로 대량 복사하기보다, 질문 후보 수집과 정제 파이프라인을 설명하는 용도로 사용하는 것을 권장합니다.
- 저작권 이슈를 피하려면 직접 정리한 질문 목록이나 생성형 AI로 만든 샘플 데이터를 사용하는 편이 안전합니다.

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
- 프로젝트 경험과 지원 직무 정보를 저장하는 개인화 RAG 컬렉션 추가
- 질문뿐 아니라 사용자 답변까지 포함해 RAG 검색 쿼리 개선
- 검색된 참고 사례 Top K를 UI에 표시
- 검색된 개인 프로젝트/직무 근거를 UI에 표시
- 개선 전/후 평가 점수 기능 추가
- 면접 질문 TOP100 기반 데이터 생성 파이프라인 추가
- 허용된 웹 페이지에서 질문 후보를 수집하는 크롤링 유틸 추가
- Qwen 로컬 모델을 기본 실행 경로에서 분리
- Gradio 앱을 독립 실행 가능한 `app.py`로 구성

## 개선 방향

- 실제 면접 답변 데이터셋 확장
- 검색 품질 평가 지표 추가
- 답변 개선 결과 저장 기능 추가
- 사용자별 답변 히스토리 관리
- FastAPI 백엔드와 React 프론트엔드로 확장
- 배포 환경에서 ChromaDB 영속성 관리
