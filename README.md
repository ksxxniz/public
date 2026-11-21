# Capstone_design2
2026-2 캡스톤디자인2 - 캐치
주제: RAG 파이프라인을 위한 다단계 캐스케이드 기반 간접 프롬프트 인젝션 탐지 및 차단 시스템

## 1. 프로젝트 소개
### 1-1. 배경 및 필요성
- RAG는 외부 문서를 검색하여 LLM의 컨텍스트로 활용하기 때문에 신뢰할 수 없는 문서에 포함된 악의적인 지시문이 모델에 전달될 수 있음.
- 이러한 구조를 악용한 간접 프롬프트 인젝션은 시스템 프롬프트 탈취, 정보 유출, 악의적 명령 수행 등의 보안 위협으로 이어질 수 있음.
- 따라서 검색된 문서가 LLM에 전달되기 전에 인젝션 위험을 탐지하고 차단하는 방어 구조가 필요함.

### 1-2. 목표
- RAG 환경에서 외부 문서를 통해 유입되는 간접 프롬프트 인젝션을 탐지하고 방어하는 시스템 구현
- 경량 분류기와 소형 LLM을 결합한 다단계 탐지 구조 구현
- 탐지율, 오탐률, 응답 지연 등을 측정하여 방어 시스템의 성능 평가

## 2. 팀원 소개 및 역할 분담
- 김서연(조장): 모델 실행·서빙 환경 구축 및 파이프라인/전체 시스템 통합
- 김서준: 문서 전처리 모듈 개발 및 통합 지원
- 이윤서: 2차 소형 LLM 기반 판정 모듈 개발
- 이화영: 학습·평가 데이터셋 구축 및 판정 로직 지원
- 최민서: 1차 인코더 모델 학습 및 성능 평가

## 3. 주요 기능 및 개발 내용
- (추후 작성 예정)

## 4. 기술 스택
- (추후 작성 예정)

## 5. 실행 결과 화면
- (추후 작성 예정)

## 6. 결과 분석 및 한계점
- (추후 작성 예정)

## 7. 실행 방법 (5단계 통합)

```bash
uv sync                                   # 의존성 설치
uv sync --group stage1                    # 1차 진짜 부품용 (torch·transformers·wandb)
uv run python -m pipeline.run --file 문서.md          # 파이프라인 검사 (1차·2차는 기본 스텁)
uv run python -m pipeline.run --file 문서.md --stage1 real --stage2 real --model exaone
uv run python -m rag.demo                 # 작은 RAG 방어 OFF/ON 비교 (7단계 시연·6단계 ASR 공용 시나리오)
uv run python -m pipeline.recheck --file 문서.md      # 재학습·규칙 수정 뒤 일괄 재검사 (⑦)
uv run pytest                             # 계약·통합 테스트 (스텁). 진짜 부품: -m needs_model
```

- 통합 설계·결정 기록: `docs/decisions.md` ⑧~⑫ (초안 — 전원 리뷰 대상)
- 파이프라인 코드: `pipeline/` (오케스트레이터·판정 합치기·정책 엔진·회로 차단기·판정 저장)
- 작은 RAG·시연 시나리오: `rag/`, `rag/scenario_data/scenarios.jsonl`

## 8. RAG 시연: 문서 넣기와 화면

```bash
uv run python -m rag.ingest --dir rag/sample_docs      # 폴더를 통째로 넣기 (넣을 때 검사하고 판정 저장)
uv run python -m rag.ingest --dir 문서폴더 --prune       # 폴더에서 지운 문서는 색인에서도 빼기
uv run python -m rag.ingest --list                      # 넣은 문서 목록
uv run python -m rag.ask "연차는 며칠이야?"              # 방어 OFF/ON 답 비교 (명령줄)
uv run python -m web                                    # 시연 화면 http://127.0.0.1:8000
uv run python -m web --ingest-dir rag/sample_docs       # 문서를 넣으면서 화면 켜기
uv run python -m web --answer real --stage2 real --model exaone   # ollama 로 답변·2차
```

- 넣기·질문·화면은 같은 DB(`data/inspection_log.db`)를 씁니다. 넣을 때 내린 판정을 질문할 때 꺼내 쓰기 때문입니다. 화면을 켜 둔 채 다른 창에서 넣어도 다음 질문부터 반영됩니다.
- 지원 형식은 txt·md·html 입니다. 읽을 수 없거나 전처리에 실패한 파일은 "검사 불가로 차단"으로 기록하고 색인하지 않으며, 다음에 넣을 때 다시 검사합니다.
- 탐지 시 조치(마스킹·청크 차단·경고만)는 질문할 때 고릅니다. 저장하는 것은 판정까지입니다. 마스킹은 공격 구간을 `[삭제된 내용]` 으로 바꿔 답변 LLM 에 넘깁니다(`common/config.py` 의 `MASK_MARKER`).
- 화면 탭: **질의 분석**(방어 적용 전후 답변 비교, 검색된 청크별 점수·판정 근거·탐지 규칙, 행을 펼치면 줄 번호가 붙은 원문에 탐지 구간 표시), **문서 분석**(붙여 넣은 문서를 색인하지 않고 검사), **색인 문서**(넣은 문서 목록과 문서 전체 원문).
- `rag/sample_docs/` 는 시연용 가상 사내 문서입니다. 정상 문서 5개와, 흰 글씨·제로폭 문자·주석·Base64 로 공격을 숨긴 문서 4개가 들어 있습니다.
- 1차·2차·답변 LLM 이 스텁이면 화면 상단에 시험용 스텁 구성 안내가 표시됩니다. 스텁 결과는 실제 탐지 성능과 무관합니다.
- 코드: `rag/service.py`(넣기·질문), `rag/explain.py`(정책 적용·화면용 설명), `web/app.py`(API), `web/static/index.html`(화면)
