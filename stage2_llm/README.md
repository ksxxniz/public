# stage2_llm — 4단계 2차 소형 LLM 판정기 (D)

1차에서 애매하다고 넘어온 청크를 소형 LLM 이 다시 읽고, 판정과 근거를 `StageResult` 로 돌려줍니다.

## 흐름

```
Chunk.text + decoded_segments
  → 랜덤 구분자 봉투 (make_envelope)
  → Ollama (JSON 양식 강제, temperature 0)
  → JudgeResult {evidence_quote, attack_type, label, confidence, reason}
  → find_quote 로 근거 문장을 원문 위치(evidence_span)로 변환
  → StageResult(stage="stage2")
```

파이프라인에서는 `judge_chunk(chunk, MODELS["exaone"])` 하나만 부르면 됩니다.

## 실행 (저장소 루트에서)

```bash
ollama serve                                         # 다른 터미널에서
uv run python -m stage2_llm.test_connection exaone   # 연결 확인
uv run python -m stage2_llm.quick_test exaone        # 붙여넣고 END
uv run python -m stage2_llm.run --model exaone      # 평가 세트 전체 판정
uv run python -m stage2_llm.report                   # 모델 비교표 만들기
uv run pytest tests/test_stage2_judge.py             # GPU 없이 도는 테스트
```

## 모델 비교

1. 모델마다 `run` 을 한 번씩 돌린다.
2. `report` 가 결과를 모아 `results/{데이터이름}/report_v0.3.md` 로 표를 만든다.

```bash
# 내 평가 세트 (evalsets/eval_v0.1.jsonl, 40건)
uv run python -m stage2_llm.run --model exaone
uv run python -m stage2_llm.report

# 팀 데이터 폴더 (metadata_*.jsonl + 문서 파일)
uv run python -m stage2_llm.run --model exaone --data datasets/v0.1 --sample 200
uv run python -m stage2_llm.report --name v0.1
```

| 옵션 | 뜻 |
|---|---|
| `--data` | jsonl 파일 또는 팀 데이터 폴더. 폴더면 `metadata_*.jsonl` 을 찾아 문서 파일 내용을 읽는다 |
| `--sample N` | 언어·라벨 비율을 유지하며 N개만. seed 고정이라 모델이 달라도 같은 문서 |
| `--limit N` | 앞에서 N개만 (시험용) |
| `--resume` | 중간에 끊겼을 때 이미 판정한 문서는 건너뛰고 이어서 |
| `--name` | 결과 폴더 이름 (기본: 데이터 파일·폴더 이름) |

팀 데이터의 label 0/1, language, file_path 는 `stage1/convert.py` 와 같은 규칙으로 읽는다.
`hard_negative: true` 인 문서는 하드 네거티브 정답률에 들어간다.

`eval_v0.1` 은 D 가 만든 개발용 세트입니다 (영어 30·한국어 10, 정상 20·공격 20,
그중 하드 네거티브 10). few-shot 예시는 팀 train 데이터에서만 고르고, test 는 쓰지 않습니다.

## 정해 둔 처리

| 상황 | 처리 |
|---|---|
| JSON 을 못 읽음 | 1번 다시 묻고, 그래도 안 되면 `injection`, `risk_score=1.0` (fail-closed) |
| 근거 문장을 원문에서 못 찾음 | 판정은 그대로, `evidence_span=None` (하이라이트 없음) |
| 근거를 비슷한 문장으로 찾음 | 하이라이트하고 `quote_method="fuzzy"`, 점수 기록 |
| label 과 attack_type 이 안 맞음 | 판정은 그대로, `inconsistent=True` 기록 |

`risk_score` 는 지금 확신도 단어(high/medium/low)로 계산하는 임시 방식입니다. logprob 기반으로 바꿀 예정입니다.

## 모델 (Ollama, Q8_0)

EXAONE-3.5-7.8B, Kanana-1.5-8B, Llama-3.1-8B, Qwen3.5-9B, gpt-oss-safeguard-20b. 이름은 `config.py`.

## 프롬프트 버전

- v0.1: 공격 정의 보강, FAQ·퀴즈 오탐 방지
- v0.2: 의심 문장 찾는 순서 + 판정 예시 2개
- v0.3: 팀 양식에 맞춤 (`injection` 라벨, 근거 한 문장 `evidence_quote`, 복원 내용 `<decoded_…>`)

v0.1·v0.2 는 기록용입니다. 지금 코드의 양식(`evidence_quote`, `injection`)과 맞지 않아 그대로는 못 씁니다.
