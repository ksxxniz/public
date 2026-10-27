# 팀 결정 기록

`common/config.py` 의 값과 함께 관리합니다. 한쪽을 바꾸면 다른 쪽도 같은 PR 에서 고치고, 전원 리뷰를 받습니다. 결정을 바꿀 때는 기존 내용을 지우지 않고 변경 내용을 덧붙입니다.

## ① 좌표 규칙

- `Span` 은 `[start, end)` 반열린 구간이고, 파이썬 문자열 인덱스 단위입니다.
- `DecodedSegment.span` 과 `StageResult.evidence_span` 은 `chunk.raw_text` 기준입니다. 문서 기준 위치는 `chunk.span.start` 를 더해서 구합니다.
- 정리본(`text`)에서 찾은 위치는 `Chunk.raw_span()` 으로 원문 위치로 바꿔서 내보냅니다.
- 관련 코드: `common/schema.py` 맨 위 설명
- (2026-10 추가) `raw_span()` 의 끝은 "다음 글자의 원문 시작 위치"까지 잡습니다. 예전 계산은 NFD 자모가 합쳐진 글자(macOS 에서 온 한글)에서 끝을 잘랐습니다. 대신 구간 바로 뒤에서 지워진 글자(제로폭 등)도 하이라이트에 포함됩니다.
- (2026-10 추가) `offset_map` 은 감소하지 않습니다. 감소하면 `Chunk` 생성이 에러를 냅니다.
- (2026-10 추가) 위치 단위는 파이썬 코드포인트입니다. 웹(JS)은 UTF-16 단위라 이모지·태그 문자가 2칸이 되므로 web 에서 바꿔 씁니다.
- (2026-10 추가) `raw_span()` 은 원문 시작 위치가 같은 다음 글자들(한 원문 묶음에서 함께 나온 글자, 예: `ﬁ́` → `fí`)을 건너뛰고, 위치가 바뀌는 글자의 시작 위치까지 잡습니다. 묶음의 앞 글자만 골라도 원문 묶음 전체가 잡힙니다.

## ② 지원 파일 형식

- `.txt`, `.md`, `.html`, `.htm` 만 지원합니다. PDF 는 지원하지 않습니다.
- 관련 코드: `common/config.py` → `SUPPORTED_EXTENSIONS`
- (2026-10 추가) 문자 인코딩은 UTF-8, BOM 이 있는 UTF-16·32, CP949(EUC-KR 포함), 히라가나가 섞인 Shift-JIS(CP932) 일본어를 읽고 그 밖은 charset-normalizer 판별에 맡깁니다. EUC-JP 일본어, GB2312 중국어(특히 짧은 글), 히라가나 없이 한자·가타카나만 있는 짧은 Shift-JIS 글(회사명 등)은 CP949 로 잘못 읽힐 수 있어 지원 범위 밖입니다.

## ③ 점수 방향

- `risk_score` 와 `final_score` 는 0.0~1.0 이고, 높을수록 위험합니다.
- 관련 코드: `common/schema.py` → `StageResult.risk_score`, `Verdict.final_score`

## ④ 청크 크기

- 청크는 384토큰, 겹침은 50토큰입니다. 1단계(전처리)와 3단계(분류)가 같은 값을 씁니다.
- 관련 코드: `common/config.py` → `CHUNK_SIZE_TOKENS`, `CHUNK_OVERLAP_TOKENS`
- (2026-10 추가) 토큰 수는 3단계 1차 모델의 토크나이저로 셉니다. 2주차 말 모델 확정 전까지는 임시로 `xlm-roberta-base` 를 씁니다. 384 에 특수 토큰([CLS], [SEP] 등)을 포함할지는 모델 확정 때 함께 정합니다.
- 관련 코드: `common/config.py` → `TOKENIZER_NAME`
- (2026-10 추가) 384 는 특수 토큰을 포함한 길이로 확정합니다. 후보 세 모델(mDeBERTa-v3-base, klue/roberta-base, xlm-roberta-base)은 모두 문장 하나에 특수 토큰 2개를 붙이므로, 청크 내용은 최대 382토큰입니다. 내용 길이는 `CHUNK_SIZE_TOKENS - (토크나이저가 붙이는 특수 토큰 수)` 로 계산하고, 데이터셋 샘플 길이 검사도 특수 토큰을 붙인 상태로 384 이하인지 봅니다.
- (2026-10-12 확정) 3단계 1차 모델은 `xlm-roberta-base` 입니다. 청크 토큰 수도 이 토크나이저로 세고, 특수 토큰 2개(`<s>`, `</s>`)를 빼면 청크 내용은 최대 382토큰입니다. 모델을 바꾸면 `TOKENIZER_NAME` 과 이 항목을 같은 PR 에서 고칩니다.
- (2026-10 추가, 제안) 문서 하나가 청크 1000개(`preprocess.chunk.MAX_CHUNKS`)를 넘으면 앞부분만 자르지 않고 `DocumentTooLongError` 를 냅니다. 앞부분만 검사하고 넘기면 뒤쪽이 우회 구멍이 되기 때문입니다. 상한값은 통합 담당과 확정합니다.

## ⑤ 실험 기록

- 실험은 Weights & Biases 의 `rag-guard` 프로젝트에 팀 공용 계정으로 기록합니다.
- 관련 코드: `common/config.py` → `WANDB_PROJECT`, `WANDB_ENTITY`

## ⑥ 전처리 원칙 (제안: 이 PR 리뷰에서 확정)

- 원문 좌표는 바이트가 아니라 디코딩된 문자열 기준입니다.
- 숨김 텍스트(흰 글씨, `display:none`, 주석 등)는 지우지 않고 `text` 에 남기며, 위치는 `Chunk.transform_spans` 에 표시합니다. 실제 RAG 의 AI 가 읽는 내용과 검사 대상을 같게 하기 위해서입니다.
- 인코딩 복원 결과(`decoded_segments`)는 본문에 섞지 않고, 별도 청크로 만들어 1차에 함께 통과시킵니다. 그 청크의 위치는 원문의 인코딩 구간을 가리킵니다. (3단계 담당과 합의 필요)
- 변환 개수는 `TransformLog`, 위치는 `Chunk.transform_spans` 에 따로 담습니다. 로그 `reason` 칸에는 개수만 들어갑니다.
- Python 은 3.11 로 고정합니다. 파이썬 버전마다 유니코드 데이터가 달라 NFKC 결과가 바뀔 수 있습니다(3.11 = 14.0, 3.12 = 15.0). 3단계 학습도 3.11 에서 하고, 청크 `meta.unicode_version` 으로 확인합니다.
- (2026-10 추가) 쪼개 쓴 자모(`ㄱㅗㅇㄱㅕㄱ`)는 NFKC 가 아니라 별도 단계에서 조립합니다. 자모로 쓰인 덩어리 안에서만, 단어 단위로 "남는 자모 0개, 음절 2개 이상"일 때만 조립합니다. 정상 채팅(`ㅋㅋㅋㅠㅠ`, `좋아ㅇㅋ`)을 바꾸지 않기 위해서입니다. 조립한 음절 수는 `TransformLog.jamo_assembled` 에 남깁니다.
- (2026-10 추가) `TransformLog.decoded_count` 는 그 청크의 `decoded_segments` 개수입니다. 복원 방법(Base64, 태그 문자 등)과 상관없이 셉니다. 종류는 `decoded_segments[].method` 로 구분하고, 숨긴 태그 글자 수는 `tag_chars_decoded` 에 따로 있습니다.
- (2026-10 추가) 태그 문자 조각이 2개 이상인 청크에는 조각별 결과 뒤에, 그 청크의 조각을 순서대로 이어 붙인 결과를 하나 더 넣습니다(method `unicode_tag`, 위치는 첫 조각 시작부터 마지막 조각 끝까지). 보이는 글자 사이에 숨긴 글자를 하나씩 끼워 해독 결과를 한 글자씩 조각내는 우회를 막기 위해서입니다.
- (2026-10 추가) 공백류(탭·줄바꿈 등)를 뺀 제어 문자(Cc), 점자 공백(U+2800), Cf 가 아닌 기본 무시 문자(U+17B4~17B5, U+2065, U+FFF0~FFF8 등)도 지우고 `zero_width_removed` 로 셉니다. 몽골 자유 변형 선택자(U+180B~180D, U+180F)는 `variation_selector_removed` 로 셉니다.
- (2026-10 추가) 유니코드 버전이 14.0.0(Python 3.11)이 아니면 `preprocess.normalize` 를 import 할 때 `RuntimeWarning` 을 띄웁니다. 3단계 학습 환경이 3.11 인지 확인하는 용도입니다.
- (2026-10 추가) ② HTML 은 태그를 지우고, 블록 태그는 줄바꿈으로, 엔티티는 원래 글자로 바꿉니다. 숨김 텍스트(인라인 스타일과 `font color`·`bgcolor` 같은 옛 색 속성으로 숨긴 글, 주석, CDATA, 브라우저가 주석으로 처리하는 `<?…>`·`<!…>`, 표준이 아닌 DOCTYPE, `script`·`style`·`template`·`noscript` 안의 글, `alt`·`title`·`aria-label`·`meta` 설명·`input type=hidden` 값)는 정리본에 남기고, 위치와 이유를 `hidden_text_found`(note 에 이유)로 남깁니다. HTML 파서는 Python 3.11.15 이상에서만 돌고, 그보다 낮으면 `HtmlParserUnavailableError` 를 냅니다.
- (2026-10 추가) ② Markdown 은 원문 자체가 AI 입력에 가까워서 글자를 하나도 바꾸지 않고 표시만 합니다. 주석(`<!-- -->`, `[//]: # (...)`), 이미지 `alt`, 링크·이미지 `title`, MD 안 HTML 의 숨김 텍스트가 대상이고, 코드 블록·인라인 코드 안은 표시하지 않습니다. MD 안 HTML 의 표시는 HTML 파서가 없는 Python(3.11.15 미만)에서는 건너뜁니다. 정리본은 같으므로 학습용 `normalize_text` 는 그대로 씁니다.
- (2026-10 추가) ⑤ 인코딩 복원은 Base64(표준·URL-safe, 패딩 유무, 줄바꿈된 것), Hex(`\x69`·`0x` 형태 포함), URL(`%XX` 3개 이상 연속), ROT13(흔한 영어 단어 비율이 크게 오를 때만, 근처에 rot13 언급이 있으면 기준을 낮춤)을 풉니다. 푼 바이트는 UTF-8 로 읽고 NUL 이 많거나 한글이 대부분이면 UTF-16 으로 읽으며, 깨진 바이트·제어 문자가 약 10% 이하이면서 글자가 하나 이상 있을 때만 받아들입니다(20자 미만은 깨진 글자가 없어야 함). 숫자로만 된 Hex 는 영어 단어로 풀릴 때만 받아들입니다. 최대 3겹까지 풀고, 겹친 속에 ROT13 이 있어도 풉니다. 이미지 data URI 는 따로 건너뛰지 않고 같은 검사로 거릅니다. 본문에 글자로 적힌 HTML 엔티티(`&#105;`)와 `\u0069` 같은 표기는 글자·숫자나 보이지 않는 문자로 풀리는 숫자 표기가 있을 때만, 그 표기가 있는 줄(앞뒤 200자까지)을 통째로 풀어 `html_entity`·`unicode_escape` 로 넣습니다. `&amp;`·`&eacute;` 같은 이름 엔티티와 기호로 풀리는 표기만 있으면 풀지 않습니다. 복원 결과는 ③·④ 를 다시 거쳐 `decoded_segments` 에 넣고 본문에는 섞지 않습니다.
- (2026-10 추가, 제안) 크기 상한: 파일 8,000,000바이트(`preprocess.load.MAX_DOCUMENT_BYTES`), 원문과 NFKC 뒤 정리본 2,000,000자(`preprocess.normalize.MAX_DOCUMENT_CHARS`), 복원 결과 10,000개·1,000,000자(`preprocess.encoding.MAX_DECODED_SEGMENTS`, `MAX_DECODED_CHARS`)를 넘으면 청크 상한(④)과 같이 `DocumentTooLongError` 로 거부합니다. 파일·원문 크기는 읽거나 정규화하기 전에 봅니다. 처리 시간 상한은 두지 않습니다. 시간으로 끊으면 컴퓨터 속도에 따라 같은 문서의 결과가 달라져 학습과 서빙이 어긋나기 때문입니다. 상한값은 통합 담당과 확정합니다.
- (2026-10 추가, 제안) 처리 시간 목표: 문서 하나(정규화+청킹)를 10KB 이하는 p95 50ms 이하, 100KB 는 p95 500ms 이하, 1MB 는 p95 5초 이하로 둡니다. 3주차 측정값(글자 토크나이저, 클라우드 2코어)은 10KB 27ms, 100KB 256ms, 1MB 2.3초(p95, 세 형식 중 가장 느린 값)입니다. 실제 토크나이저 기준은 팀 PC 에서 `uv run python -m preprocess.perf_report --out 파일.md` 로 다시 재서 확정합니다.
- 관련 코드: `common/schema.py` → `TransformLog`, `TransformSpan`, `DecodedSegment`

## ⑦ 2차 판정 실패 처리 (제안: 4단계 D)

- LLM 답변(JSON)을 읽지 못하면 1번 다시 묻고, 그래도 못 읽으면 차단합니다(fail-closed): `label="injection"`, `risk_score=1.0`, `confidence=0.0`.
- LLM 이 인용한 근거 문장을 `preprocess.quote.find_quote` 로 원문에서 찾지 못하면, 판정은 그대로 두고 `evidence_span` 만 비웁니다. 판정을 바꾸면 실제 공격을 놓칠 수 있기 때문입니다.
- 2차 서버가 응답하지 않을 때(fail-open / fail-closed)는 통합 담당과 확정합니다.
- 관련 코드: `stage2_llm/judge.py`, `stage2_llm/config.py` → `MAX_RETRIES`
