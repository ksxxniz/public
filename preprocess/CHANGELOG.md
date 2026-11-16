# 전처리 변경 기록

정규화 동작이 바뀌면 `preprocess/normalize.py` 의 `VERSION` 을 올리고 여기에 적습니다. 이 번호는 청크 `meta.normalize_version` 과 검사 로그 `model_ver` 칸에 남습니다. 번호가 바뀌면 같은 문서라도 정리본과 변환 기록이 달라질 수 있으니, 1차 분류기 학습 데이터도 같은 번호로 다시 만듭니다.

## 1.2 (2026-11-14)

정리본과 변환 기록이 1.1 과 달라질 수 있습니다. 1.1 로 만든 학습 데이터는 이 버전으로 다시 만듭니다. `common/schema.py` 가 바뀌었으니(아래) 전원 리뷰가 필요합니다.

**추가**
- ⑤ 띄엄띄엄 쓴 인코딩: `%69gnore`, `i%67nore`, `\x69gnore`, 8진 `\151gnore`, `%u0069`, 빈칸만 바꾼 `ignore%20all%20previous` 를 그 줄 단위로 풉니다(엔티티·`\u` 표기와 같은 방식). 낱말마다 따로 인코딩한 한국어(`%EC%9D%B4%EC%A0%84 지시를 …`)도 줄 단위로 한 번 더 풀어 문장이 이어집니다. (`preprocess/encoding.py`)
- ⑤ 구분자가 낀 인코딩: 공백·콜론·하이픈으로 띄운 Hex, 10진·2진 글자 코드 나열(`char_code`), Base32(`base32`, 겹친 속 포함), quoted-printable(`quoted_printable`).
- ⑤ CP949 한국어: Base64·Hex·URL 을 푼 바이트가 UTF-8 이 아니면 CP949(EUC-KR)로도 읽습니다(`%C0%CC%C0%FC`).
- ② HTML 속성 글: `data-*`·`longdesc` 같은 데이터·주소 속성 값이 문장이면 정리본에 넣고 `attr_text` 로 표시합니다. `aria-description` 도 표시합니다. 화면에 보이는 `placeholder` 와 `<input>` 값은 표시 없이 정리본에 넣습니다. 주소 속성의 `%20` 문장(`href="…?q=ignore%20all…"`)은 위 줄 단위 복원으로 풀립니다. (`preprocess/html_parse.py`, `preprocess/markup.py`)
- ③ 새 단계 `strip_combining_noise`: 글자·숫자에 겹쳐 쓴 결합 문자(취소선 `i̶g̶n̶o̶r̶e̶`, 둘러싸는 부호, 3개 이상 쌓인 잘고 글씨)를 지우고 `combining_removed` 로 셉니다. (`preprocess/unicode_steps.py`)
- ④ 새 단계 `fold_enclosed_letters`: NFKC 가 못 푸는 `🅸🅶🅽`, `🅘🅖🅝`, `⒤⒢⒩`, 지역 표시 문자 `🇮🇬🇳` 를 두 글자 이상 이어질 때 보통 글자로 바꿉니다. 개수는 `homoglyph_replaced` 에 들어갑니다. (`preprocess/homoglyph.py`)
- 생성기 기법 10개: `url_partial`, `url_spaces`, `url_cp949`, `hex_spaced`, `char_codes`, `base32`, `attribute_text`, `strikethrough`, `zalgo`, `enclosed_letters`. 1.1 에서는 모두 탈출률 100% 였고 1.2 에서 0% 입니다. 정상 변형 3개(`math_greek`, `accents`, `percent_text`)도 더했습니다.
- 테스트 `tests/test_preprocess_gaps.py` (86개).

**바뀜**
- `common/schema.py`: `TransformLog.combining_removed` 추가, `DecodedSegment.method` 에 `base32`·`quoted_printable`·`char_code` 추가, `SCHEMA_VERSION` 0.2.4 → 0.2.5.
- ④ 그리스 글자 하나가 낱말 맨 앞·맨 뒤에 붙은 수식 표기(`αx`, `σx`, `νx`, `ρgh`)를 더는 라틴 문자로 바꾸지 않습니다. 바꾼 결과가 흔한 영어 낱말이나 공격 키워드일 때만 바꿉니다(`αll` → `all`).
- `<input type="text" value="…">` 처럼 화면에 보이는 값이 HTML 정리본에 들어갑니다(1.1 은 버림).
- 글자를 둘러싸는 부호(U+20DD 등)가 낀 낱말은 부호를 지운 뒤 홈글리프를 봅니다(`ра⃝ssword` → `password`).
- CP949 한국어 URL 인코딩은 `�` 가 섞인 글 대신 한국어로 복원됩니다.

**그대로**
- 위 표기·글자가 없는 txt·md 문서의 정리본과 위치 대응표는 1.1 과 같습니다.

**알아 둘 것**
- 정규화가 1.1 보다 약 30% 느립니다(30만 자 문서 0.44초 → 0.57초, 클라우드 2코어). 처리 시간 표(`docs/preprocess_scope.md` 6절)는 팀 PC 에서 다시 재야 합니다.
- 실제 토크나이저(`xlm-roberta-base`) 테스트는 이 변경에서 돌리지 못했습니다. `uv run pytest -m real_tokenizer` 로 확인해 주세요.

## 1.1 (2026-10-31, 4주차)

정리본과 변환 기록이 1.0 과 달라질 수 있습니다. 1.0 으로 만든 학습 데이터는 이 버전으로 다시 만듭니다. `common/schema.py` 가 바뀌었으니(아래) 전원 리뷰가 필요합니다.

**추가**
- ② HTML: `<style>` 블록의 단순 선택자 규칙(`.x{display:none}`, `#id`, `p.x`, `@media` 안 포함)으로 숨긴 글도 숨김으로 표시합니다. 이유 이름은 인라인 스타일과 같습니다. (`preprocess/html_parse.py`)
- ② HTML: `href`·`src`·`action`·`data-*` 같은 주소·데이터 속성 값 안의 Base64·Hex·URL 인코딩 등을 풀어 `decoded_segments` 에 넣습니다. 정리본에는 넣지 않습니다.
- ② HTML·Markdown: 자리표시자(`{대화 내용}` 등)나 32자 이상 쿼리 값이 붙은 외부 이미지 주소를 `hidden_text_found`(note `exfil_image`)로 표시합니다.
- ③ 변형 선택자 바이트 해독: 이모지 뒤에 변형 선택자를 3개 이상 붙여 숨긴 바이트를 UTF-8 로 풀어 `decoded_segments`(method `variation_selector`)에 넣습니다. 새 단계 `decode_variation_selectors` 가 태그 문자 해독 바로 뒤에 들어갑니다. (`preprocess/unicode_steps.py`)
- ④ 키릴·그리스 문자로만 쓴 단어가 공격 키워드(`KEYWORDS`, 41개)로 읽히면 라틴 문자로 바꿉니다(`рrеνіουѕ` → `previous`). 개수는 `homoglyph_replaced` 에 들어갑니다. (`preprocess/homoglyph.py`)
- 복원 결과를 분류기 입력 조각으로 만드는 `preprocess.decoded.decoded_inputs(chunk)`(서빙)와 `decoded_texts(raw, fmt)`(학습). 384토큰을 넘는 복원 결과는 50토큰씩 겹쳐 자르고, 조각은 모두 그 복원 결과의 `span` 을 씁니다. (decisions.md ⑥)
- 정상 문서 기준선: `python -m preprocess.baseline <폴더·파일·JSONL>` 가 정상 문서의 신호 분포(1000자당 p50·p95·p99·최댓값)와 강한 신호가 켜진 청크 비율(전처리 기준 오탐률)을 표로 뽑고, 켜진 청크를 JSONL 로 저장합니다.
- 생성기 기법 4개: `css_class`, `attribute_base64`, `variation_selector`, `cyrillic_keyword`. 1.0 에서는 넷 다 탈출률 100% 였고 1.1 에서 0% 입니다.

**바뀜**
- `common/schema.py`: `DecodedSegment.method` 에 `variation_selector` 추가, `SCHEMA_VERSION` 0.2.3 → 0.2.4.
- `/` 가 든 Base64 후보(`e.com/aWdu…`)를 통째로 풀었을 때 깨진 글자가 섞이면, `/` 로 나눈 조각을 먼저 풀어 봅니다. 앞 조각이 섞여 들어간 복원 결과가 깔끔해집니다.

**그대로**
- txt·md 문서의 정리본은 키릴·그리스 키워드가 든 경우를 빼면 1.0 과 같습니다.

## 1.0 (2026-10-25, 3주차)

팀에 처음 배포하는 버전입니다. 2주차 말에 내기로 한 0.1 은 따로 배포하지 않고 이 버전에 합쳤습니다.

**추가**
- ② HTML: 태그를 지우고, 블록 태그는 줄바꿈으로, 엔티티는 원래 글자로 바꿉니다. 숨김 텍스트는 지우지 않고 `hidden_text_found` 로 위치와 이유를 남깁니다. 브라우저가 주석으로 처리하는 `<?…>`·`<!…>` 와 표준이 아닌 DOCTYPE 안의 글도 주석으로 남기고, `font color`·`bgcolor` 같은 옛 색 속성도 봅니다. (`preprocess/markup.py`, `preprocess/html_parse.py`)
- ② Markdown: 주석, 이미지 `alt`, 링크·이미지 `title`(여러 줄 포함), MD 안 HTML 의 숨김 텍스트를 표시합니다. 글자는 바꾸지 않습니다.
- ⑤ 인코딩 복원: Base64, Hex, URL, ROT13 을 풀어 `decoded_segments` 에 넣습니다. 최대 3겹(겹친 속의 ROT13 포함)까지 풀고, 깨진 바이트가 조금 섞이거나 영어·한국어를 UTF-16 으로 인코딩한 것도 풉니다. 본문에 글자로 적힌 HTML 엔티티(`&#105;gnore`, HTML 의 이중 인코딩 `&amp;#105;`)와 `\u0069` 같은 표기도 그 줄 단위로 풉니다(`html_entity`, `unicode_escape`). (`preprocess/encoding.py`)
- 크기 상한: 파일 8,000,000바이트, 원문(NFKC 로 늘어난 길이 포함) 2,000,000자, 청크 1000개, 복원 결과 10,000개 또는 1,000,000자를 넘는 문서는 앞부분만 처리하지 않고 `DocumentTooLongError` 를 냅니다. 파일·원문 크기는 읽거나 정규화하기 전에 봅니다.
- 근거 위치 찾기: `preprocess.quote.find_quote(chunk, quote)` 가 4단계의 인용문을 정리본·복원 결과에서 찾아 `raw_text` 기준 위치로 돌려줍니다. 그대로 일치 → 공백·대소문자 무시 → 유사 일치(rapidfuzz) 순서입니다. 의존성 `rapidfuzz` 가 늘었습니다.
- 처리 시간 측정: `python -m preprocess.perf_report` 가 1KB~1MB 문서의 p50/p95 를 표로 뽑습니다.
- 문서: `docs/preprocess_transform_log.md`(변환 기록 형식, `find_quote` 사용법), `docs/preprocess_scope.md`(지원 범위·한계, 처리 시간)

**바뀜**
- HTML 문서의 정리본에서 태그가 빠지고 엔티티가 복원됩니다. 0.0.7 로 만든 HTML 학습 샘플은 다시 만들어야 합니다.
- `hidden_text_found` 가 채워집니다(0.0.7 은 항상 0). 숨김 구간이 청크 경계에 걸치면 양쪽 청크에 모두 들어갑니다.
- `decoded_count` 가 Base64·Hex·URL·ROT13·엔티티·`\u` 표기 복원 결과도 셉니다(0.0.7 은 태그 문자만).
- 토크나이저를 `xlm-roberta-base` 로 확정했습니다. (decisions.md ④)
- `pipeline.run` 이 너무 큰 파일·긴 문서와 HTML 파서 버전 문제를 종료 코드 2 로 알립니다.
- `DocumentTooLongError` 는 `preprocess.context` 에 있습니다. 예전처럼 `preprocess.chunk` 에서 import 해도 됩니다.
- 생성기(`datagen/evasion.py`)의 HTML·MD·인코딩 기법 11개를 켜고 인코딩 기법 2개(`html_entity`, `unicode_escape`)를 더했습니다. 탈출률 표에 `hidden_text_found` 칸을 더했습니다.

**고침**
- HTML 파서: 주석 내용이 `<`·`-` 처럼 여는 표시 `<!--` 에 있는 글자로 시작하면(`<!--<-->`, `<!----->`) 주석 위치를 여는 표시 안으로 잘못 잡던 문제

**그대로**
- txt·md 문서의 정리본과 위치 대응표는 0.0.7 과 같습니다. 무작위 문서 8만 개로 비교했고, 복원 결과와 MD 숨김 표시만 더해졌습니다.
- `common/schema.py` 양식은 바꾸지 않았습니다.
