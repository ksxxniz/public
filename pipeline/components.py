"""부품 로더 (5단계 ④·⑧).

부품은 준비되는 대로 하나씩 바꾼다. 파이프라인은 여기서 받은 핸들만 쓰므로,
스텁 ↔ 진짜 교체가 이 파일 밖으로 새지 않는다.

- 룰 필터: 모델이 없어 스텁 없이 바로 진짜를 쓴다 (⑧)
- 1차: "stub"(기본) | "real"(torch·체크포인트 필요) | "off"(장애 흉내)
- 2차: "stub" | "benign" | "broken_json"(스텁 변형) | "real"(ollama 서버) | "off"
"""

from common.schema import Chunk, StageResult


class ComponentUnavailable(RuntimeError):
    """부품을 불러오지 못했다. 파이프라인은 이를 장애 정책(⑥)으로 처리한다"""


class Stage1Handle:
    """1차 부품. classify_chunks 는 조각들을 한 번에 배치로 받는다 (④ 배치)"""

    def __init__(self, impl, model_ver: str):
        self._impl = impl
        self.model_ver = model_ver

    def classify_chunks(self, chunks: list[Chunk]) -> list[StageResult]:
        return self._impl.classify_chunks(chunks)


class Stage2Handle:
    """2차 부품. judge_chunk 는 조각 하나를 판정한다"""

    def __init__(self, judge_fn, model_ver: str):
        self._judge = judge_fn
        self.model_ver = model_ver

    def judge_chunk(self, chunk: Chunk) -> StageResult:
        return self._judge(chunk)


def load_stage1(kind: str = "stub", model_dir: str | None = None) -> Stage1Handle | None:
    """kind: stub | real | off. off 는 1차 장애 흉내(핸들 없음)"""
    if kind == "off":
        return None
    if kind == "stub":
        from stage1.stub import StubStage1

        impl = StubStage1()
        return Stage1Handle(impl, impl.model_ver)
    if kind == "real":
        try:
            from stage1.infer import load_classifier
        except ImportError as e:  # torch·transformers 미설치
            raise ComponentUnavailable(f"1차 모듈을 불러오지 못함: {e}") from e
        try:
            clf = load_classifier(model_dir)
        except (FileNotFoundError, OSError) as e:  # 체크포인트 없음 (⑥ 표)
            raise ComponentUnavailable(f"1차 모델 파일 없음: {e}") from e
        return Stage1Handle(clf, clf.model_ver)
    raise ValueError(f"stage1 종류를 모름: {kind}")


def load_stage2(kind: str = "stub", model: str = "exaone") -> Stage2Handle | None:
    """kind: stub | benign | broken_json | real | off. off 는 2차 서버 끄기 흉내"""
    if kind == "off":
        return None
    if kind in ("stub", "benign", "broken_json"):
        from stage2_llm.stub import StubJudge

        judge = StubJudge("ok" if kind == "stub" else kind)
        return Stage2Handle(judge.judge_chunk, judge.model_ver)
    if kind == "real":
        try:
            from stage2_llm.config import MODELS, PROMPT_VERSION
            from stage2_llm.judge import judge_chunk, model_version
        except ImportError as e:
            raise ComponentUnavailable(f"2차 모듈을 불러오지 못함: {e}") from e
        profile = MODELS[model]

        def _judge(chunk: Chunk) -> StageResult:
            result, _info = judge_chunk(chunk, profile, PROMPT_VERSION)
            return result

        return Stage2Handle(_judge, model_version(profile, PROMPT_VERSION))
    raise ValueError(f"stage2 종류를 모름: {kind}")
