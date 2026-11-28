"""작은 RAG 방어 ON/OFF 통합 테스트 (5단계 ①·완료 기준).

시연 시나리오 전부: 방어 OFF 에서 공격 성공, 방어 ON 에서 공격 실패 + 정답.
시나리오 파일(rag/scenario_data/scenarios.jsonl)은 7단계 시연과 6단계 ASR
측정이 같이 읽는다 — 여기서는 스텁 답변기(지시에 복종하는 모델 흉내)로 검증한다.
"""

import pytest

from pipeline.components import load_stage1, load_stage2
from pipeline.orchestrator import Orchestrator
from rag.demo import load_scenarios, run_scenario, stub_answer


@pytest.fixture(scope="module")
def orch():
    return Orchestrator(load_stage1("stub"), load_stage2("stub"), None)


@pytest.mark.parametrize("scenario", load_scenarios(), ids=lambda s: s["id"])
def test_defense_on_off(scenario, orch):
    row = run_scenario(scenario, orch, stub_answer)
    assert row["off"]["attack_success"], "방어 OFF 에서 공격이 먹히는지 먼저 확인 (①)"
    assert not row["on"]["attack_success"], row["on"]
    assert row["on"]["correct"], row["on"]


def test_scenario_file_shape():
    for s in load_scenarios():
        assert {"id", "question", "docs", "evasion", "attack_success_if", "correct_if"} <= set(s)
