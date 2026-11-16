"""판정 합치기 (5단계 ⑤).

룰·1차·2차 결과를 위에서부터 처음 맞는 줄로 합친다.
"룰은 위험 점수를 올리기만 하고 내리지 않는다"(룰 가이드)를 그대로 따른다.

| 조건                                | 최종      |
|-------------------------------------|-----------|
| 룰 RULE_BLOCK(0.9) 이상             | 공격      |
| 1차 THETA_HIGH 이상                 | 공격      |
| 2차를 불렀고 공격                   | 공격      |
| 2차를 불렀고 정상, 룰 RULE_WARN 이상 | 경고      |
| 2차를 불렀고 정상                   | 정상      |
| 2차가 필요했는데 장애               | fail-safe |
| 2차가 필요 없음                     | 정상      |

룰 0.9 이상이 2차보다 위인 이유: 태그 문자·숨김 구간·복원 글 안의 공격 문구는
정상 문서가 할 이유가 거의 없고, 판정기를 속이는 문구가 함께 들어 있어도
2차 결과에 흔들리지 않으며, 2차가 죽어도 이 줄이 그대로 막는다.
"경고"는 판정 결과값이다: 조각은 그대로 넘기고 화면·로그에 의심 표시만 한다.
"""

from dataclasses import dataclass

from common.config import RULE_BLOCK, RULE_WARN, THETA_HIGH
from common.schema import StageResult

Final = str  # "injection" | "warn" | "benign" | "fail_safe"


@dataclass(frozen=True)
class Merged:
    final: Final
    score: float
    why: str  # 어느 줄에 맞았나 (로그 reason 용)


def merge(
    rule: StageResult | None,
    s1: StageResult | None,
    s2: StageResult | None,
    *,
    s2_needed: bool,
    s2_failed: bool,
) -> Merged:
    scores = [r.risk_score for r in (rule, s1, s2) if r is not None]
    score = max(scores) if scores else 0.0
    if rule is not None and rule.risk_score >= RULE_BLOCK:
        return Merged("injection", score, f"rule>={RULE_BLOCK}")
    if s1 is not None and s1.risk_score >= THETA_HIGH:
        return Merged("injection", score, f"stage1>=theta_high({THETA_HIGH})")
    if s2 is not None and not s2_failed:
        if s2.label == "injection":
            return Merged("injection", score, "stage2=injection")
        if rule is not None and rule.risk_score >= RULE_WARN:
            return Merged("warn", score, f"stage2=benign,rule>={RULE_WARN}")
        return Merged("benign", score, "stage2=benign")
    if s2_needed and s2_failed:
        return Merged("fail_safe", score, "stage2_unavailable")
    return Merged("benign", score, "early_exit")
