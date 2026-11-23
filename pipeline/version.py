"""pipeline_ver (5단계 ⑦).

판정은 전처리·룰·1차·2차·경계값에 따라 달라지므로, 이것들을 묶은 문자열이
같을 때만 저장된 판정을 재사용한다. 원문 해시와 둘 다 맞아야 한다.
"""

import hashlib

from common.config import (
    RULE_BLOCK,
    RULE_TO_STAGE2,
    RULE_WARN,
    THETA_HIGH,
    THETA_LOW,
)


def pipeline_ver(stage1_ver: str, stage2_ver: str) -> str:
    from preprocess.normalize import VERSION as PREPROCESS_VER
    from rule.rules import RULE_VERSION

    thresholds = f"{THETA_LOW},{THETA_HIGH},{RULE_TO_STAGE2},{RULE_WARN},{RULE_BLOCK}"
    return (
        f"pp:{PREPROCESS_VER}|rule:{RULE_VERSION}|s1:{stage1_ver}|s2:{stage2_ver}|th:{thresholds}"
    )


def raw_hash(raw_text: str) -> str:
    """조각 원문 해시. 같은 문서ID 로 내용이 바뀐 문서가 다시 들어와도
    옛 판정이 쓰이지 않게 한다 (⑦)"""
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16]
