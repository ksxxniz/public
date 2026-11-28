"""스텁 공용 도구 (5단계 ⑧).

스텁은 룰 필터의 내부 구현에 기대지 않는다. 룰 담당이 내부 함수를 고치거나
이름을 바꿔도 스텁이 깨지지 않게, 인용 판별을 여기 따로 둔다.
"""

import re

_WINDOW = 40  # 공격 문구 앞뒤로 이만큼만 본다
_EXAMPLE = re.compile(
    r"예\s*[:：)]|예시|예를\s*들|의\s*예|example|e\.g\.|for\s+instance|such\s+as", re.IGNORECASE
)
_OPEN = "\"'`“‘「『«"
_CLOSE = "\"'`”’」』»"


def looks_quoted(text: str, start: int, end: int) -> bool:
    """text[start:end] 가 따옴표 안이거나 근처에 예시 표시가 있나.
    교육 자료가 공격 문구를 인용한 경우를 진짜 공격과 구분하려는 것이다"""
    before = text[max(0, start - _WINDOW) : start]
    after = text[end : end + _WINDOW]
    if _EXAMPLE.search(before) or _EXAMPLE.search(after):
        return True
    before = before.rsplit("\n", 1)[-1]
    after = after.split("\n", 1)[0]
    # 영어 축약형(don't)의 ' 를 따옴표로 보지 않도록 단어 경계의 따옴표만 센다
    opened = any(
        c in _OPEN and (k == 0 or not before[k - 1].isalnum()) for k, c in enumerate(before)
    )
    closed = any(
        c in _CLOSE and (k + 1 == len(after) or not after[k + 1].isalnum())
        for k, c in enumerate(after)
    )
    return opened and closed
