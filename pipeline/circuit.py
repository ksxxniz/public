"""2차 회로 차단기 (5단계 ④).

- 연속 CIRCUIT_FAIL_THRESHOLD 번 실패(오류나 시간 초과)하면 열고,
  CIRCUIT_COOLDOWN_S 동안 2차를 부르지 않는다.
- 식은 뒤 한 건만 보내 본다(half-open). 성공하면 닫고, 실패하면 다시 연다.
- 열고 닫을 때마다 events 에 남긴다. 7단계 파이프라인 시각화의 재료다.

값은 출발점이고, 부하 테스트 뒤에 고친다 (common/config.py 결정 ⑪).
"""

import threading
import time

from common.config import CIRCUIT_COOLDOWN_S, CIRCUIT_FAIL_THRESHOLD


class CircuitBreaker:
    def __init__(
        self,
        threshold: int = CIRCUIT_FAIL_THRESHOLD,
        cooldown_s: float = CIRCUIT_COOLDOWN_S,
        clock=time.monotonic,
    ):
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at: float | None = None
        self._probing = False  # half-open 에서 시험 요청 1건이 나가 있는가
        self.events: list[tuple[float, str]] = []  # (시각, "open"|"half_open"|"closed")

    @property
    def state(self) -> str:
        with self._lock:
            return self._state_locked()

    def _state_locked(self) -> str:
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at >= self.cooldown_s:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        """지금 2차를 불러도 되는가. half-open 에서는 한 건만 허용한다"""
        with self._lock:
            state = self._state_locked()
            if state == "closed":
                return True
            if state == "half_open" and not self._probing:
                self._probing = True
                self.events.append((self._clock(), "half_open"))
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            if self._opened_at is not None:
                self.events.append((self._clock(), "closed"))
            self._failures = 0
            self._opened_at = None
            self._probing = False

    def record_failure(self) -> None:
        with self._lock:
            self._probing = False
            if self._opened_at is not None:
                # half-open 시험 실패 → 다시 연다
                self._opened_at = self._clock()
                self.events.append((self._clock(), "open"))
                return
            self._failures += 1
            if self._failures >= self.threshold:
                self._opened_at = self._clock()
                self.events.append((self._clock(), "open"))
