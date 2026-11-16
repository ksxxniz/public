"""시드, 경로, 팀 공통 상수.

값을 바꾸면 docs/decisions.md 도 같이 고친다.
학습·평가 스크립트는 맨 위에서 set_seed() 를 호출한다.
"""

import os
import random
from pathlib import Path

import numpy as np

# ── 재현성 ──
SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ── 경로 ──
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("RAG_GUARD_DATA_DIR", ROOT_DIR / "data"))
DB_PATH = Path(os.environ.get("RAG_GUARD_DB_PATH", DATA_DIR / "inspection_log.db"))

# ── 결정 ② 지원 파일 형식 (PDF 미지원) ──
SUPPORTED_EXTENSIONS = (".txt", ".md", ".html", ".htm")

# ── 결정 ③ 점수 방향: risk_score / final_score 는 0.0~1.0, 높을수록 위험 ──

# ── 결정 ④ 청크 크기 (1단계·3단계 공용) ──
CHUNK_SIZE_TOKENS = 384
CHUNK_OVERLAP_TOKENS = 50
# 토큰 수를 세는 토크나이저 = 3단계 1차 모델의 토크나이저 (2026-10-12 확정).
# 모델을 바꾸면 docs/decisions.md ④ 도 같은 PR 에서 고친다.
TOKENIZER_NAME = "xlm-roberta-base"

# ── 결정 ⑤ 실험 기록 ──
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "rag-guard")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")  # 팀 공용 계정

# ── 결정 ⑧ 파이프라인 경계값 (5단계 ④·⑤) — 초기값, 부하 테스트·팀 확정 뒤 고친다 ──
# 1차 점수가 [THETA_LOW, THETA_HIGH) 면 애매 → 2차. THETA_HIGH 이상이면 공격 확정.
THETA_LOW = 0.30
THETA_HIGH = 0.80
# 룰 점수가 이 값 '초과'면 2차에 보낸다 (주석 신호 SIG-COMMENT 가 정확히 0.3 이라
# "이상"으로 두면 HTML 주석 하나 있는 정상 문서도 2차로 간다. 5단계 가이드 ④).
RULE_TO_STAGE2 = 0.30
RULE_WARN = 0.70  # 2차가 정상이라 해도 룰이 이 값 이상이면 "경고"
RULE_BLOCK = 0.90  # 이 값 이상이면 2차 결과와 상관없이 공격 (결정 ⑨)

# ── 결정 ⑨ 정책 기본값 (5단계 ⑤) ──
# 공격 판정 조각의 처리: "mask" | "block" | "warn". 마스킹할 위치가 없으면 block 으로 떨어진다.
DEFAULT_POLICY = "mask"
MASK_MARKER = "[검문소가 지운 문장]"

# ── 결정 ⑩ 장애 정책 (5단계 ⑥) — 인프라 장애의 합의 fail-safe. 초기값은 경고 통과 ──
# "warn": 통과시키되 경고 표시 + 재검사 대상으로 로그에 남김 / "block": 차단
FAIL_SAFE_ACTION = "warn"

# ── 결정 ⑪ 2차 호출 상한·타임아웃·회로 차단기 (5단계 ④) — 부하 테스트 뒤 고친다 ──
STAGE2_TIMEOUT_S = 30.0  # 2차 요청 하나의 상한
STAGE2_MAX_CONCURRENT = 4  # 동시에 보낼 2차 요청 수 상한
CIRCUIT_FAIL_THRESHOLD = 5  # 연속 실패 n 번이면 회로 열림
CIRCUIT_COOLDOWN_S = 30.0  # 열린 뒤 이 시간 동안 2차를 부르지 않음
