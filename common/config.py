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
