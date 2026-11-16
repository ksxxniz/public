"""3단계 성능 지표. torch 없이 돈다.

labels 는 0/1 (1 = injection), scores 는 risk_score (0~1, 높을수록 위험).
score >= threshold 이면 공격으로 본다.

가이드 예시: 공격 100 · 정상 300 중 110개를 막았고 그중 진짜 공격이 95개
  recall    = 95 / 100 = 0.95   공격을 몇 개 잡았나 (1차는 이게 우선)
  precision = 95 / 110 = 0.86   막은 것 중 진짜 공격 비율
  fpr       = 15 / 300 = 0.05   정상을 잘못 막은 비율
"""

from collections.abc import Sequence


def confusion(
    labels: Sequence[int], scores: Sequence[float], threshold: float = 0.5
) -> dict[str, int]:
    if len(labels) != len(scores):
        raise ValueError(f"labels({len(labels)})와 scores({len(scores)}) 개수가 다릅니다")
    tp = fp = tn = fn = 0
    for y, s in zip(labels, scores, strict=True):
        pred = s >= threshold
        if y == 1:
            tp += pred
            fn += not pred
        else:
            fp += pred
            tn += not pred
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def binary_metrics(
    labels: Sequence[int], scores: Sequence[float], threshold: float = 0.5
) -> dict[str, float]:
    """recall, precision, fpr, f1 과 혼동행렬 칸(tp, fp, tn, fn)."""
    c = confusion(labels, scores, threshold)
    recall = _div(c["tp"], c["tp"] + c["fn"])
    precision = _div(c["tp"], c["tp"] + c["fp"])
    return {
        "recall": recall,
        "precision": precision,
        "fpr": _div(c["fp"], c["fp"] + c["tn"]),
        "f1": _div(2 * precision * recall, precision + recall),
        **{k: float(v) for k, v in c.items()},
    }
