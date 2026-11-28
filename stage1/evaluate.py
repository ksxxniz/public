"""1차 분류기 평가 (6단계). 서빙과 같은 방식으로 문서를 청크로 나눠 판정한다.

    uv run --no-sync python -m stage1.evaluate --data data/v1/validation.jsonl \
        --model checkpoints/stage1/xlmr-v1b --report data/eval_xlmr-v1b_val.md

보여 주는 것
1. 문서 단위 지표: 청크 중 가장 높은 1차 점수로 판정 (0.5 기준)
2. 나눠 본 지표: 공격은 숨김 방식·언어·형식별 recall, 정상은 헷갈림 여부·언어·형식별 fpr
3. 긴 문서: 청크가 여러 개인 문서와 하나인 문서의 recall (학습 때 384 토큰에서 잘린 영향)
4. 경계값 표: θ_low·θ_high 마다 룰을 켠 상태로 파이프라인 흐름을 흉내 낸다 (결정 ⑧)
   - 바로 공격: 룰 RULE_BLOCK 이상 또는 1차 θ_high 이상
   - 2차로 감: 1차 θ_low 이상 또는 룰 RULE_TO_STAGE2 초과 (그 밖은 바로 정상)
   - 확정 recall = 2차 없이 바로 공격으로 잡은 공격 문서 비율
   - 최대 recall = 2차가 다 맞힌다고 할 때 (바로 공격 + 2차로 간 공격 문서)
   - 정상 청크 2차 비율 = 2차 호출률. 예산 목표는 15% 이하
   - 바로 공격 fpr = 2차 없이 공격으로 막힌 정상 문서 비율
   - θ_high "없음" = 1차는 혼자 막지 않고, 애매하든 확신하든 2차로 보낸다
5. 문서 목록 (지금 설정 θ_low·θ_high 기준)
   - 못 잡은 공격: 바로 공격도 아니고 2차로도 안 간 공격 문서
   - 확신 오탐: 1차가 FP_CONFIDENT 이상으로 확신했거나 바로 공격으로 막힌 정상 문서

테스트 몫은 마지막에 한 번만 돌린다. 경계값은 검증 몫으로 정한다.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from common.config import RULE_BLOCK, RULE_TO_STAGE2, THETA_HIGH, THETA_LOW
from preprocess.chunk import split_document
from stage1.dataset import Example, load_jsonl
from stage1.metrics import binary_metrics

THETA_LOWS = (0.1, 0.2, 0.3, 0.4, 0.5)
THETA_HIGHS: tuple[float | None, ...] = (0.7, 0.8, 0.9, 0.95, None)  # None = 1차는 막지 않음
FP_CONFIDENT = 0.95  # 정상인데 1차가 이 이상이면 "확신 오탐" 목록에 올린다
STAGE2_BUDGET = 0.15  # 정상 청크 중 2차로 가는 비율 목표 (결정 ⑧)
CONFIRMED_FPR_MAX = 0.01  # 2차 없이 정상 문서를 막는 비율 상한 (룰 가이드 오탐률 1%)


@dataclass
class DocScore:
    ex: Example
    stage1: list[float]  # 청크별 1차 점수
    rule: list[float]  # 청크별 룰 점수 (룰을 끄면 0)
    rule_reasons: list[str]  # 청크별 룰이 걸린 규칙 id

    @property
    def doc_id(self) -> str:
        return str(self.ex.extra.get("id") or self.ex.doc)

    @property
    def is_attack(self) -> bool:
        return self.ex.label == "injection"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="1차 분류기 평가 (청크 단위 판정, 경계값 표)")
    p.add_argument("--data", type=Path, required=True, help="평가 JSONL (검증 또는 테스트 몫)")
    p.add_argument("--model", type=Path, required=True, help="stage1.train 이 저장한 폴더")
    p.add_argument("--no-rule", action="store_true", help="룰 필터 없이 1차만으로 경계값 표")
    p.add_argument("--report", type=Path, help="결과를 이 파일에도 저장 (.md). 같은 이름 .json 도")
    p.add_argument("--limit", type=int, help="앞에서 N개만 (시험용)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    from stage1.infer import Stage1Classifier  # torch 는 실제로 평가할 때만

    examples = load_jsonl(args.data)[: args.limit]
    clf = Stage1Classifier(args.model)
    check = None if args.no_rule else _rule_checker()
    started = time.perf_counter()
    docs = [score_doc(ex, clf, check) for ex in examples]
    elapsed = time.perf_counter() - started
    n_chunks = sum(len(d.stage1) for d in docs)

    result = {
        "model": clf.model_ver,
        "data": str(args.data),
        "rule": check is not None,
        "n_docs": len(docs),
        "n_chunks": n_chunks,
        "ms_per_chunk": elapsed * 1000 / max(n_chunks, 1),
        "doc_metrics": doc_metrics(docs),
        "breakdown": breakdown(docs),
        "length": length_split(docs),
        "cascade": cascade_table(docs),
        "missed_attacks": missed_attacks(docs),
        "confident_fps": confident_fps(docs),
    }
    result["suggestion"] = suggest(result["cascade"])
    text = render(result)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
        args.report.with_suffix(".json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result


def _rule_checker():
    from rule.filter import check_chunk

    return check_chunk


def score_doc(ex: Example, clf, check) -> DocScore:
    chunks = split_document(str(ex.extra.get("id") or ex.doc), ex.raw, ex.fmt)
    s1 = [r.risk_score for r in clf.classify_chunks(chunks)]
    rules = [check(c) for c in chunks] if check else []
    rule = [r.risk_score for r in rules] if check else [0.0] * len(chunks)
    reasons = [r.reason for r in rules] if check else [""] * len(chunks)
    return DocScore(ex, s1, rule, reasons)


def doc_metrics(docs: list[DocScore]) -> dict:
    labels = [int(d.is_attack) for d in docs]
    return binary_metrics(labels, [max(d.stage1, default=0.0) for d in docs])


def breakdown(docs: list[DocScore]) -> dict:
    """공격은 recall, 정상은 fpr 을 묶음별로 (0.5 기준, 1차만)"""
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [개수, 공격 판정]
    for d in docs:
        hit = int(max(d.stage1, default=0.0) >= 0.5)
        x = d.ex.extra
        keys = (
            [f"공격 · 숨김 {x.get('concealment')}", f"공격 · {x.get('lang')}", f"공격 · {d.ex.fmt}"]
            if d.is_attack
            else [
                "정상 · 헷갈림" if x.get("hard_negative") else "정상 · 일반",
                f"정상 · {x.get('lang')}",
                f"정상 · {d.ex.fmt}",
            ]
        )
        for k in keys:
            groups[k][0] += 1
            groups[k][1] += hit
    return {k: {"n": n, "rate": hit / n} for k, (n, hit) in sorted(groups.items())}


def length_split(docs: list[DocScore]) -> dict:
    out = {}
    for name, keep in (
        ("청크 1개", lambda d: len(d.stage1) == 1),
        ("청크 여러 개", lambda d: len(d.stage1) > 1),
    ):
        attacks = [d for d in docs if d.is_attack and keep(d)]
        caught = sum(max(d.stage1) >= 0.5 for d in attacks)
        out[name] = {"n": len(attacks), "recall": caught / len(attacks) if attacks else 0.0}
    return out


def chunk_route(s1: float, rule: float, low: float, high: float | None) -> str:
    """pipeline/merge.py 의 흐름을 1차·룰 점수만으로 흉내 낸다. high=None 이면 1차는 막지 않는다"""
    if rule >= RULE_BLOCK or (high is not None and s1 >= high):
        return "injection"
    if s1 >= low or rule > RULE_TO_STAGE2:
        return "stage2"
    return "benign"


def doc_routes(d: DocScore, low: float, high: float | None) -> set[str]:
    return {chunk_route(s, r, low, high) for s, r in zip(d.stage1, d.rule, strict=True)}


def cascade_table(docs: list[DocScore]) -> list[dict]:
    rows = []
    attacks = [d for d in docs if d.is_attack]
    benign = [d for d in docs if not d.is_attack]
    benign_chunks = sum(len(d.stage1) for d in benign)
    for low in THETA_LOWS:
        for high in THETA_HIGHS:
            if high is not None and low >= high:
                continue

            confirmed = sum("injection" in doc_routes(d, low, high) for d in attacks)
            reachable = sum(
                bool(doc_routes(d, low, high) & {"injection", "stage2"}) for d in attacks
            )
            to_stage2 = sum(
                chunk_route(s, r, low, high) == "stage2"
                for d in benign
                for s, r in zip(d.stage1, d.rule, strict=True)
            )
            blocked = sum("injection" in doc_routes(d, low, high) for d in benign)
            rows.append(
                {
                    "theta_low": low,
                    "theta_high": high,
                    "confirmed_recall": confirmed / max(len(attacks), 1),
                    "max_recall": reachable / max(len(attacks), 1),
                    "benign_stage2_rate": to_stage2 / max(benign_chunks, 1),
                    "confirmed_fpr": blocked / max(len(benign), 1),
                }
            )
    return rows


def _row(d: DocScore) -> dict:
    top = max(range(len(d.stage1)), key=lambda i: d.rule[i], default=None)
    x = d.ex.extra
    return {
        "id": d.doc_id,
        "concealment": x.get("concealment"),
        "lang": x.get("lang"),
        "fmt": d.ex.fmt,
        "hard_negative": bool(x.get("hard_negative")),
        "chunks": len(d.stage1),
        "stage1_max": max(d.stage1, default=0.0),
        "rule_max": max(d.rule, default=0.0),
        "rule_reason": d.rule_reasons[top] if top is not None else "",
    }


def missed_attacks(docs: list[DocScore]) -> list[dict]:
    """지금 설정에서 바로 공격도 아니고 2차로도 안 가는 공격 문서"""
    return [
        _row(d)
        for d in docs
        if d.is_attack and not doc_routes(d, THETA_LOW, THETA_HIGH) & {"injection", "stage2"}
    ]


def confident_fps(docs: list[DocScore]) -> list[dict]:
    """1차가 확신했거나 지금 설정에서 바로 공격으로 막히는 정상 문서"""
    return [
        _row(d)
        for d in docs
        if not d.is_attack
        and (
            max(d.stage1, default=0.0) >= FP_CONFIDENT
            or "injection" in doc_routes(d, THETA_LOW, THETA_HIGH)
        )
    ]


def suggest(rows: list[dict]) -> dict | None:
    """2차 예산·바로 차단 오탐 상한을 지키면서 최대 recall 이 가장 높은 줄 (같으면 2차가 적은 쪽)"""
    ok = [
        r
        for r in rows
        if r["benign_stage2_rate"] <= STAGE2_BUDGET and r["confirmed_fpr"] <= CONFIRMED_FPR_MAX
    ]
    if not ok:
        return None
    return max(ok, key=lambda r: (r["max_recall"], r["confirmed_recall"], -r["benign_stage2_rate"]))


def render(r: dict) -> str:
    m = r["doc_metrics"]
    lines = [
        f"# 1차 평가: {r['model']}",
        "",
        f"- 데이터: `{r['data']}` · 문서 {r['n_docs']}개 · 청크 {r['n_chunks']}개 · "
        f"청크당 {r['ms_per_chunk']:.1f}ms · 룰 {'켬' if r['rule'] else '끔'}",
        "",
        "## 문서 단위 (1차만, 0.5 기준)",
        "",
        "| recall | precision | fpr | f1 | tp | fp | tn | fn |",
        "|---|---|---|---|---|---|---|---|",
        f"| {m['recall']:.3f} | {m['precision']:.3f} | {m['fpr']:.3f} | {m['f1']:.3f} "
        f"| {int(m['tp'])} | {int(m['fp'])} | {int(m['tn'])} | {int(m['fn'])} |",
        "",
        "## 나눠 보기 (공격은 recall, 정상은 fpr)",
        "",
        "| 묶음 | 개수 | 비율 |",
        "|---|---|---|",
        *[f"| {k} | {v['n']} | {v['rate']:.1%} |" for k, v in r["breakdown"].items()],
        "",
        "## 문서 길이별 공격 recall",
        "",
        "| 문서 | 공격 개수 | recall |",
        "|---|---|---|",
        *[f"| {k} | {v['n']} | {v['recall']:.1%} |" for k, v in r["length"].items()],
        "",
        f"## 경계값 표 (지금 설정: θ_low {THETA_LOW}, θ_high {THETA_HIGH})",
        "",
        "| θ_low | θ_high | 확정 recall | 최대 recall | 정상 청크 2차 비율 | 바로 공격 fpr |",
        "|---|---|---|---|---|---|",
        *[
            f"| {x['theta_low']} | {_high(x['theta_high'])} | {x['confirmed_recall']:.1%} "
            f"| {x['max_recall']:.1%} | {x['benign_stage2_rate']:.1%} | {x['confirmed_fpr']:.1%} |"
            for x in r["cascade"]
        ],
        "",
    ]
    s = r["suggestion"]
    if s:
        lines.append(
            f"추천: θ_low {s['theta_low']}, θ_high {_high(s['theta_high'])} "
            f"(2차 비율 {s['benign_stage2_rate']:.1%} ≤ {STAGE2_BUDGET:.0%}, "
            f"바로 공격 fpr {s['confirmed_fpr']:.1%} ≤ {CONFIRMED_FPR_MAX:.0%}, "
            f"최대 recall {s['max_recall']:.1%})"
        )
    else:
        lines.append("추천: 2차 예산과 오탐 상한을 함께 지키는 조합이 없습니다")
    head = "| id | 숨김 | 언어 | 형식 | 헷갈림 | 청크 | 1차 최고 | 룰 최고 | 룰 규칙 |"
    for title, rows in (
        (f"못 잡은 공격 (θ_low {THETA_LOW}, θ_high {THETA_HIGH})", r["missed_attacks"]),
        (f"확신 오탐 (1차 ≥ {FP_CONFIDENT} 또는 바로 공격)", r["confident_fps"]),
    ):
        lines += [
            "",
            f"## {title}: {len(rows)}개",
            "",
            head,
            "|---|---|---|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {x['id']} | {x['concealment'] or '-'} | {x['lang']} | {x['fmt']} "
            f"| {'예' if x['hard_negative'] else ''} | {x['chunks']} | {x['stage1_max']:.3f} "
            f"| {x['rule_max']:.2f} | {x['rule_reason'] or '-'} |"
            for x in rows
        ]
    return "\n".join(lines) + "\n"


def _high(h: float | None) -> str:
    return "없음" if h is None else str(h)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
