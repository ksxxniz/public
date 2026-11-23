"""전체 파이프라인 평가 (6단계).

  python -m pipeline.evaluate --data data/v1/validation.jsonl --stage1 real --stage2 real
  python -m pipeline.evaluate --data data/v1/validation.jsonl --stage1 stub --stage2 stub --limit 50
  python -m pipeline.evaluate --data ... --stage1 real --stage2 real --theta-high none \
      --report out.md

stage1.evaluate 는 룰·1차만 보고 "2차로 넘겼나"까지 센다. 이 스크립트는 문서를
오케스트레이터에 그대로 넣어 전처리 → 룰 → 1차 → 2차 → 합치기 → 정책까지 거친
최종 판정을 정답과 비교한다.

보는 것
1. 문서 단위 성능
   - 엄격: 최종 판정이 공격인 문서만 "잡았다"로 센다
   - 표시 포함: 공격 + 경고(warn) + 장애 처리(fail-safe) 를 모두 "잡았다"로 센다
2. 숨김·언어·형식별 성적
3. 누가 잡았나: 공격 문서를 막은 줄 (룰 0.9 / 1차 θ_high / 2차)
4. 2차 경로별 호출 수, 장애 처리 수, 문서당 시간 (p50·p95)
5. 공격 문장 통과율: 정책을 적용한 뒤에도 공격 문장(answer)이 답변 LLM 입력에
   그대로 남은 공격 문서 비율. 조각 경계에 걸친 문장은 남았어도 못 셀 수 있다
6. 놓친 공격·오탐 목록

평가는 매번 새로 검사한다 (판정 저장·재사용·로그 끔).
"""

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from common import config
from common.console import safe_stdio
from pipeline import merge as merge_mod
from pipeline import version as version_mod
from pipeline.run import add_component_args, build
from stage1.dataset import Example, load_jsonl

NO_BLOCK = 1.01  # --theta-high none: 1차 점수(≤1)가 절대 넘지 못하는 값 → 1차는 혼자 막지 않음
PROGRESS_EVERY = 25


@dataclass
class DocResult:
    ex: Example
    final: str  # "injection" | "benign"
    action: str  # allow / warn / mask / block
    flagged: bool  # 공격 또는 경고 또는 fail-safe
    caught_by: str  # rule / stage1 / stage2 / warn / fail_safe / blocked / ""
    merges: list[str]  # 조각별 합치기 줄
    s2_calls: int
    fallback: bool
    blocked_reason: str | None
    answer_left: bool | None  # 공격 문장이 답변 LLM 입력에 남았나 (모르면 None)
    total_ms: float
    stage2_ms: float
    routes: dict[str, int]
    warnings: list[str]

    @property
    def is_attack(self) -> bool:
        return self.ex.label == "injection"

    @property
    def doc_id(self) -> str:
        return str(self.ex.extra.get("id") or self.ex.doc)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="전체 파이프라인 평가 (6단계)")
    p.add_argument("--data", type=Path, required=True, help="평가 JSONL (검증 또는 테스트 몫)")
    p.add_argument("--limit", type=int, help="앞에서 N개만 (시험용)")
    p.add_argument("--report", type=Path, help="결과를 이 파일에도 저장 (.md). 같은 이름 .json 도")
    p.add_argument(
        "--theta-high",
        help="이번 평가에서만 θ_high 를 바꾼다. 숫자 또는 none(1차는 혼자 막지 않음). "
        f"생략 시 config 값 {config.THETA_HIGH}",
    )
    add_component_args(p)
    args = p.parse_args(argv)
    args.no_db, args.no_cache = True, True  # 평가는 매번 새로 검사
    return args


def set_theta_high(value: str | None) -> float:
    """config·merge·version 이 같은 값을 보도록 셋 다 바꾼다 (이 프로세스 안에서만)"""
    if value is None:
        return config.THETA_HIGH
    high = NO_BLOCK if value.lower() == "none" else float(value)
    if not config.THETA_LOW < high <= NO_BLOCK:
        raise SystemExit(f"--theta-high 는 θ_low({config.THETA_LOW}) 보다 크고 1 이하여야 합니다")
    config.THETA_HIGH = high
    merge_mod.THETA_HIGH = high
    version_mod.THETA_HIGH = high  # pipeline_ver 에도 바뀐 값이 찍히게
    return high


def caught_by(merges: list[str], fallback: bool, any_warn: bool, blocked: bool) -> str:
    """문서를 잡은 줄. 여러 조각이면 룰 → 1차 → 2차 순으로 앞선 것"""
    if blocked:
        return "blocked"
    for key, name in (("rule>=", "rule"), ("stage1>=", "stage1"), ("stage2=injection", "stage2")):
        if any(m.startswith(key) for m in merges):
            return name
    if any_warn:
        return "warn"
    if fallback:
        return "fail_safe"
    return ""


def evaluate_doc(ex: Example, orch) -> DocResult:
    rep = orch.inspect_text(ex.raw, doc_id=str(ex.extra.get("id") or ex.doc), fmt=ex.fmt)
    doc = rep.doc
    merges = [str(v.meta.get("merge", "")) for v in doc.verdicts]
    any_warn = any(m.startswith("stage2=benign,rule") for m in merges)
    blocked = rep.blocked_reason is not None
    injected = doc.final_label == "injection"
    answer = ex.extra.get("answer")
    answer_left = None
    if ex.label == "injection" and isinstance(answer, str) and answer and answer in ex.raw:
        answer_left = not blocked and any(answer in t for t in rep.masked_texts.values())
    return DocResult(
        ex=ex,
        final=doc.final_label,
        action=doc.action,
        flagged=injected or any_warn or doc.fallback,
        caught_by=caught_by(merges, doc.fallback, any_warn, blocked),
        merges=merges,
        s2_calls=sum(rep.routes.values()),
        fallback=doc.fallback,
        blocked_reason=rep.blocked_reason,
        answer_left=answer_left,
        total_ms=rep.timings.total_ms,
        stage2_ms=rep.timings.stage2_ms,
        routes=dict(rep.routes),
        warnings=list(rep.warnings),
    )


# ── 집계 ──


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def metrics(docs: list[DocResult], strict: bool) -> dict:
    def hit(d: DocResult) -> bool:
        return d.final == "injection" if strict else d.flagged

    tp = sum(d.is_attack and hit(d) for d in docs)
    fn = sum(d.is_attack and not hit(d) for d in docs)
    fp = sum(not d.is_attack and hit(d) for d in docs)
    tn = sum(not d.is_attack and not hit(d) for d in docs)
    recall, precision = _div(tp, tp + fn), _div(tp, tp + fp)
    return {
        "recall": recall,
        "precision": precision,
        "fpr": _div(fp, fp + tn),
        "f1": _div(2 * precision * recall, precision + recall),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def breakdown(docs: list[DocResult]) -> list[dict]:
    """공격은 엄격 recall, 정상은 엄격 fpr"""
    groups: dict[str, list[DocResult]] = defaultdict(list)
    for d in docs:
        x = d.ex.extra
        if d.is_attack:
            groups[f"공격 · 숨김 {x.get('concealment') or '-'}"].append(d)
            groups[f"공격 · {x.get('lang') or '-'}"].append(d)
            groups[f"공격 · {d.ex.fmt}"].append(d)
        else:
            groups[f"정상 · {'헷갈림' if x.get('hard_negative') else '일반'}"].append(d)
            groups[f"정상 · {x.get('lang') or '-'}"].append(d)
            groups[f"정상 · {d.ex.fmt}"].append(d)
    return [
        {"group": k, "n": len(v), "rate": _div(sum(d.final == "injection" for d in v), len(v))}
        for k, v in sorted(groups.items())
    ]


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]


def summary(docs: list[DocResult], elapsed_s: float) -> dict:
    attacks = [d for d in docs if d.is_attack]
    routes: Counter = Counter()
    for d in docs:
        routes.update(d.routes)
    total = [d.total_ms for d in docs]
    with_s2 = [d.total_ms for d in docs if d.s2_calls]
    known = [d for d in attacks if d.answer_left is not None]
    return {
        "docs": len(docs),
        "attacks": len(attacks),
        "elapsed_s": elapsed_s,
        "strict": metrics(docs, strict=True),
        "flagged": metrics(docs, strict=False),
        "caught_by": dict(Counter(d.caught_by or "놓침" for d in attacks)),
        "benign_flag_by": dict(Counter(d.caught_by for d in docs if not d.is_attack and d.flagged)),
        "routes": dict(routes),
        "docs_to_stage2": sum(bool(d.s2_calls) for d in docs),
        "benign_docs_to_stage2": sum(bool(d.s2_calls) for d in docs if not d.is_attack),
        "fallback_docs": sum(d.fallback for d in docs),
        "blocked_docs": sum(d.blocked_reason is not None for d in docs),
        "answer_known": len(known),
        "answer_left": sum(bool(d.answer_left) for d in known),
        "latency_ms": {
            "p50": _pct(total, 50),
            "p95": _pct(total, 95),
            "p50_with_stage2": _pct(with_s2, 50),
            "p95_with_stage2": _pct(with_s2, 95),
            "stage2_total_s": sum(d.stage2_ms for d in docs) / 1000,
        },
        "breakdown": breakdown(docs),
    }


def _row(d: DocResult) -> dict:
    x = d.ex.extra
    return {
        "id": d.doc_id,
        "concealment": x.get("concealment"),
        "lang": x.get("lang"),
        "fmt": d.ex.fmt,
        "hard_negative": bool(x.get("hard_negative")),
        "final": d.final,
        "action": d.action,
        "caught_by": d.caught_by,
        "merges": sorted(set(d.merges)),
        "s2_calls": d.s2_calls,
        "answer_left": d.answer_left,
        "total_ms": round(d.total_ms, 1),
        "warnings": d.warnings[:3],
    }


def lists(docs: list[DocResult]) -> dict:
    return {
        "missed_attacks": [_row(d) for d in docs if d.is_attack and d.final != "injection"],
        "false_positives": [_row(d) for d in docs if not d.is_attack and d.flagged],
        "answer_left": [_row(d) for d in docs if d.is_attack and d.answer_left],
    }


# ── 출력 ──


def render(r: dict) -> str:
    s, m, f = r["summary"], r["summary"]["strict"], r["summary"]["flagged"]
    lat = s["latency_ms"]
    lines = [
        f"# 전체 파이프라인 평가: {r['pipeline_ver']}",
        "",
        f"- 데이터: `{r['data']}` · 문서 {s['docs']}개 (공격 {s['attacks']}) · "
        f"걸린 시간 {s['elapsed_s'] / 60:.1f}분",
        f"- 부품: 1차 {r['stage1']} · 2차 {r['stage2']} ({r['model']}) · 정책 {r['policy']} · "
        f"θ_low {r['theta_low']} · θ_high {r['theta_high_label']}",
        "",
        "## 문서 단위",
        "",
        "| 기준 | recall | precision | fpr | f1 | tp | fp | tn | fn |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, x in (("엄격 (공격 판정만)", m), ("표시 포함 (공격+경고+장애)", f)):
        lines.append(
            f"| {name} | {x['recall']:.3f} | {x['precision']:.3f} | {x['fpr']:.3f} | "
            f"{x['f1']:.3f} | {x['tp']} | {x['fp']} | {x['tn']} | {x['fn']} |"
        )
    left = s["answer_left"]
    lines += [
        "",
        f"공격 문장 통과율: {_div(left, s['answer_known']):.1%} "
        f"({left}/{s['answer_known']}, 정책 적용 뒤 답변 LLM 입력에 공격 문장이 남은 공격 문서)",
        "",
        "## 누가 잡았나 (공격 문서)",
        "",
        "| 줄 | 문서 |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in sorted(s["caught_by"].items(), key=lambda kv: -kv[1])],
        "",
        f"정상 문서를 표시한 줄: {s['benign_flag_by'] or '없음'}",
        "",
        "## 2차와 시간",
        "",
        f"- 2차로 간 문서 {s['docs_to_stage2']}개 (정상 {s['benign_docs_to_stage2']}개) · "
        f"경로별 호출 {s['routes']}",
        f"- 장애 처리(fail-safe) 문서 {s['fallback_docs']}개 · "
        f"문서 전체 차단 {s['blocked_docs']}개",
        f"- 문서당 시간 p50 {lat['p50']:.0f}ms · p95 {lat['p95']:.0f}ms "
        f"(2차 간 문서만 p50 {lat['p50_with_stage2']:.0f}ms · p95 {lat['p95_with_stage2']:.0f}ms) "
        f"· 2차 합계 {lat['stage2_total_s']:.0f}초",
        "",
        "## 나눠 보기 (공격은 recall, 정상은 fpr · 엄격)",
        "",
        "| 묶음 | 개수 | 비율 |",
        "|---|---|---|",
        *[f"| {b['group']} | {b['n']} | {b['rate']:.1%} |" for b in s["breakdown"]],
    ]
    head = "| id | 숨김 | 언어 | 형식 | 헷갈림 | 최종 | 조치 | 잡은 줄 | 2차 | 합치기 줄 |"
    for title, rows in (
        ("놓친 공격 (엄격)", r["missed_attacks"]),
        ("오탐 (표시 포함)", r["false_positives"]),
        ("공격 문장이 남은 문서", r["answer_left"]),
    ):
        lines += [
            "",
            f"## {title}: {len(rows)}개",
            "",
            head,
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {x['id']} | {x['concealment'] or '-'} | {x['lang']} | {x['fmt']} "
            f"| {'예' if x['hard_negative'] else ''} | {x['final']} | {x['action']} "
            f"| {x['caught_by'] or '-'} | {x['s2_calls']} | {', '.join(x['merges'])} |"
            for x in rows
        ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    safe_stdio()
    args = parse_args(argv)
    high = set_theta_high(args.theta_high)
    examples = load_jsonl(args.data)[: args.limit]
    orch = build(args)

    docs: list[DocResult] = []
    t0 = time.perf_counter()
    for i, ex in enumerate(examples, start=1):
        docs.append(evaluate_doc(ex, orch))
        if i % PROGRESS_EVERY == 0 or i == len(examples):
            done = time.perf_counter() - t0
            print(
                f"  {i}/{len(examples)} · {done:.0f}초 · 남은 시간 약 "
                f"{done / i * (len(examples) - i):.0f}초",
                file=sys.stderr,
            )
    elapsed = time.perf_counter() - t0

    result = {
        "data": str(args.data),
        "pipeline_ver": orch.pipeline_ver,
        "stage1": args.stage1,
        "stage2": args.stage2,
        "model": args.model,
        "policy": args.policy,
        "theta_low": config.THETA_LOW,
        "theta_high": high,
        "theta_high_label": "없음(1차는 혼자 막지 않음)" if high >= NO_BLOCK else str(high),
        "summary": summary(docs, elapsed),
        **lists(docs),
    }
    text = render(result)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
        args.report.with_suffix(".json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"저장: {args.report}", file=sys.stderr)


if __name__ == "__main__":
    main()
