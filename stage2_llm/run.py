"""데이터셋을 문서 단위로 판정하고 결과를 저장한다.

실행
  uv run python -m stage2_llm.run --model exaone                        # 내 평가 세트
  uv run python -m stage2_llm.run --model exaone --data datasets/v0.1   # 팀 데이터 폴더
  uv run python -m stage2_llm.run --model exaone --data data/v0.1.jsonl --sample 200

--data 로 받는 것
  1) 팀 데이터 폴더 (metadata_ko.jsonl / metadata_en.jsonl + 문서 파일)
     {"id", "context", "label": 0|1, "language", "file_path"} → 문서 파일 내용을 읽는다
     (stage1/convert.py 와 같은 규칙)
  2) jsonl 파일 한 개. 한 줄 = 문서 하나
     text(없으면 context), label(benign/injection 또는 0/1), lang(또는 language),
     fmt, type, hard_negative, attack_type, concealment 는 있으면 쓴다.
문서 판정 = 청크 중 하나라도 injection 이면 injection.

결과: stage2_llm/results/{데이터이름}/{모델}_{프롬프트버전}.jsonl  (+ .meta.json)
표:   uv run python -m stage2_llm.report --name {데이터이름}
"""

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import ollama

from common.config import SEED, set_seed
from stage2_llm.config import MODELS, PROMPT_VERSION
from stage2_llm.judge import judge_document

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
DEFAULT_DATA = HERE / "evalsets" / "eval_v0.1.jsonl"
LABEL_ALIASES = {
    "malicious": "injection",
    0: "benign",
    1: "injection",
    "0": "benign",
    "1": "injection",
}
FORMATS = {".txt": "txt", ".md": "md", ".html": "html", ".htm": "html"}


class DataError(ValueError):
    """데이터 형식 오류. 어느 파일 몇 번째 줄인지 메시지에 담는다"""


def _read_jsonl(path: Path) -> list[tuple[dict, str]]:
    rows = []
    with path.open(encoding="utf-8-sig") as f:
        for n, line in enumerate(f, start=1):
            if line.strip():
                try:
                    rows.append((json.loads(line), f"{path.name}:{n}"))
                except json.JSONDecodeError as e:
                    raise DataError(f"{path.name}:{n}: JSON 이 아닙니다 ({e.msg})") from e
    return rows


def doc_type(rec: dict, label: str) -> str:
    """유형별 정답률을 볼 때 쓰는 분류. 데이터에 있는 칸에서 고른다"""
    if rec.get("type"):
        return str(rec["type"])
    if rec.get("hard_negative"):
        return "hard_negative"
    if label == "benign":
        return "benign"
    return str(rec.get("concealment") or rec.get("attack_type") or "injection")


def normalize_row(rec: dict, where: str, base: Path | None) -> dict:
    """팀 데이터(metadata) 와 학습 형식(convert 결과) 을 같은 모양으로 맞춘다"""
    if "id" not in rec:
        raise DataError(f"{where}: id 칸이 없습니다")
    label = LABEL_ALIASES.get(rec.get("label"), rec.get("label"))
    if label not in ("benign", "injection"):
        raise DataError(f"{where}: label 은 0/1 또는 benign/injection 이어야 합니다")

    file_path = rec.get("file_path")
    text = rec.get("text")
    if text is None and base is not None and file_path:
        doc = base / file_path
        if not doc.is_file():
            raise DataError(f"{where}: 문서 파일이 없습니다: {doc}")
        text = doc.read_text(encoding="utf-8-sig")
    if text is None:
        text = rec.get("context")
    if not isinstance(text, str):
        raise DataError(f"{where}: text·context·file_path 중 읽을 내용이 없습니다")

    fmt = rec.get("fmt")
    if not fmt:
        fmt = FORMATS.get(Path(file_path).suffix.lower(), "txt") if file_path else "txt"
    return {
        "id": str(rec["id"]),
        "text": text,
        "label": label,
        "lang": rec.get("lang") or rec.get("language"),
        "fmt": fmt,
        "type": doc_type(rec, label),
        "attack_type": rec.get("attack_type"),
    }


def load_rows(path: str | Path) -> list[dict]:
    """jsonl 파일 또는 metadata_*.jsonl 이 든 폴더를 읽는다"""
    path = Path(path)
    if path.is_dir():
        metas = sorted(path.rglob("metadata_*.jsonl"))
        if not metas:
            raise DataError(f"{path} 아래에 metadata_*.jsonl 이 없습니다")
        files = [(m, m.parent) for m in metas]
    elif path.is_file():
        files = [(path, path.parent)]
    else:
        raise DataError(f"없는 경로입니다: {path}")
    rows = [normalize_row(rec, where, base) for f, base in files for rec, where in _read_jsonl(f)]
    dup = [k for k, v in Counter(r["id"] for r in rows).items() if v > 1]
    if dup:
        raise DataError(f"id 가 중복됩니다: {', '.join(sorted(dup)[:5])}")
    return rows


def sample_rows(rows: list[dict], n: int, seed: int) -> list[dict]:
    """(언어, 라벨) 비율을 유지한 채 n 개를 고른다. 같은 seed 면 모델이 달라도 같은 문서"""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["lang"], r["label"])].append(r)
    rng = random.Random(seed)
    picked = []
    for key in sorted(groups, key=str):
        g = groups[key]
        k = max(1, round(n * len(g) / len(rows)))
        picked += rng.sample(g, min(k, len(g)))
    return sorted(picked, key=lambda r: r["id"])


def vram_gb(model_name: str) -> float | None:
    """지금 GPU 에 올라가 있는 이 모델의 VRAM(GB). 못 읽으면 None"""
    try:
        for m in ollama.ps().models:
            if m.model == model_name or m.name == model_name:
                return round(m.size_vram / 1024**3, 2)
    except Exception:
        return None
    return None


def main() -> None:
    set_seed()
    parser = argparse.ArgumentParser(description="2차 LLM 판정 실험")
    parser.add_argument("--model", default="exaone", choices=sorted(MODELS))
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="jsonl 파일 또는 팀 데이터 폴더")
    parser.add_argument("--name", default=None, help="결과 폴더 이름 (기본: 데이터 파일·폴더 이름)")
    parser.add_argument("--sample", type=int, default=None, help="언어·라벨 비율 유지하며 N개만")
    parser.add_argument("--limit", type=int, default=None, help="앞에서 몇 개만 (시험용)")
    parser.add_argument("--resume", action="store_true", help="이미 판정한 문서는 건너뛰고 이어서")
    args = parser.parse_args()

    try:
        rows = load_rows(args.data)
    except DataError as e:
        parser.exit(2, f"오류: {e}\n")
    if args.sample:
        rows = sample_rows(rows, args.sample, SEED)
    if args.limit:
        rows = rows[: args.limit]

    profile = MODELS[args.model]
    data_path = Path(args.data)
    name = args.name or (data_path.name if data_path.is_dir() else data_path.stem)
    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_{PROMPT_VERSION}"
    out_path = out_dir / f"{stem}.jsonl"

    done: set[str] = set()
    if args.resume and out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    todo = [r for r in rows if r["id"] not in done]
    langs = Counter((r["lang"], r["label"]) for r in rows)
    print(f"데이터 {len(rows)}개 (이어서 {len(done)}개 건너뜀) · {dict(langs)}")

    judge_document("warmup", "hello", profile)  # 모델을 GPU 에 미리 올림 (저장 안 함)
    vram = vram_gb(profile["name"])

    correct = 0
    started = time.time()
    with open(out_path, "a" if args.resume else "w", encoding="utf-8") as out:
        for n, row in enumerate(todo, start=1):
            gold = row["label"]
            try:
                judged = judge_document(row["id"], row["text"], profile, fmt=row["fmt"])
            except Exception as e:  # 너무 긴 문서 등. 기록만 하고 다음 문서로
                print(f"⚠️  {row['id']} 건너뜀: {type(e).__name__}: {e}")
                continue
            pred = "injection" if any(r.label == "injection" for _, r, _ in judged) else "benign"
            correct += pred == gold
            record = {
                "id": row["id"],
                "lang": row["lang"],
                "type": row["type"],
                "gold_label": gold,
                "gold_attack_type": row.get("attack_type"),
                "pred_label": pred,
                "model": profile["name"],
                "prompt_version": PROMPT_VERSION,
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "result": r.model_dump(),
                        "info": asdict(i),
                        "highlight": (
                            c.raw_text[r.evidence_span.start : r.evidence_span.end]
                            if r.evidence_span
                            else None
                        ),
                    }
                    for c, r, i in judged
                ],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            latency = sum(r.latency_ms for _, r, _ in judged)
            elapsed = time.time() - started
            eta = elapsed / n * (len(todo) - n)
            mark = "✅" if pred == gold else "❌"
            print(
                f"[{n}/{len(todo)}] {mark} {row['id']}  예측={pred}  정답={gold}  "
                f"{latency:.0f}ms  (남은 시간 약 {eta / 60:.0f}분)"
            )

    meta = {
        "model_key": args.model,
        "model": profile["name"],
        "think": profile.get("think"),
        "prompt_version": PROMPT_VERSION,
        "data": str(args.data),
        "sample": args.sample,
        "n_docs": len(rows),
        "vram_gb": vram,
        "elapsed_s": round(time.time() - started, 1),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / f"{stem}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    vram_text = f"{vram}GB" if vram is not None else "확인 못 함"
    print(f"이번 실행 정답 {correct}/{len(todo)} · VRAM {vram_text} · 저장: {out_path}")
    print(f"표 만들기: uv run python -m stage2_llm.report --name {name}")


if __name__ == "__main__":
    main()
