"""3단계 후보 모델 비교. 같은 데이터·같은 설정으로 학습해서 표 하나로 정리한다.

    uv run python -m stage1.compare --data data/v0.1.jsonl --report data/compare_v0.1.md

데이터가 작으면(v0.1 = 100개) val 이 10개 남짓이라 한 번 나눠서는 점수가 크게 흔들린다.
그래서 train/val 을 seed 만 바꿔 --runs 번 나누고(기본 5번, val 20%), 매번 새로 학습해서
평균 ± 표준편차를 낸다. 세 모델 모두 같은 seed 들을 쓰므로 같은 문제지로 비교된다.

표 칸
  파라미터      모델 크기 (백만 개)
  recall …     val 지표의 평균 ± 표준편차 (각 run 에서 f1 이 가장 좋은 epoch 기준)
  판정 속도     조각 1개씩 넣었을 때 평균 ms (실제 pipeline 처럼 1개씩). 첫 run 모델로 잰다
  학습 시간     run 1번 평균 (초)
  384 초과     특수 토큰 포함 384 토큰을 넘어 잘리는 샘플 수 (토크나이저마다 다르다)

체크포인트는 임시 폴더에 저장했다가 측정이 끝나면 지운다 (모델 1개 ≈ 1GB × 15번).
표는 --report 로 저장하고, 숫자 전체는 같은 이름의 .json 으로 저장한다.
"""

import argparse
import gc
import json
import statistics
import tempfile
import time
from pathlib import Path

from common.config import CHUNK_SIZE_TOKENS
from stage1 import train as train_mod
from stage1.dataset import load_jsonl, split_examples

CANDIDATES = ("microsoft/mdeberta-v3-base", "klue/roberta-base", "FacebookAI/xlm-roberta-base")
METRICS = ("recall", "precision", "fpr", "f1")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3단계 후보 모델 비교")
    p.add_argument("--data", type=Path, required=True, help="학습 JSONL (stage1.convert 결과 등)")
    p.add_argument("--models", default=",".join(CANDIDATES), help="쉼표로 구분한 모델 이름")
    p.add_argument("--runs", type=int, default=5, help="train/val 을 다르게 나눠 학습할 횟수")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--speed-samples", type=int, default=50, help="판정 속도를 잴 샘플 수")
    p.add_argument("--report", type=Path, help="결과 표를 저장할 Markdown 파일")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> list[dict]:
    args = parse_args(argv)
    examples = load_jsonl(args.data)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = list(range(args.runs))
    print(f"데이터 {len(examples)}개, 모델 {len(models)}개 × {args.runs}번")

    results = []
    for name in models:
        print(f"\n===== {name} =====")
        try:
            results.append(evaluate_model(name, examples, seeds, args))
        except Exception as e:  # 한 모델이 실패해도 나머지는 계속 비교한다
            print(f"{name}: 실패 ({type(e).__name__}: {e})")
            results.append({"model": name, "error": f"{type(e).__name__}: {e}"})
        free_gpu()

    table = render(results, args, len(examples))
    print("\n" + table)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(table, encoding="utf-8", newline="\n")
        args.report.with_suffix(".json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"저장: {args.report}")
    return results


def evaluate_model(name: str, examples, seeds: list[int], args) -> dict:
    runs, speed, n_params, n_long = [], None, None, None
    for seed in seeds:
        _, val = split_examples(examples, args.val_ratio, seed)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / name.replace("/", "__")
            t0 = time.perf_counter()
            best = train_mod.main(
                [
                    "--data", str(args.data),
                    "--model", name,
                    "--out", str(out),
                    "--epochs", str(args.epochs),
                    "--lr", str(args.lr),
                    "--batch-size", str(args.batch_size),
                    "--val-ratio", str(args.val_ratio),
                    "--seed", str(seed),
                ]
            )  # fmt: skip
            best["train_sec"] = time.perf_counter() - t0
            runs.append(best)
            if speed is None:  # 첫 run 모델로 크기·속도·잘림 수를 잰다
                speed, n_params, n_long = measure(out, val, examples, args.speed_samples)
        free_gpu()

    summary = {"model": name, "params_m": n_params / 1e6, "runs": runs, **speed}
    for k in (*METRICS, "train_sec"):
        vals = [r[k] for r in runs]
        summary[k] = statistics.mean(vals)
        summary[f"{k}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    summary["n_truncated"] = n_long
    return summary


def measure(model_dir: Path, val, examples, n_samples: int) -> tuple[dict, int, int]:
    """조각 1개씩 판정할 때의 속도(ms), 파라미터 수, 384 토큰을 넘는 샘플 수."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    n_long = sum(1 for ex in examples if len(tok(ex.text)["input_ids"]) > CHUNK_SIZE_TOKENS)

    texts = [ex.text for ex in (val or examples)][:n_samples]
    times = []
    with torch.inference_mode():
        for i, text in enumerate([texts[0]] * 3 + texts):  # 앞 3번은 예열(측정 안 함)
            enc = tok(text, truncation=True, max_length=CHUNK_SIZE_TOKENS, return_tensors="pt")
            enc = enc.to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                model(**enc)
            if device == "cuda":
                torch.cuda.synchronize()
            if i >= 3:
                times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    speed = {
        "ms_mean": statistics.mean(times),
        "ms_p95": times[min(len(times) - 1, int(len(times) * 0.95))],
        "device": device,
    }
    del model
    return speed, n_params, n_long


def free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def render(results: list[dict], args, n_data: int) -> str:
    lines = [
        f"데이터 {args.data} ({n_data}개), val {args.val_ratio:.0%} × {args.runs}번, "
        f"epoch {args.epochs}, lr {args.lr}, batch {args.batch_size}",
        "",
        "| 모델 | 파라미터(M) | recall | precision | fpr | f1 "
        "| 판정 속도 ms (평균/p95) | 학습 시간(초/run) | 384 초과 |",
        "|---|---:|---|---|---|---|---:|---:|---:|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['model']} | 실패: {r['error'][:80]} |||||||||")  # 전체는 .json
            continue
        cells = [f"{r[k]:.3f} ± {r[f'{k}_std']:.3f}" for k in METRICS]
        lines.append(
            f"| {r['model']} | {r['params_m']:.0f} | {' | '.join(cells)} "
            f"| {r['ms_mean']:.1f} / {r['ms_p95']:.1f} | {r['train_sec']:.0f} "
            f"| {r['n_truncated']} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
