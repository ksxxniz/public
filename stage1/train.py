"""3단계 1차 분류기 학습.

    uv run python -m stage1.train --data datasets/train.jsonl \
        --model xlm-roberta-base --out checkpoints/stage1/xlmr-lr2e5

  --data 의 샘플을 source_group 기준으로 train/val 로 나눈다 (val 15%, stage1/dataset.py).
  --wandb 를 붙이면 W&B(rag-guard) 에 기록한다 (.env 의 WANDB_ENTITY 사용).

필요한 라이브러리: torch, transformers (mDeBERTa·XLM-R 은 sentencepiece, protobuf 도).
pyproject.toml 에는 아직 없다. 3단계 담당 PC 에서 따로 추가한다.

학습 방식
- 입력은 정리본 (dataset.py 가 normalize_text 를 거쳐 만든다)
- 길이는 특수 토큰 포함 384 토큰 (결정 ④). 넘는 샘플 수를 시작할 때 알려준다
- 정상:공격 = 3:1 이라 그냥 학습하면 "전부 정상"으로 게을러진다.
  공격을 틀렸을 때 벌점을 키운다 (가중 손실, 기본값 = 정상 수 / 공격 수)
- GPU 가 BF16 을 지원하면 BF16 으로 학습한다 (mDeBERTa 는 FP16 이 불안정하다)
- epoch 마다 val 의 recall · precision · fpr · f1 을 출력하고, f1 이 가장 좋은 epoch 를 저장한다

저장 결과 (--out 폴더): 모델 · 토크나이저 · stage1_meta.json
  stage1_meta.json = model_ver(폴더 이름), 베이스 모델, 정규화 버전, 유니코드 버전,
                     하이퍼파라미터, 데이터 개수, val 지표
"""

import argparse
import json
import math
import time
from pathlib import Path

from common.config import CHUNK_SIZE_TOKENS, SEED, WANDB_ENTITY, WANDB_PROJECT, set_seed
from preprocess.normalize import UNICODE_VERSION
from preprocess.normalize import VERSION as NORMALIZE_VERSION
from stage1.dataset import ID2LABEL, LABEL2ID, Example, label_counts, load_jsonl, split_examples
from stage1.metrics import binary_metrics

META_FILE = "stage1_meta.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3단계 1차 분류기 학습")
    p.add_argument("--data", type=Path, required=True, help="학습 JSONL (train/val 자동 분할)")
    p.add_argument("--model", default="xlm-roberta-base", help="베이스 모델 (HF 이름 또는 폴더)")
    p.add_argument("--out", type=Path, required=True, help="저장 폴더. 폴더 이름이 model_ver")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--warmup", type=float, default=0.1, help="전체 스텝 중 워밍업 비율")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--pos-weight", type=float, help="공격 클래스 가중치. 생략 시 정상 수/공격 수")
    p.add_argument("--max-length", type=int, default=CHUNK_SIZE_TOKENS)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--precision", choices=["auto", "bf16", "fp32"], default="auto")
    p.add_argument("--wandb", action="store_true", help="W&B 에 기록")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    set_seed(args.seed)  # 규칙: 학습 스크립트 맨 위에서 호출

    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    train, val = split_examples(load_jsonl(args.data), args.val_ratio, args.seed)
    if not train or not val:
        raise SystemExit(f"데이터가 부족합니다: train {len(train)}개, val {len(val)}개")
    counts = label_counts(train)
    print(f"train {len(train)} {counts} / val {len(val)} {label_counts(val)}")
    if min(counts.values()) == 0:
        raise SystemExit("train 에 benign 과 injection 이 모두 있어야 합니다")
    pos_weight = args.pos_weight or counts["benign"] / counts["injection"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = args.precision == "bf16" or (
        args.precision == "auto" and device == "cuda" and torch.cuda.is_bf16_supported()
    )
    print(f"device={device} bf16={use_bf16} pos_weight={pos_weight:.2f}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    ).to(device)
    report_truncation(tokenizer, train + val, args.max_length)

    def batches(data: list[Example], shuffle: bool):
        order = torch.randperm(len(data)).tolist() if shuffle else list(range(len(data)))
        for i in range(0, len(order), args.batch_size):
            part = [data[j] for j in order[i : i + args.batch_size]]
            enc = tokenizer(
                [ex.text for ex in part],
                truncation=True,
                max_length=args.max_length,  # 특수 토큰 포함 (결정 ④)
                padding=True,
                return_tensors="pt",
            ).to(device)
            yield enc, torch.tensor([ex.label_id for ex in part], device=device)

    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup), total_steps
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight], device=device))

    run = init_wandb(args, pos_weight, counts) if args.wandb else None
    best, best_epoch = None, -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total_loss = time.perf_counter(), 0.0
        for enc, y in batches(train, shuffle=True):
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(**enc).logits
            loss = loss_fn(logits.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        scores = predict(model, batches(val, shuffle=False), torch)
        m = binary_metrics([ex.label_id for ex in val], scores)
        m["train_loss"] = total_loss / steps_per_epoch
        m["epoch_sec"] = time.perf_counter() - t0
        print(
            f"epoch {epoch}: loss {m['train_loss']:.4f} | recall {m['recall']:.3f} "
            f"precision {m['precision']:.3f} fpr {m['fpr']:.3f} f1 {m['f1']:.3f}"
        )
        if run:
            run.log({f"val/{k}": v for k, v in m.items()}, step=epoch)
        if best is None or m["f1"] > best["f1"]:
            best, best_epoch = m, epoch
            save(model, tokenizer, args, best, best_epoch, pos_weight, len(train), len(val))

    print(f"가장 좋은 epoch {best_epoch} (val f1 {best['f1']:.3f}) → {args.out}")
    if run:
        run.summary.update({f"best/{k}": v for k, v in best.items()})
        run.finish()
    return best


def predict(model, batches, torch) -> list[float]:
    """val 샘플마다 injection 확률(risk_score)."""
    model.eval()
    scores: list[float] = []
    with torch.inference_mode():
        for enc, _ in batches:
            probs = torch.softmax(model(**enc).logits.float(), dim=-1)
            scores.extend(probs[:, LABEL2ID["injection"]].tolist())
    return scores


def report_truncation(tokenizer, data: list[Example], max_length: int) -> None:
    """특수 토큰 포함 max_length 를 넘어 잘리는 샘플 수 (데이터 규칙: 샘플은 384 이하)."""
    n_long = sum(1 for ex in data if len(tokenizer(ex.text)["input_ids"]) > max_length)
    if n_long:
        print(f"경고: {n_long}/{len(data)}개 샘플이 {max_length} 토큰을 넘어 잘립니다")


def save(model, tokenizer, args, metrics, epoch, pos_weight, n_train, n_val) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    meta = {
        "model_ver": args.out.name,
        "base_model": args.model,
        "label2id": LABEL2ID,
        "max_length": args.max_length,
        "normalize_version": NORMALIZE_VERSION,
        "unicode_version": UNICODE_VERSION,
        "seed": args.seed,
        "epoch": epoch,
        "hparams": {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "warmup": args.warmup,
            "weight_decay": args.weight_decay,
            "pos_weight": pos_weight,
        },
        "data": {"path": str(args.data), "n_train": n_train, "n_val": n_val},
        "val_metrics": metrics,
    }
    (args.out / META_FILE).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def init_wandb(args, pos_weight: float, counts: dict):
    import wandb

    return wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=args.out.name,
        config={
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "pos_weight": pos_weight,
            "train_counts": counts,
            "normalize_version": NORMALIZE_VERSION,
        },
    )


if __name__ == "__main__":
    main()
