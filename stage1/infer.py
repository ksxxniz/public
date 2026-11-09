"""3단계 1차 분류기 판정. 파이프라인이 청크마다 부른다.

    from stage1.infer import load_classifier
    clf = load_classifier("checkpoints/stage1/xlmr-v1")   # 한 번만 불러 둔다
    result = clf.classify_chunk(chunk)                    # StageResult(stage="stage1")

  python -m stage1.infer --model checkpoints/stage1/xlmr-v1 --file 문서.md   # 확인·시연용

판정 방식
- 본문(chunk.text)과 복원 결과 조각(preprocess.decoded.decoded_inputs)을 각각 점수 매기고
  가장 높은 점수를 청크 점수로 쓴다 (결정 ⑥). 학습도 같은 조각으로 했다 (dataset.add_decoded).
- risk_score = injection 확률. label 은 0.5 기준이다.
  실제로 넘기거나 막는 기준값(T_low, T_high)은 파이프라인이 정한다.
- evidence_span: 복원 조각이 가장 높으면 그 조각의 원문 인코딩 구간.
  본문이 가장 높으면 None (1차는 문장 위치를 짚지 못한다. 2차가 채운다).
- reason: 점수가 나온 곳. "body" 또는 "decoded:base64" 처럼.
- attack_type: None (1차는 정상/공격만 가른다).

모델 폴더는 train.py 가 저장한 것 (모델 · 토크나이저 · stage1_meta.json).
폴더를 주지 않으면 환경변수 STAGE1_MODEL_DIR, 그것도 없으면 DEFAULT_MODEL_DIR.
"""

import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from common.config import ROOT_DIR
from common.schema import Chunk, Span, StageResult
from preprocess.decoded import decoded_inputs
from preprocess.normalize import UNICODE_VERSION
from preprocess.normalize import VERSION as NORMALIZE_VERSION
from stage1.dataset import LABEL2ID
from stage1.train import META_FILE

ENV_VAR = "STAGE1_MODEL_DIR"
DEFAULT_MODEL_DIR = ROOT_DIR / "checkpoints" / "stage1" / "current"
THRESHOLD = 0.5  # label 만 정한다. 넘기기·막기 기준값은 파이프라인 몫


@dataclass(frozen=True)
class _Input:
    chunk: int  # 몇 번째 청크의 입력인가
    text: str
    source: str  # "body" 또는 "decoded:<method>"
    span: Span | None  # 복원 조각이면 원문 인코딩 구간


class Stage1Classifier:
    def __init__(self, model_dir: str | Path, device: str | None = None, batch_size: int = 16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.model_dir = Path(model_dir)
        meta_path = self.model_dir / META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} 가 없습니다. stage1.train 이 저장한 폴더인가요?")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        _check_versions(self.meta)

        self.model_ver: str = self.meta.get("model_ver", self.model_dir.name)
        self.max_length: int = self.meta.get("max_length", 384)
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device).eval()
        self._injection = self.meta.get("label2id", LABEL2ID)["injection"]

    def classify_chunk(self, chunk: Chunk) -> StageResult:
        return self.classify_chunks([chunk])[0]

    def classify_chunks(self, chunks: list[Chunk]) -> list[StageResult]:
        """청크 여러 개를 한꺼번에 묶어 점수 매긴다. 결과는 청크 순서대로."""
        started = time.perf_counter()
        inputs = [inp for k, c in enumerate(chunks) for inp in _inputs(k, c)]
        scores = self.score([inp.text for inp in inputs])
        per_chunk_ms = (time.perf_counter() - started) * 1000 / max(len(chunks), 1)

        best: dict[int, tuple[float, _Input]] = {}
        for inp, s in zip(inputs, scores, strict=True):
            if inp.chunk not in best or s > best[inp.chunk][0]:
                best[inp.chunk] = (s, inp)
        results = []
        for k in range(len(chunks)):
            score, inp = best[k]
            results.append(
                StageResult(
                    stage="stage1",
                    label="injection" if score >= THRESHOLD else "benign",
                    risk_score=score,
                    confidence=max(score, 1 - score),
                    reason=inp.source,
                    attack_type=None,
                    evidence_span=inp.span,
                    latency_ms=per_chunk_ms,
                    model_ver=self.model_ver,
                )
            )
        return results

    def score(self, texts: list[str]) -> list[float]:
        """글마다 injection 확률. 학습과 같은 방식으로 자른다 (특수 토큰 포함 max_length)."""
        torch = self._torch
        out: list[float] = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tokenizer(
                    texts[i : i + self.batch_size],
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                probs = torch.softmax(self.model(**enc).logits.float(), dim=-1)
                out.extend(probs[:, self._injection].tolist())
        # 부동소수 오차로 1 을 살짝 넘는 경우 스키마(0~1) 검사에 걸리지 않게
        return [min(max(s, 0.0), 1.0) for s in out]


def _inputs(k: int, chunk: Chunk) -> list[_Input]:
    out = [_Input(k, chunk.text, "body", None)]
    for d in decoded_inputs(chunk):
        out.append(_Input(k, d.text, f"decoded:{d.method}", d.span))
    return out


def _check_versions(meta: dict) -> None:
    """학습 때와 지금의 정규화·유니코드 버전이 다르면 입력이 어긋나 점수가 틀어질 수 있다."""
    for key, now in (
        ("normalize_version", NORMALIZE_VERSION),
        ("unicode_version", UNICODE_VERSION),
    ):
        trained = meta.get(key)
        if trained and trained != now:
            warnings.warn(
                f"모델은 {key}={trained} 로 학습했는데 지금은 {now} 입니다. 다시 학습을 권합니다",
                RuntimeWarning,
                stacklevel=3,
            )


@lru_cache(maxsize=2)
def _cached(model_dir: str) -> Stage1Classifier:
    return Stage1Classifier(model_dir)


def load_classifier(model_dir: str | Path | None = None) -> Stage1Classifier:
    """모델을 한 번만 불러 재사용한다. 같은 폴더를 다시 부르면 같은 객체."""
    path = model_dir or os.environ.get(ENV_VAR) or DEFAULT_MODEL_DIR
    return _cached(str(Path(path).resolve()))


def main(argv: list[str] | None = None) -> None:
    from preprocess.chunk import split_document
    from preprocess.load import read_document

    p = argparse.ArgumentParser(description="3단계 1차 분류기로 문서 하나를 청크별로 판정")
    p.add_argument("--file", required=True, help="검사할 문서 (.txt/.md/.html)")
    p.add_argument("--model", help=f"모델 폴더. 생략 시 ${ENV_VAR} 또는 {DEFAULT_MODEL_DIR}")
    args = p.parse_args(argv)

    clf = load_classifier(args.model)
    doc = read_document(args.file)
    chunks = split_document(Path(args.file).stem, doc.raw, doc.fmt)
    results = clf.classify_chunks(chunks)
    print(f"[stage1 {clf.model_ver}] {args.file}: 청크 {len(chunks)}개")
    for c, r in zip(chunks, results, strict=True):
        where = f" 근거 {r.evidence_span.start}~{r.evidence_span.end}" if r.evidence_span else ""
        print(f"  {c.chunk_id}: {r.label:9} {r.risk_score:.3f} ({r.reason}){where}")
    top = max(r.risk_score for r in results) if results else 0.0
    print(f"문서 최고 점수 {top:.3f}")


if __name__ == "__main__":
    main()
