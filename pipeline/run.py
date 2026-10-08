"""파이프라인 진입점. --file 로 받은 문서를 단계별로 검사한다. (스텁)

지금은 전처리 결과를 만드는 것까지만 한다. rule/stage1/stage2 는
담당자 작업이 끝나는 대로 이어붙인다.
"""

import argparse
from pathlib import Path

from preprocess.chunk import split_document
from preprocess.load import LoadedDocument, UnsupportedDocumentError, read_document
from preprocess.normalize import VERSION
from preprocess.tokens import TokenizerLoadError


def run(path: str | Path) -> None:
    report(path, read_document(path))


def report(path: str | Path, doc: LoadedDocument) -> None:
    doc_id = Path(path).stem
    chunks = split_document(doc_id, doc.raw, fmt=doc.fmt)
    print(f"[preprocess v{VERSION}] {doc_id} ({doc.fmt}, {doc.encoding}): {len(chunks)}개 청크")
    for c in chunks:
        print(f"  {c.chunk_id}: {len(c.text)}자")


def main() -> None:
    parser = argparse.ArgumentParser(description="prompt-injection-protection 파이프라인")
    parser.add_argument("--file", required=True, help="검사할 문서 경로")
    args = parser.parse_args()
    try:
        doc = read_document(args.file)
    except UnsupportedDocumentError as e:
        parser.exit(2, f"오류: {e}\n")
    except FileNotFoundError:
        parser.exit(2, f"오류: 파일이 없습니다: {args.file}\n")
    except OSError as e:  # 폴더를 넘겼거나 권한이 없는 경우 등
        parser.exit(2, f"오류: 파일을 읽을 수 없습니다: {args.file} ({e.strerror or e})\n")
    try:
        report(args.file, doc)
    except TokenizerLoadError as e:
        parser.exit(2, f"오류: {e}\n")


if __name__ == "__main__":
    main()
