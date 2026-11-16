"""Dump the text of a PDF to a UTF-8 .txt file (one section per page)."""
import sys
from pathlib import Path

import pymupdf


def main(src: Path, dst: Path) -> None:
    doc = pymupdf.open(src)
    parts = [f"\n===== PAGE {i + 1} =====\n{page.get_text()}" for i, page in enumerate(doc)]
    dst.write_text("".join(parts), encoding="utf-8")
    print(f"{src.name}: {len(doc)} pages, {sum(len(p) for p in parts)} chars -> {dst}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
