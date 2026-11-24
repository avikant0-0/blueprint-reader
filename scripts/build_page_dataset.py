"""Render drawing-set PDFs into page images, pre-sorted by the keyword baseline.

Usage:
    python scripts/build_page_dataset.py --pdfs data/pdfs --out data/pages

Output:
    data/pages/<label>/<pdf>_p0007.jpg   (+ .txt with the page's text layer)

The pre-sort is only a first guess. Open the folders and MOVE every wrongly
filed page into the right one by hand; that corrected folder tree is the
labelled dataset for train_page_classifier.py. Keep the .txt next to its .jpg.
Hold out whole PDFs for testing (see --test-pdfs in the training script).
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from blueprint_reader.classify import LABELS, classify_text
from blueprint_reader.pdf import load_page, open_pdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=50, help="classifier sees the whole page; low dpi is enough")
    args = ap.parse_args()

    out = Path(args.out)
    for label in LABELS:
        (out / label).mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    for pdf in sorted(Path(args.pdfs).glob("*.pdf")):
        with open_pdf(pdf) as doc:
            for i in range(doc.page_count):
                page = load_page(doc, i, dpi=args.dpi)
                label, _ = classify_text(page.text)
                stem = f"{pdf.stem}_p{i:04d}"
                page.image.save(out / label / f"{stem}.jpg", quality=85)
                (out / label / f"{stem}.txt").write_text(page.text)
                counts[label] += 1
        print(f"{pdf.name}: done")
    print(dict(counts))
    print(f"Now correct the folders under {out} by hand.")


if __name__ == "__main__":
    main()
