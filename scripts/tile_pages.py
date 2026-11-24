"""Cut floor-plan pages of real drawing sets into tiles for labelling.

The detector is trained on CubiCasa5K (clean residential plans). Real
construction sheets look different, so expect accuracy to drop on them. To
measure and then close that gap, label a small set of real tiles:

    python scripts/tile_pages.py --pdfs data/pdfs --out data/real_tiles

Label the tiles in any tool that exports YOLO format (CVAT, Label Studio,
Roboflow) using the class order in configs/classes.yaml, then:
  * score the CubiCasa-only model on them  -> the honest "before" number
  * fine-tune on a train split of them     -> the "after" number
Split by PDF, never by tile, so the test sheets are truly unseen.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from blueprint_reader.classify import classify_text
from blueprint_reader.detect import tile_origins
from blueprint_reader.pdf import load_page, open_pdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=float, default=0.0)
    ap.add_argument("--all-pages", action="store_true",
                    help="tile every page instead of only keyword-classified floor plans")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for pdf in sorted(Path(args.pdfs).glob("*.pdf")):
        with open_pdf(pdf) as doc:
            for i in range(doc.page_count):
                page = load_page(doc, i, dpi=args.dpi)
                if not args.all_pages and classify_text(page.text)[0] != "floor_plan":
                    continue
                w, h = page.image.size
                for y in tile_origins(h, args.tile, args.overlap):
                    for x in tile_origins(w, args.tile, args.overlap):
                        tile = page.image.crop((x, y, min(x + args.tile, w), min(y + args.tile, h)))
                        if tile.convert("L").getextrema()[0] > 200:
                            continue  # blank tile
                        tile.save(out / f"{pdf.stem}_p{i:04d}_x{x}_y{y}.jpg", quality=90)
                        n += 1
    print(f"wrote {n} tiles to {out}")


if __name__ == "__main__":
    main()
