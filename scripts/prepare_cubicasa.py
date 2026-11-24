"""Build a YOLO-format detection dataset from CubiCasa5K.

Usage:
    python scripts/prepare_cubicasa.py --src data/cubicasa5k --out data/yolo \
        --visualize 30

ALWAYS look at the images written to <out>/preview before training. They show
the converted boxes drawn on the plans; if boxes are misplaced, fix the
conversion first (try --opening-mode opening).
"""
from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from blueprint_reader.cubicasa import ClassMap, parse_svg, read_split

COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
          "#008080", "#9a6324", "#800000", "#808000", "#000075"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="extracted cubicasa5k folder")
    ap.add_argument("--out", required=True, help="output YOLO dataset folder")
    ap.add_argument("--classes", default="configs/classes.yaml")
    ap.add_argument("--opening-mode", choices=["full", "opening"], default="full")
    ap.add_argument("--visualize", type=int, default=30,
                    help="number of random samples to draw into <out>/preview")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap samples per split")
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    classes = ClassMap.load(args.classes)
    random.seed(0)
    counts: Counter = Counter()
    skipped = 0
    previews: list[tuple[Path, list]] = []

    for split in ("train", "val", "test"):
        folders = read_split(src / f"{split}.txt")
        if args.limit:
            folders = folders[: args.limit]
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for rel in folders:
            img_path = src / rel / "F1_scaled.png"
            svg_path = src / rel / "model.svg"
            if not img_path.exists() or not svg_path.exists():
                skipped += 1
                continue
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
                boxes = parse_svg(svg_path, w, h, classes, opening_mode=args.opening_mode)
            except Exception as exc:  # a few SVGs in the dataset are malformed
                print(f"skip {rel}: {exc}")
                skipped += 1
                continue
            stem = rel.replace("/", "_")
            dst_img = out / "images" / split / f"{stem}.png"
            if dst_img.exists() or dst_img.is_symlink():
                dst_img.unlink()
            if args.copy:
                shutil.copy(img_path, dst_img)
            else:
                dst_img.symlink_to(img_path.resolve())
            (out / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(b.to_yolo(w, h) for b in boxes))
            for b in boxes:
                counts[(split, classes.names[b.cls])] += 1
            if split == "train":
                previews.append((img_path, boxes))

    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(classes.names)),
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    prev_dir = out / "preview"
    prev_dir.mkdir(exist_ok=True)
    for i, (img_path, boxes) in enumerate(random.sample(previews, min(args.visualize, len(previews)))):
        im = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(im)
        for b in boxes:
            color = COLORS[b.cls % len(COLORS)]
            draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=color, width=3)
            draw.text((b.x1 + 2, b.y1 + 2), classes.names[b.cls], fill=color)
        im.save(prev_dir / f"{i:03d}.jpg", quality=85)

    print(f"\nwrote {out / 'data.yaml'}; skipped {skipped} samples")
    print(f"{'class':<12}{'train':>8}{'val':>8}{'test':>8}")
    for name in classes.names:
        print(f"{name:<12}" + "".join(f"{counts[(s, name)]:>8}" for s in ("train", "val", "test")))
    print(f"\nCheck {prev_dir} before training.")


if __name__ == "__main__":
    main()
