"""Fine-tune a YOLO detector on the converted CubiCasa5K dataset and score it
on the held-out test split.

Usage:
    python scripts/train_detector.py --data data/yolo/data.yaml --epochs 60

Prints where the weights were saved and writes results/detector_test.json
with the numbers that go in the README and on the resume.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11s.pt", help="pretrained checkpoint to start from")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1024,
                    help="floor-plan symbols are small; 1024 keeps them visible")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="cubicasa")
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/detector_test.json")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        project=str(Path("runs/detector").resolve()), name=args.name, device=args.device,
        workers=args.workers, exist_ok=True,
        # Plans are line drawings: colour jitter is pointless, and mosaic/flip
        # are fine because symbols appear in every orientation.
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, degrees=0.0, fliplr=0.5, flipud=0.5,
        patience=15,
    )

    # Ask the trainer where it saved: the run folder layout differs between
    # Ultralytics versions, so do not rebuild the path by hand.
    best = Path(model.trainer.best)
    metrics = YOLO(str(best)).val(data=args.data, split="test", imgsz=args.imgsz,
                                  batch=args.batch, device=args.device,
                                  workers=args.workers,
                                  project=str(best.parent.parent.parent),
                                  name=f"{args.name}_test", exist_ok=True)
    names = metrics.names
    per_class = {names[int(c)]: {"mAP50": float(metrics.box.ap50[i]),
                                 "mAP50-95": float(metrics.box.ap[i])}
                 for i, c in enumerate(metrics.box.ap_class_index)}
    summary = {
        "weights": str(best),
        "model": args.model,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "split": "test",
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class": per_class,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
