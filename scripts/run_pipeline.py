"""Run the whole pipeline on one PDF from the command line.

Usage:
    python scripts/run_pipeline.py drawings.pdf --out out/drawings
Models are picked up from BR_DETECTOR_WEIGHTS, BR_CLASSIFIER_WEIGHTS and the
VLM_* environment variables; anything missing falls back to its baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from blueprint_reader.pipeline import Models, analyze_pdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--out", default="out")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--max-pages", type=int, default=60)
    args = ap.parse_args()

    result = analyze_pdf(args.pdf, args.out, Models.from_env(), dpi=args.dpi, max_pages=args.max_pages)
    Path(args.out, "result.json").write_text(json.dumps(result, indent=2))
    labels: dict[str, int] = {}
    for p in result["pages"]:
        labels[p["label"]] = labels.get(p["label"], 0) + 1
    print("methods:", result["methods"])
    print("pages by label:", labels)
    print("schedule rows:", len(result["schedule"]["rows"]))
    if result["crosscheck"]:
        c = result["crosscheck"]
        print(f"doors detected: {c['doors_detected']}  schedule rows: {c['schedule_rows']}")
        print("in schedule, not on plan:", c["in_schedule_not_on_plan"])
    for n in result["notes"]:
        print("note:", n)
    print(f"full result: {Path(args.out, 'result.json')}")


if __name__ == "__main__":
    main()
