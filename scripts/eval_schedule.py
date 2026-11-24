"""Evaluate door-schedule extraction on hand-labelled pages.

Layout expected:
    data/schedules/<name>.pdf        one-page PDF of a schedule sheet
    data/schedules/<name>.gold.json  {"rows": [{"mark": "101", "width": "3'-0\"", ...}]}

Usage:
    python scripts/eval_schedule.py --data data/schedules            # baseline only
    VLM_API_KEY=... VLM_MODEL=... python scripts/eval_schedule.py --data data/schedules

Writes results/schedule_eval.json and saves each extractor's raw rows next to
the PDFs (<name>.pred.<extractor>.json) so mistakes can be inspected.

`--make-template` writes a draft <name>.gold.json from the baseline for every
PDF that has none. CORRECT IT BY HAND against the sheet before trusting it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from blueprint_reader.evaluate_schedule import aggregate, score
from blueprint_reader.pdf import load_page, open_pdf
from blueprint_reader.schedule import VLMExtractor, extract_from_tables


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--page", type=int, default=0, help="0-based page inside each PDF")
    ap.add_argument("--make-template", action="store_true")
    args = ap.parse_args()

    data = Path(args.data)
    vlm = VLMExtractor() if VLMExtractor.configured() else None
    if vlm is None:
        print("VLM not configured (set VLM_API_KEY and VLM_MODEL); scoring the baseline only.")
    per_page: dict[str, dict] = {}
    scores: dict[str, list[dict]] = {"text_table_baseline": [], "vlm": []}

    for pdf in sorted(data.glob("*.pdf")):
        gold_path = pdf.with_suffix(".gold.json")
        with open_pdf(pdf) as doc:
            page = load_page(doc, args.page, dpi=args.dpi, with_tables=True)
        preds = {"text_table_baseline": extract_from_tables(page.tables)}
        if vlm is not None:
            preds["vlm"] = vlm.extract(page.image)
        for name, rows in preds.items():
            pdf.with_suffix(f".pred.{name}.json").write_text(json.dumps({"rows": rows}, indent=2))
        if not gold_path.exists():
            if args.make_template:
                gold_path.write_text(json.dumps({"rows": preds["text_table_baseline"]}, indent=2))
                print(f"{pdf.name}: wrote draft {gold_path.name} - correct it by hand")
            else:
                print(f"{pdf.name}: no gold file, skipped")
            continue
        if args.make_template:
            continue
        gold = json.loads(gold_path.read_text())["rows"]
        per_page[pdf.name] = {}
        for name, rows in preds.items():
            s = score(rows, gold)
            per_page[pdf.name][name] = s
            scores[name].append(s)

    summary = {name: aggregate(s) for name, s in scores.items() if s}
    if summary:
        Path("results").mkdir(exist_ok=True)
        Path("results/schedule_eval.json").write_text(
            json.dumps({"summary": summary, "per_page": per_page}, indent=2))
        for name, s in summary.items():
            print(f"{name:<22} pages {s['pages']:>3}  row F1 {s['rows']['f1']:.3f}  "
                  f"cell acc {s['cell_accuracy']:.3f}  exact rows {s['exact_row_rate']:.3f}")


if __name__ == "__main__":
    main()
