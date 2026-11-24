"""Score extracted door-schedule rows against hand-labelled ground truth.

Rows are matched by door mark. We report:

* row detection: precision / recall / F1 over marks (did we find the rows?)
* cell accuracy: per field and overall, over matched rows (did we read them?)
* exact rows:    share of ground-truth rows reproduced with every field right

Dimensions are compared as inches, so 3'-0" equals 36" equals 3' 0".
"""
from __future__ import annotations

import re

from .metrics import prf
from .schedule import FIELDS

_DIM_FIELDS = {"width", "height"}
_QUOTES = str.maketrans({"’": "'", "′": "'", "”": '"', "″": '"', "‘": "'", "“": '"'})


def norm_mark(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def to_inches(value: str) -> float | None:
    """Parse 3'-0", 3' 0", 7'-10 1/2", 36", 36 -> inches. None if not a dimension."""
    s = (value or "").translate(_QUOTES).strip().upper().replace("IN.", '"').replace("FT.", "'")
    if not s:
        return None

    def frac(text: str) -> float | None:
        text = text.strip()
        if not text:
            return 0.0
        m = re.fullmatch(r"(?:(\d+(?:\.\d+)?)\s*)?(?:(\d+)\s*/\s*(\d+))?", text)
        if not m or (m.group(1) is None and m.group(2) is None):
            return None
        whole = float(m.group(1)) if m.group(1) else 0.0
        part = int(m.group(2)) / int(m.group(3)) if m.group(2) and int(m.group(3)) else 0.0
        return whole + part

    m = re.fullmatch(r"(\d+)\s*'\s*-?\s*([\d\s./]*)\s*\"?", s)
    if m:
        inches = frac(m.group(2))
        return None if inches is None else int(m.group(1)) * 12 + inches
    m = re.fullmatch(r"([\d\s./]+)\s*\"?", s)
    if m:
        return frac(m.group(1))
    return None


def norm_cell(field: str, value: str) -> str:
    text = re.sub(r"\s+", " ", (value or "").translate(_QUOTES)).strip().upper()
    if text in {"-", "--", "N/A", "NA", "NONE"}:
        text = ""
    if field in _DIM_FIELDS:
        inches = to_inches(text)
        if inches is not None:
            return f"{inches:.3f}"
    if field == "mark":
        return norm_mark(text)
    return re.sub(r"\s*([/\-.,])\s*", r"\1", text)


def score(pred: list[dict], gold: list[dict], fields: list[str] | None = None) -> dict:
    fields = [f for f in (fields or FIELDS) if f != "mark"]
    gold_by = {norm_mark(r["mark"]): r for r in gold if norm_mark(r.get("mark", ""))}
    pred_by: dict[str, dict] = {}
    duplicates = 0
    for r in pred:
        key = norm_mark(r.get("mark", ""))
        if not key:
            continue
        if key in pred_by:
            duplicates += 1  # a repeated mark counts as a false positive
        else:
            pred_by[key] = r

    matched = sorted(set(gold_by) & set(pred_by))
    rows = prf(tp=len(matched),
               fp=len(set(pred_by) - set(gold_by)) + duplicates,
               fn=len(set(gold_by) - set(pred_by)))

    per_field = {f: {"correct": 0, "total": 0} for f in fields}
    exact = 0
    for key in matched:
        all_ok = True
        for f in fields:
            ok = norm_cell(f, pred_by[key].get(f, "")) == norm_cell(f, gold_by[key].get(f, ""))
            per_field[f]["total"] += 1
            per_field[f]["correct"] += int(ok)
            all_ok &= ok
        exact += int(all_ok)
    total = sum(v["total"] for v in per_field.values())
    correct = sum(v["correct"] for v in per_field.values())
    return {
        "rows": rows,
        "cell_accuracy": correct / total if total else 0.0,
        "per_field_accuracy": {f: (v["correct"] / v["total"] if v["total"] else None)
                               for f, v in per_field.items()},
        "exact_row_rate": exact / len(gold_by) if gold_by else 0.0,
        "gold_rows": len(gold_by),
        "pred_rows": len(pred_by) + duplicates,
    }


def aggregate(scores: list[dict]) -> dict:
    """Micro-average several per-page scores."""
    tp = sum(s["rows"]["tp"] for s in scores)
    fp = sum(s["rows"]["fp"] for s in scores)
    fn = sum(s["rows"]["fn"] for s in scores)
    gold = sum(s["gold_rows"] for s in scores)
    cells = sum(s["cell_accuracy"] * s["rows"]["tp"] for s in scores)
    exact = sum(s["exact_row_rate"] * s["gold_rows"] for s in scores)
    return {"pages": len(scores), "rows": prf(tp, fp, fn),
            "cell_accuracy": cells / tp if tp else 0.0,
            "exact_row_rate": exact / gold if gold else 0.0,
            "gold_rows": gold}
