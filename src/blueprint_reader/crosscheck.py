"""Cross-check doors found on the plans against the door schedule.

Estimators care about one failure above all: a door that is on the drawings but
missing from the schedule (or the reverse), because a missed opening is a
missed cost. This module compares the two sources and lists the mismatches.

Inputs
  schedule rows  - from schedule.py (each has a "mark" such as 101A)
  door boxes     - from detect.py, per floor-plan page
  words          - text-layer words with positions, per floor-plan page

Method
  1. A door tag on a plan is a word whose text equals a schedule mark.
  2. A detected door is "tagged" if a mark-like word lies within `radius` door
     sizes of its centre.
  3. Report: schedule marks never seen on any plan; mark-like words next to a
     detected door that are not in the schedule; detected doors with no tag.

Scanned sheets have no text layer, so only the count comparison is available
for them (adding OCR for tags is the natural next step).
"""
from __future__ import annotations

import math
import re

from .evaluate_schedule import norm_mark

# Door marks: 101, 101A, 1-12, D12, A101.2 ... short alphanumerics with a digit.
MARK_LIKE = re.compile(r"^(?=.*\d)[A-Z]{0,2}-?\d{1,4}(?:[.\-]\d{1,2})?[A-Z]?$")


def is_mark_like(text: str) -> bool:
    return bool(MARK_LIKE.match((text or "").strip().upper()))


def crosscheck(schedule_rows: list[dict], pages: list[dict], radius: float = 1.6) -> dict:
    """pages: [{"index": int, "doors": [Detection], "words": [Word]}] for floor plans."""
    marks = {norm_mark(r["mark"]): r["mark"] for r in schedule_rows if norm_mark(r.get("mark", ""))}
    seen_on_plan: set[str] = set()
    near_door: set[str] = set()
    unknown_tags: list[dict] = []
    untagged: list[dict] = []
    total_doors = 0
    any_text = False

    for page in pages:
        words = [w for w in page.get("words", []) if is_mark_like(w.text)]
        any_text |= bool(page.get("words"))
        for w in words:
            key = norm_mark(w.text)
            if key in marks:
                seen_on_plan.add(key)
        for d in page.get("doors", []):
            total_doors += 1
            cx, cy = d.center
            reach = radius * max(d.x2 - d.x1, d.y2 - d.y1)
            close = [(math.hypot(w.center[0] - cx, w.center[1] - cy), w) for w in words]
            close = sorted((c for c in close if c[0] <= reach), key=lambda c: c[0])
            if not close:
                untagged.append({"page": page["index"], "box": [d.x1, d.y1, d.x2, d.y2]})
                continue
            known = [w for _, w in close if norm_mark(w.text) in marks]
            if known:
                near_door.add(norm_mark(known[0].text))
            else:
                w = close[0][1]
                unknown_tags.append({"page": page["index"], "tag": w.text,
                                     "box": [d.x1, d.y1, d.x2, d.y2]})

    in_schedule_not_on_plan = sorted(marks[k] for k in marks if k not in seen_on_plan)
    on_plan_no_door_detected = sorted(marks[k] for k in seen_on_plan - near_door)
    # The same unknown tag can sit near several boxes; report each tag once per page.
    dedup = {(u["page"], norm_mark(u["tag"])): u for u in unknown_tags}

    return {
        "schedule_rows": len(marks),
        "doors_detected": total_doors,
        "count_difference": total_doors - len(marks),
        "text_layer_available": any_text,
        "in_schedule_not_on_plan": in_schedule_not_on_plan,
        "tag_on_plan_but_no_door_detected": on_plan_no_door_detected,
        "tags_near_doors_not_in_schedule": list(dedup.values()),
        "detected_doors_without_tag": untagged,
    }
