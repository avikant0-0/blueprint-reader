"""Door-schedule extraction: turn the schedule table on a sheet into rows.

Two extractors produce the same row format so they can be scored side by side:

* `extract_from_tables` - baseline. Uses the PDF's own text-layer table (via
                          PyMuPDF) and maps header names onto our fields.
                          Fails on scans and on tables drawn as loose lines.
* `VLMExtractor`        - sends the rendered page to a vision-language model
                          through any OpenAI-compatible endpoint and asks for
                          JSON. Works on scans; can misread or invent cells,
                          which is exactly what evaluate_schedule measures.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re

FIELDS = ["mark", "width", "height", "door_type", "door_material",
          "frame_type", "frame_material", "fire_rating", "hardware_set", "remarks"]

# Header text (lower-cased, punctuation stripped) -> field. First match wins,
# so more specific patterns come first.
_HEADER_PATTERNS: list[tuple[str, str]] = [
    (r"^(door )?(mark|no|number|num|tag|id)$|^opening( no| number)?$|^door$", "mark"),
    (r"hardware|hdwr|hdw|hw( set| group)?", "hardware_set"),
    (r"fire|rating|label", "fire_rating"),
    (r"remark|note|comment", "remarks"),
    (r"width|^w$|^wd$", "width"),
    (r"height|^h$|^ht$", "height"),
    (r"frame.*(mat|matl|material)", "frame_material"),
    (r"frame.*(type|elev)|^frame$", "frame_type"),
    (r"(door )?(mat|matl|material)", "door_material"),
    (r"(door )?(type|elev)", "door_type"),
]


def _clean_header(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (text or "").lower())).strip()


def map_header(cells: list[str], group: list[str] | None = None) -> dict[int, str]:
    """Map column index -> field name for a header row.

    `group` is an optional row above the header with spanning titles such as
    DOOR / FRAME, used to tell "frame type" from "door type".
    """
    mapping: dict[int, str] = {}
    current_group = ""
    for i, cell in enumerate(cells):
        if group is not None and i < len(group) and _clean_header(group[i]):
            current_group = _clean_header(group[i])
        label = _clean_header(cell)
        if not label:
            continue
        if current_group and current_group not in label and current_group in ("door", "frame"):
            label = f"{current_group} {label}"
        for pattern, field in _HEADER_PATTERNS:
            if re.search(pattern, label) and field not in mapping.values():
                mapping[i] = field
                break
    return mapping


def extract_from_tables(tables: list[list[list[str]]]) -> list[dict]:
    """Baseline: pick the text-layer table that looks most like a door schedule."""
    best_rows: list[dict] = []
    for table in tables:
        for h in range(min(4, len(table))):  # header may sit below a title/group row
            group = table[h - 1] if h > 0 else None
            mapping = map_header(table[h], group)
            if "mark" not in mapping.values() or len(mapping) < 3:
                continue
            rows = []
            for raw in table[h + 1:]:
                row = {f: "" for f in FIELDS}
                for idx, field in mapping.items():
                    if idx < len(raw):
                        row[field] = re.sub(r"\s+", " ", raw[idx] or "").strip()
                if row["mark"]:
                    rows.append(row)
            if len(rows) > len(best_rows):
                best_rows = rows
            break
    return best_rows


PROMPT = f"""You are reading one sheet from a construction drawing set.
Find the DOOR SCHEDULE table on this sheet and transcribe every row.

Return ONLY a JSON object: {{"rows": [ ... ]}}. Each row has exactly these keys:
{", ".join(FIELDS)}

Rules:
- "mark" is the door number/tag (for example 101, 101A, D-12).
- Copy cell text exactly as printed, including feet-inch marks. Do not convert units.
- If a cell is blank, or the sheet has no such column, use "".
- Do not invent rows or values. If you cannot read a cell, use "".
- If the sheet has no door schedule, return {{"rows": []}}."""


def parse_rows(text: str) -> list[dict]:
    """Parse the model's reply into normalised rows; tolerant of code fences."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    rows = []
    for raw in data.get("rows", []) if isinstance(data, dict) else []:
        if not isinstance(raw, dict):
            continue
        row = {f: str(raw.get(f, "") or "").strip() for f in FIELDS}
        if row["mark"]:
            rows.append(row)
    return rows


class VLMExtractor:
    """Vision-language extraction through an OpenAI-compatible chat endpoint.

    Configure with environment variables (or constructor arguments):
        VLM_API_KEY, VLM_BASE_URL, VLM_MODEL
    Any provider with an OpenAI-compatible API and a vision model works.
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, max_side: int = 2200):
        from openai import OpenAI

        self.model = model or os.environ["VLM_MODEL"]
        self.client = OpenAI(api_key=api_key or os.environ["VLM_API_KEY"],
                             base_url=base_url or os.environ.get("VLM_BASE_URL"))
        self.max_side = max_side

    @staticmethod
    def configured() -> bool:
        return bool(os.environ.get("VLM_API_KEY") and os.environ.get("VLM_MODEL"))

    def _encode(self, image) -> str:
        im = image.convert("RGB")
        if max(im.size) > self.max_side:
            r = self.max_side / max(im.size)
            im = im.resize((int(im.width * r), int(im.height * r)))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()

    def extract(self, image) -> list[dict]:
        reply = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{self._encode(image)}"}},
            ]}],
        )
        return parse_rows(reply.choices[0].message.content or "")
