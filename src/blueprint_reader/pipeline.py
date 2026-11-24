"""End-to-end run over one drawing-set PDF.

    classify every page -> detect symbols on floor plans
                        -> extract the door schedule from schedule sheets
                        -> cross-check plans against the schedule

Every model is optional so the pipeline is usable while you are still training:
no classifier weights -> keyword baseline; no detector weights -> detection is
skipped; no VLM key -> text-layer table baseline. The result records which
method produced each part.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .classify import classify_text
from .crosscheck import crosscheck
from .pdf import load_page, open_pdf
from .schedule import VLMExtractor, extract_from_tables


@dataclass
class Models:
    detector: object | None = None      # blueprint_reader.detect.Detector
    classifier: object | None = None    # blueprint_reader.classify.PageClassifier
    vlm: object | None = None           # blueprint_reader.schedule.VLMExtractor

    @classmethod
    def from_env(cls) -> "Models":
        """Load whatever is configured; missing pieces stay None."""
        m = cls()
        det = os.environ.get("BR_DETECTOR_WEIGHTS")
        if det and Path(det).exists():
            from .detect import Detector
            m.detector = Detector(det)
        clf = os.environ.get("BR_CLASSIFIER_WEIGHTS")
        if clf and Path(clf).exists():
            from .classify import PageClassifier
            m.classifier = PageClassifier(clf)
        if VLMExtractor.configured():
            m.vlm = VLMExtractor()
        return m

    def describe(self) -> dict:
        return {"page_classifier": "resnet18" if self.classifier else "keyword_baseline",
                "detector": "yolo" if self.detector else None,
                "schedule_extractor": "vlm" if self.vlm else "text_table_baseline"}


def analyze_pdf(pdf_path: str | Path, out_dir: str | Path, models: Models | None = None,
                *, dpi: int = 150, max_pages: int = 60) -> dict:
    models = models or Models()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pages_out: list[dict] = []
    plan_pages: list[dict] = []
    schedule_rows: list[dict] = []
    schedule_pages: list[int] = []
    seen_marks: set[str] = set()

    with open_pdf(pdf_path) as doc:
        total = doc.page_count
        for i in range(min(total, max_pages)):
            page = load_page(doc, i, dpi=dpi, with_tables=True)
            if models.classifier is not None:
                label, conf = models.classifier.predict(page.image)
            else:
                label, _ = classify_text(page.text)
                conf = None

            detections = []
            if label == "floor_plan" and models.detector is not None:
                detections = models.detector.predict(page.image)
                plan_pages.append({"index": i, "words": page.words,
                                   "doors": [d for d in detections if d.cls == "door"]})
            elif label == "floor_plan":
                plan_pages.append({"index": i, "words": page.words, "doors": []})

            if label == "schedule":
                rows = (models.vlm.extract(page.image) if models.vlm is not None
                        else extract_from_tables(page.tables))
                if rows:
                    schedule_pages.append(i)
                for r in rows:  # a schedule can continue across sheets
                    if r["mark"] not in seen_marks:
                        seen_marks.add(r["mark"])
                        schedule_rows.append(r)

            name = f"page_{i:04d}.jpg"
            page.image.save(out / name, quality=80)
            counts: dict[str, int] = {}
            for d in detections:
                counts[d.cls] = counts.get(d.cls, 0) + 1
            pages_out.append({
                "index": i, "label": label, "confidence": conf,
                "image": name, "width": page.image.width, "height": page.image.height,
                "has_text_layer": bool(page.words),
                "detections": [d.to_dict() for d in detections],
                "counts": counts,
            })

    check = None
    if schedule_rows and models.detector is not None:
        check = crosscheck(schedule_rows, plan_pages)

    return {
        "file": Path(pdf_path).name,
        "pages_total": total,
        "pages_processed": len(pages_out),
        "methods": models.describe(),
        "pages": pages_out,
        "schedule": {"rows": schedule_rows, "pages": schedule_pages},
        "crosscheck": check,
        "notes": _notes(models, schedule_rows, plan_pages),
    }


def _notes(models: Models, rows: list[dict], plans: list[dict]) -> list[str]:
    notes = []
    if models.detector is None:
        notes.append("No detector weights loaded: symbol detection and the cross-check were skipped.")
    if not plans:
        notes.append("No page was classified as a floor plan.")
    if not rows:
        notes.append("No door schedule rows were extracted.")
    return notes
