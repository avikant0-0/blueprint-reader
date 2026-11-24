"""FastAPI service: upload a drawing-set PDF, get the analysis back.

Run:
    BR_DETECTOR_WEIGHTS=models/detector.pt \
    BR_CLASSIFIER_WEIGHTS=models/page_classifier.pt \
    uvicorn api.main:app --reload

Then open http://localhost:8000
"""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from blueprint_reader.pipeline import Models, analyze_pdf

JOBS = Path(os.environ.get("BR_JOBS_DIR", "jobs")).resolve()
MAX_BYTES = int(os.environ.get("BR_MAX_UPLOAD_MB", "80")) * 1024 * 1024
MAX_PAGES = int(os.environ.get("BR_MAX_PAGES", "60"))
WEB = Path(__file__).resolve().parent.parent / "web"

JOBS.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="Blueprint Reader")
_models: Models | None = None


def get_models() -> Models:
    global _models
    if _models is None:
        _models = Models.from_env()
    return _models


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "methods": get_models().describe()}


@app.post("/api/analyze")
def analyze(file: UploadFile = File(...)) -> dict:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Upload a PDF file.")
    job = uuid.uuid4().hex
    job_dir = JOBS / job
    job_dir.mkdir(parents=True)
    pdf_path = job_dir / "input.pdf"
    size = 0
    with pdf_path.open("wb") as fh:
        while chunk := file.file.read(1 << 20):
            size += len(chunk)
            if size > MAX_BYTES:
                fh.close()
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(413, "File too large.")
            fh.write(chunk)
    with pdf_path.open("rb") as fh:
        if fh.read(5) != b"%PDF-":
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(400, "That file is not a PDF.")
    try:
        result = analyze_pdf(pdf_path, job_dir, get_models(), max_pages=MAX_PAGES)
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(422, f"Could not analyse this PDF: {exc}") from exc
    result["job"] = job
    result["file"] = file.filename
    for page in result["pages"]:
        page["image_url"] = f"/files/{job}/{page['image']}"
    return result


app.mount("/files", StaticFiles(directory=str(JOBS)), name="files")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")
