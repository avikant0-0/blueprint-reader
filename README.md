# Blueprint Reader

Reads a construction drawing-set PDF and answers the question an estimator asks first: **which doors are on the plans, what does the door schedule say, and do the two agree?**

```
PDF ─► classify each page ─► floor plans ─► detect doors, windows, fixtures ─┐
                          └► schedules  ─► extract door-schedule rows ───────┴► cross-check ─► mismatches
```

| Stage | What it does | Method | Baseline it is compared against |
|---|---|---|---|
| Page classification | floor plan / schedule / specification / other | Fine-tuned ResNet-18 on rendered pages | Keyword rules on the PDF text layer |
| Symbol detection | doors, windows, toilets, sinks, bathtubs, showers, closets, appliances, fireplaces | YOLO fine-tuned on CubiCasa5K, tiled inference for large sheets | Smaller model / lower resolution (you run both) |
| Schedule extraction | door-schedule table to structured rows | Vision-language model, JSON output | PDF text-layer table parser |
| Cross-check | plan doors vs schedule marks | Tag matching near detected doors | Count comparison only |
| Demo | upload a PDF, see everything | FastAPI + one HTML page | |

## Status: read this first

**The code is written and tested. The models are not trained and there are no real results yet.**

Verified in development:
- 39 unit and integration tests pass (`pytest`), including the full pipeline and the API on a synthetic drawing set.
- The CubiCasa5K converter was written against the official loader code and tested on synthetic SVGs with the same structure. **It has not been run on the real dataset.** That is why the notebook makes you inspect the preview images before training.
- Detector training, test-split scoring, tiled inference and classifier training were each run end to end on tiny synthetic data to confirm they work mechanically. Those runs say nothing about accuracy.
- The vision-language extractor has not been called against a live API.

Every number in the results table below is blank until you produce it. Do not put a number on your resume that is not in `results/`.

## Results (fill in from your own runs)

| Stage | Metric | Baseline | Model | Test set |
|---|---|---|---|---|
| Symbol detection (CubiCasa5K) | mAP@50 / mAP@50-95 | _yolo11n @ 640:_ | _yolo11s @ 1024:_ | CubiCasa5K official test split |
| Door class only | AP@50 | | | same |
| Symbol detection (real sheets) | mAP@50 before / after fine-tuning | _CubiCasa-only:_ | _+ real tiles:_ | your labelled tiles, held-out PDFs |
| Page classification | accuracy / macro-F1 | _keywords:_ | _ResNet-18:_ | pages of held-out PDFs |
| Schedule extraction | row F1 / cell accuracy / exact rows | _text table:_ | _VLM:_ | your hand-labelled schedule sheets |

## One-week plan

| Day | Goal | Commands |
|---|---|---|
| 1 | Push the repo to GitHub. Run the notebook: download CubiCasa5K, convert, **inspect previews**, 2-epoch sanity run. | `notebooks/train_detector.ipynb` |
| 2 | Full detector training in two settings; record both. Read the confusion matrix and note which classes fail. | `scripts/train_detector.py` |
| 3 | Collect 8-12 public drawing-set PDFs (see below). Build the page dataset, correct the labels by hand, train the classifier, compare with the keyword baseline. | `scripts/build_page_dataset.py`, `scripts/train_page_classifier.py` |
| 4 | Cut 20-30 schedule sheets into one-page PDFs, hand-label them, score the text-table baseline and a VLM. | `scripts/eval_schedule.py` |
| 5 | Run the whole pipeline on held-out PDFs. The CubiCasa-trained detector will struggle on real sheets; measure how badly by labelling ~150 real tiles. | `scripts/run_pipeline.py`, `scripts/tile_pages.py` |
| 6 | Fine-tune on the real tiles and re-measure. Write up before/after and what still fails. | `scripts/train_detector.py --model <best.pt>` |
| 7 | Run the demo, record a short screen capture, fill in the results table, write the resume bullets. | `uvicorn api.main:app` |

If you only have three days: do days 1, 2 and 4, plus the demo.

## Setup

```bash
git clone <your repo> && cd blueprint-reader
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train,vlm,api,dev]"
pytest
```

### 1. Detector

Use `notebooks/train_detector.ipynb` on Colab or Kaggle. Locally:

```bash
python scripts/prepare_cubicasa.py --src data/cubicasa5k --out data/yolo --visualize 30
# look at data/yolo/preview/*.jpg before going further
python scripts/train_detector.py --data data/yolo/data.yaml --model yolo11s.pt --epochs 60 --imgsz 1024
```

### 2. Page classifier

```bash
python scripts/build_page_dataset.py --pdfs data/pdfs --out data/pages
# move wrongly filed pages into the right folder by hand
python scripts/train_page_classifier.py --data data/pages --test-pdfs setX,setY
```

### 3. Schedule extraction

```bash
python scripts/eval_schedule.py --data data/schedules --make-template   # draft gold files
# correct every data/schedules/*.gold.json against the sheet by hand
export VLM_API_KEY=...  VLM_MODEL=...  VLM_BASE_URL=...   # any OpenAI-compatible vision endpoint
python scripts/eval_schedule.py --data data/schedules
```

The draft gold files come from the text-table baseline, so an uncorrected gold file flatters the baseline. Check each one.

### 4. Pipeline and demo

```bash
export BR_DETECTOR_WEIGHTS=models/detector.pt BR_CLASSIFIER_WEIGHTS=models/page_classifier.pt
python scripts/run_pipeline.py data/pdfs/some_set.pdf --out out/some_set
uvicorn api.main:app --reload      # then open http://localhost:8000
```

Anything not configured falls back to its baseline, and the result says which method produced each part.

## Getting real drawing sets

Public agencies (cities, school districts, state procurement portals) publish full bid sets with floor plans, door schedules and specifications. Download 8-12 from different projects and architects so the test set is varied. Keep them in `data/` (git-ignored), do not redistribute them, and check each site's terms. Hold out whole PDFs for testing.

## Design decisions worth being able to explain

- **Tiled inference.** A 36x24 inch sheet at 150 dpi is 5400x3600 px and a door drawn at 1/8 inch scale is about 55 px wide. Resizing the sheet to the detector's input would erase it, so pages are cut into overlapping 1024 px tiles and the duplicate boxes in the overlaps are merged with class-wise NMS (`detect.py`).
- **Door boxes include the swing arc.** CubiCasa5K annotates the wall opening; the converter takes the union of everything drawn in the door group, because the arc is what makes a door recognisable (`cubicasa.py`).
- **Split by PDF, not by page.** Pages from one drawing set share a title block and style. A per-page split leaks and inflates accuracy.
- **Every model has a baseline.** A number means little without something to compare it with, and sometimes the baseline wins (text-layer tables on clean vector PDFs).
- **Dimensions are compared as inches.** `3'-0"`, `36"` and `3' 0"` are the same door (`evaluate_schedule.py`).
- **Cell text is read from cell rectangles.** PyMuPDF's built-in table text dropped spaces and displaced hyphens in testing, which corrupts dimensions (`pdf.py`).
- **Known limits.** The cross-check needs a text layer for door tags (no OCR yet); room numbers can look like door marks; the detector is trained on residential plans and must be adapted to commercial sheets.

## Resume bullets (templates: replace every blank with a measured number)

- Built an end-to-end pipeline that reads construction drawing-set PDFs: page classification, door and fixture detection, door-schedule extraction, and a plan-vs-schedule cross-check, served through a FastAPI upload demo.
- Fine-tuned YOLO on CubiCasa5K (5,000 floor plans) with tiled inference for full-size sheets, reaching ___ mAP@50 (___ on doors); adapting to real construction sheets with ___ hand-labelled tiles raised mAP@50 from ___ to ___.
- Extracted door schedules with a vision-language model at ___ row F1 and ___% cell accuracy on ___ hand-labelled sheets, against ___ for a text-layer baseline; page classifier scored ___ macro-F1 on held-out drawing sets against ___ for keyword rules.

## Licences

CubiCasa5K is CC BY-NC-SA 4.0 (non-commercial; cite the dataset). Ultralytics YOLO is AGPL-3.0. Both are fine for a portfolio project; say so if asked.

## Layout

```
src/blueprint_reader/   cubicasa.py detect.py classify.py schedule.py evaluate_schedule.py crosscheck.py pipeline.py pdf.py metrics.py
scripts/                prepare_cubicasa.py train_detector.py build_page_dataset.py train_page_classifier.py eval_schedule.py tile_pages.py run_pipeline.py
api/main.py  web/index.html  notebooks/train_detector.ipynb  configs/classes.yaml  tests/
```
