"""PDF access: render pages to images and read the text layer with positions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

MAX_PIXELS = 40_000_000  # cap one rendered page at ~40 MP to bound memory


@dataclass
class Word:
    text: str
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2


@dataclass
class Page:
    index: int            # 0-based
    image: Image.Image    # RGB render
    scale: float          # pixels per PDF point
    text: str             # full text layer ("" for scanned pages)
    words: list[Word]     # word boxes in image pixel coordinates
    tables: list[list[list[str]]]  # text-layer tables: rows of cell strings


def open_pdf(path: str | Path):
    import pymupdf

    return pymupdf.open(str(path))


def load_page(doc, index: int, dpi: int = 150, *, with_tables: bool = False) -> Page:
    import pymupdf

    page = doc[index]
    scale = dpi / 72.0
    w_px, h_px = page.rect.width * scale, page.rect.height * scale
    if w_px * h_px > MAX_PIXELS:
        scale *= (MAX_PIXELS / (w_px * h_px)) ** 0.5
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    words = [Word(w[4], w[0] * scale, w[1] * scale, w[2] * scale, w[3] * scale)
             for w in page.get_text("words")]
    tables: list[list[list[str]]] = []
    if with_tables:
        try:
            for t in page.find_tables().tables:
                rows = _table_rows(page, t)
                if rows:
                    tables.append(rows)
        except Exception:
            pass  # table finder is best-effort
    return Page(index, image, scale, page.get_text("text"), words, tables)


def _table_rows(page, table) -> list[list[str]]:
    """Read each cell's text from its rectangle.

    PyMuPDF's own `table.extract()` rebuilds text character by character and can
    drop spaces and displace hyphens (3'-0" came back as 3'0" plus a stray "-"),
    which would corrupt dimensions. Clipping the page text to the cell box keeps
    the text as written.
    """
    import pymupdf

    rows = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            if cell is None:  # merged or missing cell
                cells.append("")
                continue
            rect = pymupdf.Rect(cell) + (1, 1, -1, -1)  # stay inside the borders
            text = page.get_textbox(rect) if not rect.is_empty else ""
            cells.append(" ".join(text.split()))
        rows.append(cells)
    return rows


def page_count(path: str | Path) -> int:
    with open_pdf(path) as doc:
        return doc.page_count
