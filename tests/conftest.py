"""Builds a small synthetic drawing set so the pipeline can be tested without
real blueprints or trained weights."""
import pymupdf
import pytest

SCHEDULE = [["MARK", "WIDTH", "HEIGHT", "TYPE", "FIRE RATING", "HARDWARE SET"],
            ["101", "3'-0\"", "7'-0\"", "A", "20 MIN", "HW-1"],
            ["102", "3'-0\"", "7'-0\"", "A", "", "HW-2"],
            ["103A", "6'-0\"", "7'-0\"", "B", "45 MIN", "HW-3"],
            ["104", "3'-0\"", "7'-0\"", "A", "", "HW-1"]]

# Door tag positions on the plan page, in PDF points. 104 is in the schedule but
# deliberately missing from the plan; 199 is on the plan but not in the schedule.
PLAN_TAGS = {"101": (150, 200), "102": (400, 200), "103A": (150, 420), "199": (400, 420)}


@pytest.fixture(scope="session")
def drawing_set(tmp_path_factory):
    path = tmp_path_factory.mktemp("pdf") / "set.pdf"
    doc = pymupdf.open()

    plan = doc.new_page(width=792, height=612)
    plan.insert_text((560, 580), "A-101  FIRST FLOOR PLAN", fontsize=14)
    plan.draw_rect(pymupdf.Rect(80, 120, 520, 500), width=3)
    for tag, (x, y) in PLAN_TAGS.items():
        plan.draw_line((x, y), (x + 30, y), width=1)
        plan.insert_text((x + 4, y + 22), tag, fontsize=8)

    sched = doc.new_page(width=792, height=612)
    sched.insert_text((60, 60), "DOOR SCHEDULE", fontsize=16)
    x0, y0, cw, rh = 60, 90, 105, 24
    for r, row in enumerate(SCHEDULE):
        for c, cell in enumerate(row):
            sched.draw_rect(pymupdf.Rect(x0 + c * cw, y0 + r * rh, x0 + (c + 1) * cw, y0 + (r + 1) * rh), width=0.7)
            if cell:
                sched.insert_text((x0 + c * cw + 5, y0 + r * rh + 16), cell, fontsize=9)

    spec = doc.new_page(width=612, height=792)
    spec.insert_text((60, 70), "SECTION 08 11 13 - HOLLOW METAL DOORS AND FRAMES", fontsize=12)
    spec.insert_text((60, 100), "PART 1 - GENERAL", fontsize=11)
    spec.insert_text((60, 125), "1.1 SUBMITTALS: Product data for each type of door and frame.", fontsize=10)
    spec.insert_text((60, 160), "PART 2 - PRODUCTS", fontsize=11)

    cover = doc.new_page(width=792, height=612)
    cover.insert_text((250, 300), "RIVERSIDE APARTMENTS", fontsize=24)

    doc.save(str(path))
    doc.close()
    return path
