from conftest import PLAN_TAGS, SCHEDULE

from blueprint_reader.classify import classify_text
from blueprint_reader.crosscheck import crosscheck, is_mark_like
from blueprint_reader.detect import Detection
from blueprint_reader.pdf import Word, load_page, open_pdf
from blueprint_reader.pipeline import Models, analyze_pdf


class FakeDetector:
    """Stands in for trained weights: one door box right above every tag, plus
    one door with no tag anywhere near it."""

    def predict(self, image):
        s = 150 / 72
        dets = [Detection("door", 0.9, x * s, (y - 30) * s, (x + 30) * s, y * s)
                for x, y in PLAN_TAGS.values()]
        dets.append(Detection("door", 0.8, 300 * s, 300 * s, 330 * s, 330 * s))
        dets.append(Detection("window", 0.7, 90 * s, 110 * s, 140 * s, 125 * s))
        return dets


def test_keyword_baseline_labels():
    assert classify_text("A-101 FIRST FLOOR PLAN  SCALE: 1/8\" = 1'-0\"")[0] == "floor_plan"
    assert classify_text("DOOR SCHEDULE MARK WIDTH HEIGHT HARDWARE SET")[0] == "schedule"
    assert classify_text("SECTION 08 11 13 PART 1 - GENERAL submittals")[0] == "specification"
    assert classify_text("RIVERSIDE APARTMENTS")[0] == "other"
    assert classify_text("")[0] == "other"


def test_mark_like():
    for ok in ["101", "101A", "D12", "1-12", "A101.2", "103a"]:
        assert is_mark_like(ok), ok
    for bad in ["PLAN", "A", "FIRST", "3'-0\"", "12345678", ""]:
        assert not is_mark_like(bad), bad


def test_page_loading_reads_words_and_tables(drawing_set):
    with open_pdf(drawing_set) as doc:
        plan = load_page(doc, 0, dpi=150)
        sched = load_page(doc, 1, dpi=150, with_tables=True)
    assert plan.image.size == (1650, 1275)
    tag = next(w for w in plan.words if w.text == "103A")
    assert abs(tag.x1 - (150 + 4) * 150 / 72) < 3  # word boxes are in pixel coordinates
    assert any(t and t[0][0] == "MARK" for t in sched.tables)


def test_pipeline_without_models_uses_baselines(drawing_set, tmp_path):
    res = analyze_pdf(drawing_set, tmp_path)
    assert [p["label"] for p in res["pages"]] == ["floor_plan", "schedule", "specification", "other"]
    assert [r["mark"] for r in res["schedule"]["rows"]] == [row[0] for row in SCHEDULE[1:]]
    assert res["schedule"]["rows"][2]["fire_rating"] == "45 MIN"
    assert res["crosscheck"] is None and res["methods"]["detector"] is None
    assert any("No detector" in n for n in res["notes"])
    assert (tmp_path / "page_0000.jpg").exists()


def test_pipeline_crosscheck_flags_mismatches(drawing_set, tmp_path):
    res = analyze_pdf(drawing_set, tmp_path, Models(detector=FakeDetector()))
    plan = res["pages"][0]
    assert plan["counts"] == {"door": 5, "window": 1}
    c = res["crosscheck"]
    assert c["doors_detected"] == 5 and c["schedule_rows"] == 4
    assert c["in_schedule_not_on_plan"] == ["104"]
    assert [u["tag"] for u in c["tags_near_doors_not_in_schedule"]] == ["199"]
    assert len(c["detected_doors_without_tag"]) == 1
    assert c["tag_on_plan_but_no_door_detected"] == []


def test_crosscheck_reports_tag_with_no_detected_door():
    rows = [{"mark": "101"}, {"mark": "102"}]
    words = [Word("101", 100, 100, 120, 110), Word("102", 900, 900, 920, 910)]
    doors = [Detection("door", 0.9, 90, 60, 130, 100)]
    c = crosscheck(rows, [{"index": 0, "words": words, "doors": doors}])
    assert c["tag_on_plan_but_no_door_detected"] == ["102"]
    assert c["in_schedule_not_on_plan"] == []
