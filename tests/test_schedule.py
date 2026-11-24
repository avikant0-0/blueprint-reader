import pytest

from blueprint_reader.evaluate_schedule import aggregate, norm_cell, score, to_inches
from blueprint_reader.schedule import extract_from_tables, map_header, parse_rows


def test_map_header_common_names():
    m = map_header(["DOOR NO.", "WIDTH", "HEIGHT", "TYPE", "MATL", "FIRE RATING", "HDWR SET", "REMARKS"])
    assert m == {0: "mark", 1: "width", 2: "height", 3: "door_type", 4: "door_material",
                 5: "fire_rating", 6: "hardware_set", 7: "remarks"}


def test_map_header_uses_group_row_for_door_vs_frame():
    group = ["", "DOOR", "", "", "FRAME", "", ""]
    cells = ["MARK", "TYPE", "MATL", "WIDTH", "TYPE", "MATL", "HARDWARE"]
    m = map_header(cells, group)
    assert m[1] == "door_type" and m[2] == "door_material"
    assert m[4] == "frame_type" and m[5] == "frame_material"


def test_extract_from_tables_skips_title_row_and_blank_marks():
    table = [["DOOR SCHEDULE", "", "", ""],
             ["MARK", "WIDTH", "HEIGHT", "HARDWARE SET"],
             ["101", "3'-0\"", "7'-0\"", "HW-1"],
             ["", "", "", ""],
             ["102A", "6'-0\"", "7'-0\"", "HW-3"]]
    rows = extract_from_tables([[["A", "B"], ["1", "2"]], table])
    assert [r["mark"] for r in rows] == ["101", "102A"]
    assert rows[1]["width"] == "6'-0\"" and rows[1]["hardware_set"] == "HW-3"


def test_parse_rows_tolerates_code_fences_and_bad_json():
    reply = 'Here you go:\n```json\n{"rows": [{"mark": "101", "width": "3\'-0\\""}, {"mark": ""}]}\n```'
    rows = parse_rows(reply)
    assert len(rows) == 1 and rows[0]["mark"] == "101" and rows[0]["remarks"] == ""
    assert parse_rows("no table found") == []
    assert parse_rows("{not json}") == []


@pytest.mark.parametrize("text,inches", [
    ("3'-0\"", 36), ("3' - 0\"", 36), ("3'0\"", 36), ("36\"", 36), ("36", 36),
    ("7'-10 1/2\"", 94.5), ("3'", 36), ("2'-8\"", 32), ("1 3/4\"", 1.75),
    ("3’-0”", 36),
])
def test_to_inches(text, inches):
    assert to_inches(text) == pytest.approx(inches)


def test_to_inches_rejects_non_dimensions():
    assert to_inches("HM") is None and to_inches("") is None and to_inches("PAIR") is None


def test_norm_cell():
    assert norm_cell("width", "3'-0\"") == norm_cell("width", "36\"")
    assert norm_cell("fire_rating", " 45 min ") == "45 MIN"
    assert norm_cell("remarks", "N/A") == ""
    assert norm_cell("hardware_set", "HW - 1") == norm_cell("hardware_set", "HW-1")
    assert norm_cell("mark", "101-a") == "101A"


def test_score_rows_and_cells():
    gold = [{"mark": "101", "width": "3'-0\"", "height": "7'-0\"", "hardware_set": "HW-1"},
            {"mark": "102", "width": "3'-0\"", "height": "7'-0\"", "hardware_set": "HW-2"},
            {"mark": "103", "width": "6'-0\"", "height": "7'-0\"", "hardware_set": "HW-3"}]
    pred = [{"mark": "101", "width": "36\"", "height": "7'-0\"", "hardware_set": "HW-1"},   # all right
            {"mark": "102", "width": "3'-0\"", "height": "7'-0\"", "hardware_set": "HW-9"},  # one wrong cell
            {"mark": "999", "width": "", "height": "", "hardware_set": ""}]                   # invented row
    s = score(pred, gold, fields=["mark", "width", "height", "hardware_set"])
    assert (s["rows"]["tp"], s["rows"]["fp"], s["rows"]["fn"]) == (2, 1, 1)
    assert s["rows"]["precision"] == pytest.approx(2 / 3)
    assert s["cell_accuracy"] == pytest.approx(5 / 6)
    assert s["per_field_accuracy"]["hardware_set"] == pytest.approx(0.5)
    assert s["exact_row_rate"] == pytest.approx(1 / 3)


def test_duplicate_marks_count_as_false_positives():
    gold = [{"mark": "101", "width": "36"}]
    pred = [{"mark": "101", "width": "36"}, {"mark": "101", "width": "36"}]
    s = score(pred, gold, fields=["mark", "width"])
    assert (s["rows"]["tp"], s["rows"]["fp"]) == (1, 1)


def test_aggregate_micro_averages():
    a = score([{"mark": "1", "width": "36"}], [{"mark": "1", "width": "36"}], fields=["mark", "width"])
    b = score([], [{"mark": "2", "width": "36"}, {"mark": "3", "width": "36"}], fields=["mark", "width"])
    agg = aggregate([a, b])
    assert agg["rows"]["recall"] == pytest.approx(1 / 3)
    assert agg["cell_accuracy"] == pytest.approx(1.0)
    assert agg["exact_row_rate"] == pytest.approx(1 / 3)
