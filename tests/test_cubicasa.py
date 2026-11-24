from pathlib import Path

import numpy as np

from blueprint_reader.cubicasa import ClassMap, parse_svg, parse_transform, read_split

CLASSES = ClassMap.load(Path(__file__).parent.parent / "configs" / "classes.yaml")

# Mirrors the structure the official CubiCasa5K loader reads: a Door group whose
# first polygon is the wall opening, plus a nested panel with the swing arc; a
# Window group; and fixtures positioned with matrix transforms, one of them
# inside a FixedFurnitureSet.
SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600">
 <g id="Model">
  <g id="Wall" class="Wall External"><polygon points="0,0 800,0 800,10 0,10 "/></g>
  <g id="Door" class="Door Swing Beside">
    <polygon points="100,200 140,200 140,210 100,210 "/>
    <g id="Panel" class="Panel">
      <polygon points="100,200 104,200 104,160 100,160 "/>
      <path d="M 140,200 Q 140,160 100,160"/>
    </g>
  </g>
  <g id="Window" class="Window Regular">
    <polygon points="300,0 380,0 380,10 300,10 "/>
  </g>
  <g class="FixedFurnitureSet" transform="matrix(1,0,0,1,400,300)">
    <g class="FixedFurniture Toilet" transform="matrix(1,0,0,1,10,20)">
      <g class="BoundaryPolygon"><polygon points="0,0 30,0 30,50 0,50 "/></g>
    </g>
  </g>
  <g class="FixedFurniture Sink" transform="matrix(0,1,-1,0,600,100)">
    <g class="BoundaryPolygon"><polygon points="0,0 40,0 40,20 0,20 "/></g>
  </g>
  <g class="FixedFurniture Misc" transform="matrix(1,0,0,1,0,0)">
    <g class="BoundaryPolygon"><polygon points="0,0 40,0 40,20 0,20 "/></g>
  </g>
 </g>
</svg>"""


def boxes_by_name(tmp_path, **kw):
    svg = tmp_path / "model.svg"
    svg.write_text(SVG)
    out = {}
    for b in parse_svg(svg, 800, 600, CLASSES, **kw):
        out.setdefault(CLASSES.names[b.cls], []).append((b.x1, b.y1, b.x2, b.y2))
    return out


def test_door_full_box_includes_swing(tmp_path):
    got = boxes_by_name(tmp_path)
    assert got["door"] == [(100.0, 160.0, 140.0, 210.0)]


def test_door_opening_mode_is_only_the_wall_opening(tmp_path):
    got = boxes_by_name(tmp_path, opening_mode="opening")
    assert got["door"] == [(100.0, 200.0, 140.0, 210.0)]


def test_window_and_unmapped_classes(tmp_path):
    got = boxes_by_name(tmp_path)
    assert got["window"] == [(300.0, 0.0, 380.0, 10.0)]
    assert set(got) == {"door", "window", "toilet", "sink"}  # Misc and Wall are ignored


def test_fixture_transforms_compose_with_parent_set(tmp_path):
    got = boxes_by_name(tmp_path)
    assert got["toilet"] == [(410.0, 320.0, 440.0, 370.0)]


def test_fixture_rotation(tmp_path):
    # matrix(0,1,-1,0,600,100) rotates 90 degrees: (x, y) -> (600 - y, 100 + x)
    got = boxes_by_name(tmp_path)
    assert got["sink"] == [(580.0, 100.0, 600.0, 140.0)]


def test_yolo_line_is_normalised(tmp_path):
    svg = tmp_path / "model.svg"
    svg.write_text(SVG)
    door = next(b for b in parse_svg(svg, 800, 600, CLASSES) if CLASSES.names[b.cls] == "door")
    cls, cx, cy, w, h = door.to_yolo(800, 600).split()
    assert cls == "0"
    assert np.allclose([float(cx), float(cy), float(w), float(h)],
                       [120 / 800, 185 / 600, 40 / 800, 50 / 600], atol=1e-6)


def test_parse_transform_variants():
    assert np.allclose(parse_transform("translate(5 7)") @ [1, 1, 1], [6, 8, 1])
    assert np.allclose(parse_transform("scale(2)") @ [3, 4, 1], [6, 8, 1])
    assert np.allclose(parse_transform("rotate(90)") @ [1, 0, 1], [0, 1, 1], atol=1e-9)
    assert np.allclose(parse_transform("translate(10,0) scale(2)") @ [1, 1, 1], [12, 2, 1])


def test_read_split_strips_slashes(tmp_path):
    f = tmp_path / "train.txt"
    f.write_text("/high_quality/17/\n/colorful/3/\n\n")
    assert read_split(f) == ["high_quality/17", "colorful/3"]
