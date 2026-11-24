"""Convert CubiCasa5K SVG annotations into axis-aligned boxes.

CubiCasa5K ships one folder per floor plan with `F1_scaled.png` and `model.svg`.
The SVG coordinates line up with `F1_scaled.png` (this is what the official
loader assumes). Annotations we use:

* doors:    `<g id="Door" ...>` whose first child `<polygon>` is the opening in
            the wall; nested groups hold the door panel and swing arc.
* windows:  `<g id="Window" ...>`, same structure.
* fixtures: `<g class="FixedFurniture <Name> ..." transform="matrix(...)">`,
            usually with a `<g class="BoundaryPolygon">` outline, sometimes
            nested in a `<g class="FixedFurnitureSet" transform=...>`.

The official loader (github.com/CubiCasa/CubiCasa5k, floortrans/loaders) builds
segmentation masks. We need detection boxes, so this module walks the same
elements and computes bounding boxes, applying every ancestor transform.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from xml.dom import minidom

import numpy as np
import yaml

_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_TRANSFORM = re.compile(r"(matrix|translate|scale|rotate)\s*\(([^)]*)\)")


@dataclass(frozen=True)
class Box:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    def to_yolo(self, img_w: int, img_h: int) -> str:
        cx = (self.x1 + self.x2) / 2 / img_w
        cy = (self.y1 + self.y2) / 2 / img_h
        return f"{self.cls} {cx:.6f} {cy:.6f} {self.w / img_w:.6f} {self.h / img_h:.6f}"


@dataclass(frozen=True)
class ClassMap:
    names: list[str]
    by_id: dict[str, int]
    by_furniture: dict[str, int]

    @classmethod
    def load(cls, path: str | Path) -> "ClassMap":
        cfg = yaml.safe_load(Path(path).read_text())
        names, by_id, by_furn = [], {}, {}
        for idx, entry in enumerate(cfg["classes"]):
            names.append(entry["name"])
            for i in entry.get("ids", []):
                by_id[i] = idx
            for f in entry.get("furniture", []):
                by_furn[f] = idx
        return cls(names, by_id, by_furn)


# ---------------------------------------------------------------- transforms
def parse_transform(text: str) -> np.ndarray:
    """Parse an SVG `transform` attribute into a 3x3 matrix."""
    m = np.eye(3)
    for name, args in _TRANSFORM.findall(text or ""):
        v = [float(x) for x in _NUM.findall(args)]
        t = np.eye(3)
        if name == "matrix" and len(v) == 6:
            a, b, c, d, e, f = v
            t = np.array([[a, c, e], [b, d, f], [0, 0, 1]])
        elif name == "translate" and v:
            t[0, 2], t[1, 2] = v[0], (v[1] if len(v) > 1 else 0.0)
        elif name == "scale" and v:
            t[0, 0], t[1, 1] = v[0], (v[1] if len(v) > 1 else v[0])
        elif name == "rotate" and v:
            r = math.radians(v[0])
            rot = np.array([[math.cos(r), -math.sin(r), 0],
                            [math.sin(r), math.cos(r), 0],
                            [0, 0, 1]])
            if len(v) == 3:
                cx, cy = v[1], v[2]
                to = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
                back = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
                rot = to @ rot @ back
            t = rot
        m = m @ t
    return m


def _global_transform(node) -> np.ndarray:
    """Compose transforms from the outermost ancestor down to `node`."""
    chain = []
    while node is not None and node.nodeType == node.ELEMENT_NODE:
        if node.hasAttribute("transform"):
            chain.append(parse_transform(node.getAttribute("transform")))
        node = node.parentNode
    m = np.eye(3)
    for t in reversed(chain):
        m = m @ t
    return m


# ------------------------------------------------------------------ geometry
def _pairs(text: str) -> np.ndarray:
    nums = [float(x) for x in _NUM.findall(text or "")]
    nums = nums[: len(nums) // 2 * 2]
    return np.array(nums, dtype=float).reshape(-1, 2)


def _f(node, name: str, default: float = 0.0) -> float:
    v = node.getAttribute(name)
    return float(v) if v not in ("", None) else default


def _local_points(node) -> np.ndarray:
    """Points (in the node's own coordinate system) that bound its shape."""
    tag = node.nodeName
    if tag in ("polygon", "polyline"):
        return _pairs(node.getAttribute("points"))
    if tag == "rect":
        x, y = _f(node, "x"), _f(node, "y")
        w, h = _f(node, "width"), _f(node, "height")
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    if tag == "line":
        return np.array([[_f(node, "x1"), _f(node, "y1")], [_f(node, "x2"), _f(node, "y2")]])
    if tag in ("circle", "ellipse"):
        cx, cy = _f(node, "cx"), _f(node, "cy")
        rx = _f(node, "r") if tag == "circle" else _f(node, "rx")
        ry = _f(node, "r") if tag == "circle" else _f(node, "ry")
        return np.array([[cx - rx, cy - ry], [cx + rx, cy - ry],
                         [cx + rx, cy + ry], [cx - rx, cy + ry]])
    if tag == "path":
        d = node.getAttribute("d")
        if not d:
            return np.empty((0, 2))
        try:
            from svgpathtools import parse_path

            xmin, xmax, ymin, ymax = parse_path(d).bbox()
            return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        except Exception:
            return np.empty((0, 2))  # unparsable path: ignore rather than guess
    return np.empty((0, 2))


def _shape_nodes(node, *, direct_only: bool = False):
    shapes = ("polygon", "polyline", "rect", "line", "circle", "ellipse", "path")
    if direct_only:
        return [c for c in node.childNodes if c.nodeName in shapes]
    out = []
    stack = list(node.childNodes)
    while stack:
        c = stack.pop()
        if c.nodeType != c.ELEMENT_NODE:
            continue
        if c.nodeName in shapes:
            out.append(c)
        else:
            stack.extend(c.childNodes)
    return out


def _bbox(nodes) -> tuple[float, float, float, float] | None:
    pts = []
    for n in nodes:
        local = _local_points(n)
        if len(local) == 0:
            continue
        m = _global_transform(n)
        homo = np.c_[local, np.ones(len(local))] @ m.T
        pts.append(homo[:, :2])
    if not pts:
        return None
    allp = np.vstack(pts)
    if not np.isfinite(allp).all():
        return None
    return (float(allp[:, 0].min()), float(allp[:, 1].min()),
            float(allp[:, 0].max()), float(allp[:, 1].max()))


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _opening_box(g, mode: str):
    """Box for a door/window group.

    mode="opening": only the wall opening (first direct child polygon).
    mode="full":    everything drawn in the group, so a door box includes its
                    swing arc. Falls back to the opening if the full box looks
                    implausible (far larger than the opening).
    """
    opening = _bbox([c for c in g.childNodes if c.nodeName == "polygon"][:1])
    if mode == "opening":
        return opening
    full = _bbox(_shape_nodes(g))
    if full is None:
        return opening
    if opening is not None and _area(opening) > 0 and _area(full) > 60 * _area(opening):
        return opening
    return full


def _furniture_box(g):
    boundary = [c for c in g.childNodes
                if c.nodeName == "g" and c.getAttribute("class") == "BoundaryPolygon"]
    if boundary:
        b = _bbox(_shape_nodes(boundary[0]))
        if b is not None:
            return b
    return _bbox(_shape_nodes(g))


# ---------------------------------------------------------------------- main
def parse_svg(svg_path: str | Path, img_w: int, img_h: int, classes: ClassMap,
              *, opening_mode: str = "full", min_size: float = 4.0) -> list[Box]:
    """Return clipped boxes for every annotated object we map to a class."""
    doc = minidom.parse(str(svg_path))
    boxes: list[Box] = []
    for g in doc.getElementsByTagName("g"):
        gid = g.getAttribute("id")
        gclass = g.getAttribute("class")
        cls_idx, raw = None, None
        if gid in classes.by_id:
            cls_idx, raw = classes.by_id[gid], _opening_box(g, opening_mode)
        elif gclass.startswith("FixedFurniture "):
            parts = gclass.split()
            name = parts[1] if len(parts) > 1 else ""
            if name in classes.by_furniture:
                cls_idx, raw = classes.by_furniture[name], _furniture_box(g)
        if cls_idx is None or raw is None:
            continue
        x1, y1 = max(0.0, raw[0]), max(0.0, raw[1])
        x2, y2 = min(float(img_w), raw[2]), min(float(img_h), raw[3])
        if x2 - x1 < min_size or y2 - y1 < min_size:
            continue
        boxes.append(Box(cls_idx, x1, y1, x2, y2))
    return boxes


def read_split(path: str | Path) -> list[str]:
    """Read train.txt / val.txt / test.txt; entries look like `/high_quality/17/`."""
    return [ln.strip().strip("/") for ln in Path(path).read_text().splitlines() if ln.strip()]
