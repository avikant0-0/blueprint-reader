"""Symbol detection on large drawing pages.

A construction sheet rendered at a readable resolution is far larger than the
detector's input size, and a door on it may be only ~40 px wide. Shrinking the
whole page to 1024 px would erase the symbols, so we slice the page into
overlapping tiles, detect on each tile, shift the boxes back into page
coordinates and merge duplicates from the overlap with class-wise NMS.

The tiling and NMS are plain numpy so they can be unit-tested without a GPU.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Detection:
    cls: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["conf"] = round(d["conf"], 4)
        for k in ("x1", "y1", "x2", "y2"):
            d[k] = round(d[k], 1)
        return d

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2


def tile_origins(size: int, tile: int, overlap: float) -> list[int]:
    """Start offsets along one axis so tiles cover [0, size) with overlap."""
    if size <= tile:
        return [0]
    step = max(1, int(tile * (1 - overlap)))
    starts = list(range(0, size - tile, step))
    starts.append(size - tile)  # last tile sits flush with the edge
    return sorted(set(starts))


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.clip(area + areas - inter, 1e-9, None)


def nms(dets: list[Detection], iou_thr: float = 0.5) -> list[Detection]:
    """Class-wise greedy non-maximum suppression."""
    keep: list[Detection] = []
    for cls in sorted({d.cls for d in dets}):
        group = sorted((d for d in dets if d.cls == cls), key=lambda d: -d.conf)
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in group], dtype=float)
        alive = np.ones(len(group), dtype=bool)
        for i in range(len(group)):
            if not alive[i]:
                continue
            keep.append(group[i])
            if i + 1 < len(group):
                overlaps = iou_one_to_many(boxes[i], boxes[i + 1:])
                alive[i + 1:] &= overlaps < iou_thr
    return sorted(keep, key=lambda d: -d.conf)


class Detector:
    """Thin wrapper over an Ultralytics YOLO checkpoint with tiled inference."""

    def __init__(self, weights: str, *, tile: int = 1024, overlap: float = 0.2,
                 conf: float = 0.25, iou: float = 0.5, device: str | None = None):
        from ultralytics import YOLO  # imported lazily: heavy dependency

        self.model = YOLO(weights)
        self.tile, self.overlap, self.conf, self.iou = tile, overlap, conf, iou
        self.device = device

    def _predict_array(self, img: np.ndarray) -> list[Detection]:
        res = self.model.predict(img, imgsz=self.tile, conf=self.conf, iou=self.iou,
                                 device=self.device, verbose=False)[0]
        names = res.names
        out = []
        for xyxy, conf, cls in zip(res.boxes.xyxy.cpu().numpy(),
                                   res.boxes.conf.cpu().numpy(),
                                   res.boxes.cls.cpu().numpy()):
            out.append(Detection(names[int(cls)], float(conf), *map(float, xyxy)))
        return out

    def predict(self, image) -> list[Detection]:
        """image: PIL.Image or HxWx3 uint8 RGB array."""
        img = np.asarray(image.convert("RGB")) if hasattr(image, "convert") else np.asarray(image)
        h, w = img.shape[:2]
        dets: list[Detection] = []
        for y0 in tile_origins(h, self.tile, self.overlap):
            for x0 in tile_origins(w, self.tile, self.overlap):
                # Ultralytics treats numpy input as BGR, so flip the channels.
                crop = img[y0:y0 + self.tile, x0:x0 + self.tile, ::-1]
                for d in self._predict_array(np.ascontiguousarray(crop)):
                    dets.append(Detection(d.cls, d.conf, d.x1 + x0, d.y1 + y0,
                                          d.x2 + x0, d.y2 + y0))
        return nms(dets, self.iou)
