"""Page classification: what kind of sheet is this?

Two classifiers share one label set so they can be compared on the same data:

* `classify_text`  - keyword rules over the PDF text layer. Zero training, and
                     useless on scanned pages. This is the baseline.
* `PageClassifier` - a fine-tuned ResNet-18 that looks at the rendered page, so
                     it also works on scans.
"""
from __future__ import annotations

import re
from pathlib import Path

LABELS = ["floor_plan", "schedule", "specification", "other"]

_RULES: dict[str, list[tuple[str, float]]] = {
    "schedule": [
        (r"\bdoor\s+(?:and|&)?\s*(?:frame\s+)?schedule\b", 5),
        (r"\b(?:opening|hardware|finish|window|room\s+finish)\s+schedule\b", 4),
        (r"\bschedule\b", 1),
        (r"\bhardware\s+set\b", 2),
        (r"\bfire\s+rating\b", 1),
    ],
    "floor_plan": [
        (r"\b(?:first|second|third|fourth|ground|upper|lower|basement|typical)\s+(?:floor|level)\b", 3),
        (r"\bfloor\s+plan\b", 5),
        (r"\blevel\s+\d+\s+(?:floor\s+)?plan\b", 4),
        (r"\b(?:unit|enlarged|partial|overall)\s+plan\b", 3),
        (r"\bscale\s*:?\s*\d/\d+\s*\"\s*=\s*1'", 1),
    ],
    "specification": [
        (r"\bsection\s+\d{2}\s?\d{2}\s?\d{2}\b", 5),
        (r"\bpart\s+[123]\s*[-–]\s*(?:general|products|execution)\b", 5),
        (r"\bsubmittals?\b", 1),
        (r"\bquality\s+assurance\b", 1),
        (r"\bproject\s+manual\b", 3),
    ],
}


def classify_text(text: str) -> tuple[str, dict[str, float]]:
    """Keyword baseline. Returns (label, per-label scores)."""
    t = re.sub(r"\s+", " ", text.lower())
    scores = {label: 0.0 for label in LABELS}
    for label, rules in _RULES.items():
        for pattern, weight in rules:
            hits = len(re.findall(pattern, t))
            scores[label] += weight * min(hits, 3)
    # Dense prose with no drawing keywords is almost always a spec page.
    if len(t.split()) > 400 and scores["floor_plan"] == 0 and scores["schedule"] < 4:
        scores["specification"] += 2
    best = max(scores, key=scores.get)
    if scores[best] < 3:
        best = "other"
    return best, scores


def build_model(num_classes: int = len(LABELS), pretrained: bool = True):
    import torch.nn as nn
    from torchvision import models

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def page_transform(train: bool, size: int = 448):
    """Pages are resized whole (not cropped): layout is the signal."""
    from torchvision import transforms as T

    steps = [T.Grayscale(num_output_channels=3), T.Resize((size, size))]
    if train:
        steps += [T.RandomAffine(degrees=2, translate=(0.03, 0.03), scale=(0.95, 1.05), fill=255)]
    steps += [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return T.Compose(steps)


class PageClassifier:
    def __init__(self, weights: str | Path, device: str | None = None):
        import torch

        ckpt = torch.load(str(weights), map_location="cpu")
        self.labels = ckpt.get("labels", LABELS)
        self.size = ckpt.get("size", 448)
        self.model = build_model(len(self.labels), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.tf = page_transform(train=False, size=self.size)

    def predict(self, image) -> tuple[str, float]:
        import torch

        x = self.tf(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = torch.softmax(self.model(x), dim=1)[0]
        idx = int(prob.argmax())
        return self.labels[idx], float(prob[idx])
