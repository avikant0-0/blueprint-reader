"""Fine-tune the page classifier and compare it with the keyword baseline.

Usage:
    python scripts/train_page_classifier.py --data data/pages \
        --test-pdfs setA,setB --epochs 15

--test-pdfs names the PDFs (file stems) whose pages are held out. Splitting by
PDF, not by page, matters: pages of one drawing set look alike, so a per-page
split leaks and inflates the score.

Writes models/page_classifier.pt and results/page_classifier.json.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from blueprint_reader.classify import LABELS, build_model, classify_text, page_transform
from blueprint_reader.metrics import classification_report


def pdf_stem(path: Path) -> str:
    return re.sub(r"_p\d{4}$", "", path.stem)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--test-pdfs", required=True, help="comma-separated PDF stems to hold out")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--out", default="models/page_classifier.pt")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="start from random weights (offline use; expect worse results)")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    held_out = {s.strip() for s in args.test_pdfs.split(",") if s.strip()}
    items = [(p, LABELS.index(label))
             for label in LABELS for p in sorted((Path(args.data) / label).glob("*.jpg"))]
    train = [it for it in items if pdf_stem(it[0]) not in held_out]
    test = [it for it in items if pdf_stem(it[0]) in held_out]
    if not train or not test:
        raise SystemExit("Need pages in both train and test; check --test-pdfs.")
    print(f"train pages: {len(train)}  test pages: {len(test)}")

    class Pages(Dataset):
        def __init__(self, rows, tf):
            self.rows, self.tf = rows, tf

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            path, y = self.rows[i]
            return self.tf(Image.open(path).convert("RGB")), y

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(pretrained=not args.no_pretrained).to(device)
    # Weight classes inversely to frequency: "other" usually dominates.
    freq = torch.bincount(torch.tensor([y for _, y in train]), minlength=len(LABELS)).float()
    weight = (freq.sum() / (freq.clamp(min=1) * len(LABELS))).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    train_dl = DataLoader(Pages(train, page_transform(True, args.size)),
                          batch_size=args.batch, shuffle=True, num_workers=2)
    test_dl = DataLoader(Pages(test, page_transform(False, args.size)),
                         batch_size=args.batch, num_workers=2)

    def predict_all():
        model.eval()
        preds = []
        with torch.no_grad():
            for x, _ in test_dl:
                preds += model(x.to(device)).argmax(1).cpu().tolist()
        return preds

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        sched.step()
        rep = classification_report([y for _, y in test], predict_all(), LABELS)
        print(f"epoch {epoch + 1:02d}  loss {total / len(train):.4f}  "
              f"test acc {rep['accuracy']:.3f}  macro-F1 {rep['macro_f1']:.3f}")

    y_true = [y for _, y in test]
    cnn = classification_report(y_true, predict_all(), LABELS)

    # Keyword baseline on the same held-out pages, from the saved text layer.
    base_pred = []
    for path, _ in test:
        txt = path.with_suffix(".txt")
        label, _ = classify_text(txt.read_text() if txt.exists() else "")
        base_pred.append(LABELS.index(label))
    baseline = classification_report(y_true, base_pred, LABELS)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "labels": LABELS, "size": args.size}, out)
    summary = {"train_pages": len(train), "test_pages": len(test),
               "held_out_pdfs": sorted(held_out), "cnn": cnn, "keyword_baseline": baseline}
    Path("results").mkdir(exist_ok=True)
    Path("results/page_classifier.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"cnn": {k: cnn[k] for k in ("accuracy", "macro_f1")},
                      "keyword_baseline": {k: baseline[k] for k in ("accuracy", "macro_f1")}},
                     indent=2))


if __name__ == "__main__":
    main()
