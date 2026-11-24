"""Small, dependency-free metrics so results are easy to audit."""
from __future__ import annotations


def classification_report(y_true: list[int], y_pred: list[int], labels: list[str]) -> dict:
    n = len(labels)
    cm = [[0] * n for _ in range(n)]  # rows: true, cols: predicted
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    per_class, f1s = {}, []
    for i, name in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n)) - tp
        fn = sum(cm[i]) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        support = sum(cm[i])
        per_class[name] = {"precision": prec, "recall": rec, "f1": f1, "support": support}
        if support:
            f1s.append(f1)
    correct = sum(cm[i][i] for i in range(n))
    return {
        "accuracy": correct / len(y_true) if y_true else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": labels,
    }


def prf(tp: int, fp: int, fn: int) -> dict:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
