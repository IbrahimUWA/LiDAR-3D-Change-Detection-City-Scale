#!/usr/bin/env python3
"""Export confusion matrices and metrics from the labels. Rows of each confusion matrix are ground truth; columns are predictions.
The JSON contains every run, loop, pooled all-loop and pooled test-loop matrix.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

CHANGE = ("Added", "Removed", "Increased", "Decreased", "Unchanged")
SEMANTIC = ("Ground", "Building", "Vegetation", "Mobile")
RUNS = ("default", "sor_k=10", "sor_k=40", "sor_std=1.0", "sor_std=3.0",
        "r_n=0.4", "r_n=1.6", "ransac=0.3", "ransac=1.2", "leaf=0.15",
        "leaf=0.5", "theta=1.0", "theta=1.5", "oracle_seg", "perturb_10",
        "perturb_20", "perturb_30", "no_lod", "no_class_assoc",
        "no_semantic", "pointwise")
SEM_RUNS = ("default", "perturb_10", "perturb_20", "perturb_30")
LOOPS = tuple(f"loop{i:02d}" for i in range(1, 21))


def cm_from_labels(gt, pred, k):
    if gt.shape != pred.shape or gt.ndim != 1 or np.any((gt < 0) | (gt >= k)) or np.any((pred < 0) | (pred >= k)):
        raise ValueError("Bad shape or class range")
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (gt, pred), 1)
    return cm


def describe(cm, classes):
    tp = np.diag(cm).astype(float)
    actual = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    fp, fn = predicted - tp, actual - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = 100 * np.divide(tp, tp + fp, out=np.full(len(classes), np.nan), where=(tp + fp) > 0)
        recall = 100 * np.divide(tp, tp + fn, out=np.full(len(classes), np.nan), where=(tp + fn) > 0)
        f1 = 100 * np.divide(2 * tp, 2 * tp + fp + fn, out=np.full(len(classes), np.nan), where=(2 * tp + fp + fn) > 0)
        iou = 100 * np.divide(tp, tp + fp + fn, out=np.full(len(classes), np.nan), where=(tp + fp + fn) > 0)

    def maybe(value):
        return None if np.isnan(value) else float(value)

    return {"labels": list(classes), "matrix": cm.tolist(),
            "n": int(cm.sum()), "correct": int(tp.sum()),
            "accuracy_pct": maybe(100 * tp.sum() / cm.sum()) if cm.sum() else None,
            "macro_f1_pct": maybe(np.nanmean(f1)),
            "macro_iou_pct": maybe(np.nanmean(iou)),
            "per_class": {
                name: {"actual": int(actual[i]), "predicted": int(predicted[i]),
                       "tp": int(tp[i]), "fp": int(fp[i]), "fn": int(fn[i]),
                       "precision_pct": maybe(precision[i]),
                       "recall_pct": maybe(recall[i]), "f1_pct": maybe(f1[i]),
                       "iou_pct": maybe(iou[i])}
                for i, name in enumerate(classes)}}


def load_evaluator(path):
    spec = importlib.util.spec_from_file_location("table_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize_cm(cm, classes, evaluator):
    report = describe(cm, classes)
    if evaluator:
        result = evaluator.scores(cm)
        assert np.isclose(report["accuracy_pct"], result["acc"])
        assert np.isclose(report["macro_f1_pct"], result["mf1"])
        assert np.isclose(report["macro_iou_pct"], result["miou"])
        assert np.allclose([report["per_class"][c]["iou_pct"] for c in classes], result["iou"])
    return report


def write_summary(out, result):
    all_default = result["change"]["pooled_all"]["default"]
    test_default = result["change"]["pooled_test"]["default"]
    semantic = result["semantic"]["pooled_all"]["default"]
    lines = ["# Confusion matrices from the 20-loops", "",
             "Rows are GT; columns are predictions. Class order for change: " + ", ".join(CHANGE) + ". "
             "Labels.", ""]

    for title, entry in (("Default, all 20 loops", all_default),
                         ("Default, test loops 01–05", test_default),
                         ("Default semantic, all 20 loops", semantic)):
        names = entry["labels"]
        lines += [f"## {title}", "", "| GT \\ prediction | " + " | ".join(names) + " | GT total |",
                  "| --- | " + " | ".join("---:" for _ in names) + " | ---: |"]
        for name, row in zip(names, entry["matrix"]):
            lines.append("| " + name + " | " + " | ".join(map(str, row)) + f" | {sum(row)} |")
        lines.append("| Predicted total | " + " | ".join(str(entry["per_class"][name]["predicted"]) for name in names) +
                     f" | {entry['n']} |")
        lines += ["", f"ACC {entry['accuracy_pct']:.1f}%, macro F1 {entry['macro_f1_pct']:.1f}%, "
                  f"macro IoU {entry['macro_iou_pct']:.1f}%.", ""]

    lines += ["## Per-run pooled change metrics", "",
              "| Run | All ACC | All mF1 | All mIoU | Test ACC | Test mF1 | Test mIoU |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for run in RUNS:
        a = result["change"]["pooled_all"][run]
        t = result["change"]["pooled_test"][run]
        lines.append(f"| {run} | {a['accuracy_pct']:.1f} | {a['macro_f1_pct']:.1f} | "
                     f"{a['macro_iou_pct']:.1f} | {t['accuracy_pct']:.1f} | "
                     f"{t['macro_f1_pct']:.1f} | {t['macro_iou_pct']:.1f} |")
    lines += ["", "`intermediate_results.json` includes all 420 per-loop change matrices, "
              "their pooled all/test matrices, 80 per-loop semantic matrices, their pooled matrices, "
              "and TP/FP/FN, GT and prediction counts, precision, recall, F1, and IoU for every class.",
              "", "Network activations, losses, gradients, association decisions, and runtime cannot "
              "be recovered from hard GT/prediction labels alone.", ""]
    (out / "SUMMARY.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fixture", type=Path)
    ap.add_argument("--out", type=Path, default=Path("intermediate_results"))
    ap.add_argument("--evaluator", type=Path, help="Original table script for independent cross-check")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    evaluator = load_evaluator(args.evaluator) if args.evaluator else None
    change_loops = {}
    semantic_loops = {}
    change_cm = {run: {} for run in RUNS}
    semantic_cm = {run: {} for run in SEM_RUNS}
    for lp in LOOPS:
        folder = args.fixture / lp
        gt_change = np.load(folder / "gt_change.npy")
        gt_sem = np.load(folder / "gt_sem.npy")
        change_loops[lp] = {}
        semantic_loops[lp] = {}
        for run in RUNS:
            pred = np.load(folder / "change" / f"{run}.npy")
            cm = cm_from_labels(gt_change, pred, 5)
            change_cm[run][lp] = cm
            change_loops[lp][run] = summarize_cm(cm, CHANGE, evaluator)
        for run in SEM_RUNS:
            path = folder / "pred_sem.npy" if run == "default" else folder / "sem" / f"{run}.npy"
            cm = cm_from_labels(gt_sem, np.load(path), 4)
            semantic_cm[run][lp] = cm
            semantic_loops[lp][run] = summarize_cm(cm, SEMANTIC, evaluator)

    def pooled(matrices, names, classes):
        return {run: summarize_cm(sum((by_loop[lp] for lp in names), np.zeros((len(classes), len(classes)), np.int64)),
                                  classes, evaluator)
                for run, by_loop in matrices.items()}

    result = {"note": "Ablation study results.",
              "matrix_orientation": "rows=ground truth; columns=prediction",
              "change": {"classes": CHANGE, "by_loop": change_loops,
                         "pooled_all": pooled(change_cm, LOOPS, CHANGE),
                         "pooled_test": pooled(change_cm, LOOPS[:5], CHANGE)},
              "semantic": {"classes": SEMANTIC, "by_loop": semantic_loops,
                           "pooled_all": pooled(semantic_cm, LOOPS, SEMANTIC)}}
    if evaluator:
        for run in RUNS:
            for key, loops in (("pooled_all", LOOPS), ("pooled_test", LOOPS[:5])):
                cm, count = evaluator.pooled_change_cm(str(args.fixture), loops, run)
                assert count == len(loops) and cm.tolist() == result["change"][key][run]["matrix"]
        for run in SEM_RUNS:
            cm, count = evaluator.pooled_sem_cm(str(args.fixture), LOOPS, None if run == "default" else run)
            assert count == 20 and cm.tolist() == result["semantic"]["pooled_all"][run]["matrix"]
    (args.out / "intermediate_results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    write_summary(args.out, result)
    print("PASS: 420 per-loop change matrices, 80 per-loop semantic matrices, 42 pooled change matrices, 4 pooled semantic matrices")


if __name__ == "__main__":
    main()
