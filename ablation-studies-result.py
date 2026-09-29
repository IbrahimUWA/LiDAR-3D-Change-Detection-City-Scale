"""
ablation-studies-result.py
=====================================
Computes the manuscript's Table 4 (sensitivity), Table 5 (segmentation accuracy
+ oracle / perturbation coupling) and Table 6 (ablation) from our pipeline
outputs evaluated against the Subiaco loops labels.

----------------------------------------------------------------------------
Expected input layout  (ROOT is given on the command line)
----------------------------------------------------------------------------
ROOT/
  loop01/ ... loop20/
      gt_change.npy                 int  in {0..4}  ground-truth change label per
                                    evaluation unit (point, or 0.5 m cell)
                                    0 Added, 1 Removed, 2 Increased, 3 Decreased, 4 Unchanged
      gt_sem.npy                    int  in {0..3}  manual semantic label per point
                                    0 ground, 1 building, 2 vegetation, 3 mobile
      pred_sem.npy                  unsupervised semantic label per point (same order as gt_sem)
      change/<run>.npy              predicted change label per evaluation unit, one file per run:
          default.npy
          sor_k=10.npy  sor_k=40.npy  sor_std=1.0.npy  sor_std=3.0.npy
          r_n=0.4.npy   r_n=1.6.npy   ransac=0.3.npy   ransac=1.2.npy
          leaf=0.15.npy leaf=0.5.npy  theta=1.0.npy    theta=1.5.npy
          oracle_seg.npy            change analysis re-run with manual semantic labels
          perturb_10.npy perturb_20.npy perturb_30.npy   (segmentation thresholds +-10/20/30 %)
          no_lod.npy no_class_assoc.npy no_semantic.npy pointwise.npy   (ablation variants)
      sem/<run>.npy                 (optional) predicted semantic labels for perturb_10/20/30,
                                    used for the "Seg. mIoU" column of Table 5 (bottom)
  runtime.csv                       (optional) columns: run, seconds   -> runtime ratio in the text

Units: gt_change and every change/<run>.npy must have identical length within a
loop. A value of -1 in gt_change marks units outside the visibility mask; they
are ignored everywhere.

Aggregation follows the manuscript: Tables 4 and 5 use all 20 loops; Table 6
uses the held-out test loops 01-05. Metrics are computed from the pooled
confusion matrix over the selected loops (micro accuracy, macro F1, macro IoU,
per-class IoU), so macro IoU is by construction the mean of the class IoUs.

Run:
    python make_tables_4_5_6_from_annotations.py ROOT  [--out TABLES_4_5_6.tex]
    python make_tables_4_5_6_from_annotations.py ROOT --check     # validate layout only
"""
import argparse
import os
import sys

import numpy as np

CHANGE_CLASSES = ["Added", "Removed", "Increased", "Decreased", "Unchanged"]
SEM_CLASSES = ["Ground", "Building", "Vegetation", "Mobile"]

SENSITIVITY = [  # (label, default value, [(run name, printed value)])
    (r"SOR neighbours $k$",               "20",   [("sor_k=10", "10"), ("sor_k=40", "40")]),
    (r"SOR std.\ ratio $\sigma$",         "2.0",  [("sor_std=1.0", "1.0"), ("sor_std=3.0", "3.0")]),
    (r"Normal radius $r_n$ (m)",          "0.8",  [("r_n=0.4", "0.4"), ("r_n=1.6", "1.6")]),
    (r"RANSAC inlier thr.\ (m)",          "0.6",  [("ransac=0.3", "0.3"), ("ransac=1.2", "1.2")]),
    (r"Finest voxel leaf (m)",            "0.25", [("leaf=0.15", "0.15"), ("leaf=0.5", "0.50")]),
    (r"LoD gate $\theta_{\mathrm{LoD}}$", "1.2",  [("theta=1.0", "1.0"), ("theta=1.5", "1.5")]),
]
ABLATION = [
    ("default",        r"Full pipeline (ours)"),
    ("no_lod",         r"w/o LoD gating (Eqs.~\ref{eq:lod},~\ref{eq:conf})"),
    ("no_class_assoc", r"w/o class-constrained association (Eq.~\ref{eq:assign})"),
    ("no_semantic",    r"w/o semantic/instance layer"),
    ("pointwise",      r"w/o object-centric decisions (point-wise)"),
]
COUPLING = [
    ("oracle_seg", "Oracle (manual labels)"),
    ("default",    "Unsupervised (default)"),
    ("perturb_10", r"Perturbed $\pm10\%$"),
    ("perturb_20", r"Perturbed $\pm20\%$"),
    ("perturb_30", r"Perturbed $\pm30\%$"),
]
ALL_LOOPS = [f"loop{i:02d}" for i in range(1, 21)]
TEST_LOOPS = ALL_LOOPS[:5]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def confusion(gt, pred, k):
    m = (gt >= 0) & (gt < k) & (pred >= 0) & (pred < k)
    return np.bincount(gt[m] * k + pred[m], minlength=k * k).reshape(k, k).astype(np.int64)


def scores(cm):
    """micro accuracy, macro F1, macro IoU, per-class IoU, per-class P/R (all in %)."""
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), np.nan)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
        prec = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        rec = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
    acc = tp.sum() / cm.sum() if cm.sum() else np.nan
    return dict(acc=100 * acc, mf1=100 * np.nanmean(f1), miou=100 * np.nanmean(iou),
                iou=100 * iou, prec=100 * prec, rec=100 * rec)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load(path):
    if path.endswith(".npy"):
        return np.load(path).astype(np.int64).ravel()
    return np.loadtxt(path, dtype=np.int64, delimiter=",").ravel()


def find(root, loop, rel):
    for ext in (".npy", ".csv"):
        p = os.path.join(root, loop, rel + ext)
        if os.path.exists(p):
            return p
    return None


def pooled_change_cm(root, loops, run):
    cm = np.zeros((5, 5), np.int64)
    n = 0
    for lp in loops:
        g, p = find(root, lp, "gt_change"), find(root, lp, f"change/{run}")
        if g is None or p is None:
            continue
        gt, pr = load(g), load(p)
        if len(gt) != len(pr):
            sys.exit(f"length mismatch in {lp}/{run}: gt {len(gt)} vs pred {len(pr)}")
        cm += confusion(gt, pr, 5)
        n += 1
    return cm, n


def pooled_sem_cm(root, loops, run=None):
    cm = np.zeros((4, 4), np.int64)
    n = 0
    for lp in loops:
        g = find(root, lp, "gt_sem")
        p = find(root, lp, "pred_sem") if run is None else find(root, lp, f"sem/{run}")
        if g is None or p is None:
            continue
        gt, pr = load(g), load(p)
        if len(gt) != len(pr):
            sys.exit(f"length mismatch in {lp} semantic labels")
        cm += confusion(gt, pr, 4)
        n += 1
    return cm, n


def check_layout(root):
    ok = True
    for lp in ALL_LOOPS:
        d = os.path.join(root, lp)
        if not os.path.isdir(d):
            print(f"[missing] {lp}/"); ok = False; continue
        for req in ("gt_change", "gt_sem", "pred_sem", "change/default"):
            if find(root, lp, req) is None:
                print(f"[missing] {lp}/{req}.npy"); ok = False
        runs = [r for _, _, rr in SENSITIVITY for r, _ in rr] + [r for r, _ in ABLATION] + [r for r, _ in COUPLING]
        for r in sorted(set(runs)):
            if find(root, lp, f"change/{r}") is None:
                print(f"[missing] {lp}/change/{r}.npy")
    print("layout OK" if ok else "layout incomplete (see above)")
    return ok


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def f(x, d=1):
    return "---" if x is None or np.isnan(x) else f"{x:.{d}f}"


def table4(root):
    d_cm, n = pooled_change_cm(root, ALL_LOOPS, "default")
    d = scores(d_cm)
    rows = []
    for label, dval, runs in SENSITIVITY:
        block = []
        entries = [(r, v, scores(pooled_change_cm(root, ALL_LOOPS, r)[0])) for r, v in runs]
        entries.append(("default", dval + "$^{*}$", d))
        entries.sort(key=lambda e: float(e[1].replace("$^{*}$", "")))
        for i, (r, v, s) in enumerate(entries):
            lab = rf"\multirow{{3}}{{*}}{{{label}}}" if i == 0 else ""
            block.append(f"{lab} & {v} & {f(s['acc'])} & {f(s['mf1'])} & {f(s['miou'])} \\\\")
        rows.append("\n".join(block))
    body = "\n\\midrule\n".join(rows)
    return rf"""
\begin{{table}}[t]
\centering
\caption{{Sensitivity of the full pipeline to the main pre-processing parameters on all {n} loops.
One parameter is varied at a time; defaults are marked with $^{{*}}$. All values are percentages.}}
\label{{tab:sensitivity}}
\scriptsize
\setlength{{\tabcolsep}}{{4.5pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{tabular}}{{llccc}}
\toprule
Parameter & Value & ACC & mF1 & mIoU \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
""", d


def table5(root, default_scores):
    cm, n = pooled_sem_cm(root, ALL_LOOPS)
    s = scores(cm)
    top = "\n".join(f"{c} & {f(s['prec'][i])} & {f(s['rec'][i])} & {f(s['iou'][i])} \\\\"
                    for i, c in enumerate(SEM_CLASSES))
    bottom = []
    for run, label in COUPLING:
        cd = scores(pooled_change_cm(root, ALL_LOOPS, run)[0]) if run != "default" else default_scores
        if run == "oracle_seg":
            seg = 100.0
        elif run == "default":
            seg = s["miou"]
        else:
            scm, sn = pooled_sem_cm(root, ALL_LOOPS, run)
            seg = scores(scm)["miou"] if sn else np.nan
        bottom.append(f"{label} & {f(seg)} & {f(cd['mf1'])} & {f(cd['miou'])} \\\\")
    bottom = "\n".join(bottom)
    return rf"""
\begin{{table}}[t]
\centering
\caption{{Top: accuracy of the unsupervised semantic segmentation against manual annotations on all {n} loops.
Bottom: effect of segmentation quality on change detection (CD) performance. All values are percentages.}}
\label{{tab:seg_acc}}
\scriptsize
\setlength{{\tabcolsep}}{{4.5pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{tabular}}{{lccc}}
\toprule
Class & Precision & Recall & IoU \\
\midrule
{top}
\midrule
Overall & \multicolumn{{2}}{{c}}{{OA = {f(s['acc'])}}} & mIoU = {f(s['miou'])} \\
\bottomrule
\toprule
Segmentation variant & Seg.\ mIoU & CD mF1 & CD mIoU \\
\midrule
{bottom}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def table6(root):
    rows = []
    for run, label in ABLATION:
        cm, n = pooled_change_cm(root, TEST_LOOPS, run)
        s = scores(cm)
        cells = [f(s["acc"]), f(s["mf1"]), f(s["miou"])] + [f(v) for v in s["iou"]]
        if run == "default":
            cells = [rf"\textbf{{{c}}}" for c in cells]
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    body = "\n".join(rows)
    return rf"""
\begin{{table*}}[t]
\centering
\caption{{Ablation study on the held-out test partition (loops 1--5). Each variant removes one component
of the full pipeline; all other parameters are fixed. All values are percentages.}}
\label{{tab:ablation}}
\small
\setlength{{\tabcolsep}}{{5.2pt}}
\renewcommand{{\arraystretch}}{{1.12}}
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}} l ccc ccccc}}
\toprule
Variant & ACC & mF1 & mIoU & Added & Removed & Increased & Decreased & Unchanged \\
\midrule
{body}
\bottomrule
\end{{tabular*}}
\end{{table*}}
"""


def runtime_note(root):
    p = os.path.join(root, "runtime.csv")
    if not os.path.exists(p):
        return ""
    import csv
    rt = {r["run"]: float(r["seconds"]) for r in csv.DictReader(open(p))}
    if "default" not in rt:
        return ""
    lines = [f"% runtime ratio vs default: {k} = {v / rt['default']:.2f}x" for k, v in sorted(rt.items())]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="folder with loop01..loop20 (see module docstring)")
    ap.add_argument("--out", default="TABLES_4_5_6.tex")
    ap.add_argument("--check", action="store_true", help="validate the input layout and exit")
    a = ap.parse_args()

    if a.check:
        sys.exit(0 if check_layout(a.root) else 1)
    if not check_layout(a.root):
        print("continuing with the loops/runs that are present; missing runs print as ---")

    t4, default_scores = table4(a.root)
    t5 = table5(a.root, default_scores)
    t6 = table6(a.root)
    header = ("% Generated by make_tables_4_5_6_from_annotations.py from real predictions vs manual labels.\n"
              f"% root = {os.path.abspath(a.root)}\n" + runtime_note(a.root))
    out = header + t4 + t5 + t6
    with open(a.out, "w", encoding="utf8") as fh:
        fh.write(out)
    print(out)
    print("written:", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
