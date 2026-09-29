"""Per-pair preview figures: top-down height maps of the aligned 2023 and 2025 crops plus a height-difference map.
Usage: python make_figures.py PairA PairB ...   (default: Pair1..Pair20). Writes figures/<pair>.png and figures/contact_sheet.png"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from align import load

ROOT = "E:/Subiaco Change Detection point cloud data/Pairs"
CELL = 0.5
pairs = sys.argv[1:] or [f'Pair{i}' for i in range(1, 21)]
os.makedirs('figures', exist_ok=True)


def hmap(P, lo, hi):
    nx = int(np.ceil((hi[0] - lo[0]) / CELL)); ny = int(np.ceil((hi[1] - lo[1]) / CELL))
    ix = np.clip(((P[:, 0] - lo[0]) / CELL).astype(int), 0, nx - 1); iy = np.clip(((P[:, 1] - lo[1]) / CELL).astype(int), 0, ny - 1)
    g = np.full((ny, nx), -np.inf); np.maximum.at(g, (iy, ix), P[:, 2]); g[~np.isfinite(g)] = np.nan
    return g


thumbs = []
for pr in pairs:
    d = f'{ROOT}/{pr}'
    if not os.path.exists(f'{d}/2023.ply'):
        print('skip', pr); continue
    A = load(f'{d}/2023.ply'); B = load(f'{d}/2025.ply')
    lo = np.minimum(A[:, :2].min(0), B[:, :2].min(0)); hi = np.maximum(A[:, :2].max(0), B[:, :2].max(0))
    ga, gb = hmap(A, lo, hi), hmap(B, lo, hi)
    z0 = np.nanpercentile(np.r_[ga.ravel(), gb.ravel()], 1); z1 = np.nanpercentile(np.r_[ga.ravel(), gb.ravel()], 99)
    diff = gb - ga
    ext = (lo[0], hi[0], lo[1], hi[1])
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, g, ttl in ((axs[0], ga, f'{pr}: 2023 (OS1-64) height'), (axs[1], gb, f'{pr}: 2025 (OS1-128) height')):
        im = ax.imshow(g, origin='lower', extent=ext, cmap='Blues', vmin=z0, vmax=z1, interpolation='nearest')
        ax.set_title(ttl, fontsize=10); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label='height above 2025 ground (m)')
    im = axs[2].imshow(diff, origin='lower', extent=ext, cmap='RdBu_r', vmin=-5, vmax=5, interpolation='nearest')
    axs[2].set_title(f'{pr}: 2025 minus 2023 max height (m)', fontsize=10); axs[2].set_xlabel('x (m)'); axs[2].set_ylabel('y (m)')
    plt.colorbar(im, ax=axs[2], fraction=0.046, pad=0.03, label='height change (m); grey = missing in one epoch')
    for ax in axs:
        ax.set_facecolor('#e5e4df'); ax.tick_params(colors='#52514e')
        for sp in ax.spines.values(): sp.set_color('#c3c2b7')
    both = np.isfinite(ga) & np.isfinite(gb)
    stats = {'cells_both': int(both.sum()), 'cells_2023_only': int((np.isfinite(ga) & ~np.isfinite(gb)).sum()),
             'cells_2025_only': int((~np.isfinite(ga) & np.isfinite(gb)).sum()),
             'dz_median_m': float(np.nanmedian(diff[both])), 'dz_p90_abs_m': float(np.nanpercentile(np.abs(diff[both]), 90)),
             'frac_cells_abs_dz_gt_1m': float((np.abs(diff[both]) > 1).mean()),
             'frac_cells_abs_dz_gt_2m': float((np.abs(diff[both]) > 2).mean())}
    fig.suptitle(f"{pr}  |  {len(A):,} pts (2023) vs {len(B):,} pts (2025)  |  {stats['frac_cells_abs_dz_gt_1m']*100:.1f}% of shared 0.5 m cells changed height by >1 m", fontsize=11)
    plt.tight_layout(); plt.savefig(f'figures/{pr}.png', dpi=110); plt.close(fig)
    json.dump(stats, open(f'figures/{pr}_stats.json', 'w'))
    thumbs.append((pr, gb, ext, z0, z1))
    print(pr, stats, flush=True)

if thumbs:
    n = len(thumbs); cols = 5; rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    for ax in axs.ravel(): ax.axis('off')
    for ax, (pr, g, ext, z0, z1) in zip(axs.ravel(), thumbs):
        ax.imshow(g, origin='lower', extent=ext, cmap='Blues', vmin=z0, vmax=z1, interpolation='nearest'); ax.set_title(pr, fontsize=10)
    fig.suptitle('2025 height maps of all change-detection pairs (Pair1-8 existing, Pair9-20 new)', fontsize=12)
    plt.tight_layout(); plt.savefig('figures/contact_sheet.png', dpi=110); print('contact sheet written')
