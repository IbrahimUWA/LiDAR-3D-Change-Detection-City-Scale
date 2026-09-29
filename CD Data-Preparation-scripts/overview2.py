"""Overview figure: one panel per 2023 source map (each in its own SLAM frame) showing the map footprint, the located
existing pairs (black) and the new pairs (red).  Writes figures/overview_maps.png"""
import json, os, glob
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

os.makedirs('figures', exist_ok=True)
fps = {os.path.basename(f)[:-4]: np.load(f) for f in glob.glob('footprints/M2023*.npz')}
loc = json.load(open('existing_located.json')); sel = json.load(open('selected.json'))
maps = sorted(fps)
fig, axs = plt.subplots(2, 3, figsize=(18, 12)); axs = axs.ravel()
for ax, m in zip(axs, maps):
    d = fps[m]; x = d['cx'] * 2.0; y = d['cy'] * 2.0
    dens = np.log10(d['n'] + 1)
    ax.scatter(x, y, c=dens, s=0.6, cmap='Blues', vmin=0.5, vmax=4.5, marker='s', linewidths=0, rasterized=True)
    for p, e in loc.items():
        if e.get('map', e.get('map2023')) != m: continue
        cx, cy = e['centre_new_frame']; hx, hy = e['half_size_m']
        ax.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, fill=False, ec='#0b0b0b', lw=1.5))
        ax.text(cx - hx, cy + hy + 6, f"{p} ({e['score']:.2f})", fontsize=8, color='#0b0b0b')
    for s in sel:
        if s['map'] != m: continue
        ax.add_patch(Rectangle((s['xmin'], s['ymin']), 80, 80, fill=False, ec='#e34948', lw=2))
        ax.text(s['xmin'], s['ymax'] + 6, f"{s['pair']} +{s['loop']}", fontsize=8, color='#e34948', fontweight='bold')
    ax.set_title(f"2023 map {m[6:19]}  ({int(d['n'].sum()):,} pts)", fontsize=10); ax.set_aspect('equal')
    ax.set_xlabel('x (m, this map\'s own SLAM frame)'); ax.set_ylabel('y (m)'); ax.tick_params(colors='#52514e')
    for sp in ax.spines.values(): sp.set_color('#c3c2b7')
axs[-1].axis('off')
axs[-1].text(0.02, 0.9, 'Legend\n\nBlue: 2023 map footprint (2 m cells, darker = more points)\n'
             'Black: existing Pair1-8, located in the current data by relief correlation\n   (number = correlation score)\n'
             'Red: new Pair9-20 (80 m x 80 m), label shows the 2025 loop used\n\n'
             'Each 2023 map and each 2025 loop is in its own SLAM frame,\nso panels cannot be overlaid on one another.',
             fontsize=10, va='top', color='#0b0b0b')
fig.suptitle('Subiaco change-detection pairs: where the existing and new crops sit in the 2023 source maps', fontsize=13)
plt.tight_layout(); plt.savefig('figures/overview_maps.png', dpi=110); print('figures/overview_maps.png written')
