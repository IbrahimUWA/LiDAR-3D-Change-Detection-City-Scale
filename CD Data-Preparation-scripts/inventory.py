"""Inventory of the raw Subiaco data: file, size, points, XY extent, footprint area (from 2 m footprints)."""
import os, glob, csv, numpy as np
R = "E:/Subiaco Change Detection point cloud data"
LOOP_TIME = {1:'18:45',2:'18:50',3:'18:52',4:'18:55',5:'18:59',6:'19:02',7:'19:22',8:'19:20',9:'19:18',10:'19:14',11:'19:13',12:'19:10',13:'19:08',
             14:'19:04',15:'20:10',16:'19:35',17:'19:40',18:'20:06',19:'19:48',20:'19:58',21:'unknown'}
rows = []
for f in sorted(glob.glob(f"{R}/Subiaco2023/2023*.ply")):
    tag = os.path.basename(f)[:13]; fp = f'footprints/M2023_{tag}.npz'
    rows.append(('2023', os.path.basename(f), f"2023-07-15 {tag[9:11]}:{tag[11:13]}", 'Ouster OS1-64 (SN 122217000110)', f, fp))
for i in range(1, 22):
    f = f"{R}/Subiaco2025/Loop{i}.ply"
    if os.path.exists(f):
        rows.append(('2025', f'Loop{i}.ply', f"2025-02-26 {LOOP_TIME[i]}", 'Ouster OS1-128 (SN 122313001294)', f, f'footprints/Loop{i}.npz'))
out = []
for epoch, name, when, sensor, f, fp in rows:
    sz = os.path.getsize(f) / 1e9
    if os.path.exists(fp):
        d = np.load(fp); n = int(d['n'].sum()); area = len(d['n']) * 4 / 1e4
        ext = f"x[{d['cx'].min()*2:.0f},{d['cx'].max()*2+2:.0f}] y[{d['cy'].min()*2:.0f},{d['cy'].max()*2+2:.0f}]"
        zr = f"[{d['zmin'].min():.0f},{d['zmax'].max():.0f}]"
    else:
        n, area, ext, zr = None, None, 'n/a', 'n/a'
    out.append({'epoch': epoch, 'file': name, 'captured': when, 'sensor': sensor, 'size_GB': round(sz, 2), 'points': n,
                'footprint_ha': round(area, 1) if area else None, 'xy_extent_m': ext, 'z_range_m': zr})
with open('inventory.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
for o in out: print(o)
print('TOTAL 2023:', sum(o['points'] or 0 for o in out if o['epoch']=='2023'), 'pts', round(sum(o['size_GB'] for o in out if o['epoch']=='2023'),1), 'GB')
print('TOTAL 2025:', sum(o['points'] or 0 for o in out if o['epoch']=='2025'), 'pts', round(sum(o['size_GB'] for o in out if o['epoch']=='2025'),1), 'GB')
