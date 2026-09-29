"""Locate each existing pair's raw 2023 crop inside the (re-registered) 2023 map footprint held here.
Search yaw + XY shift by masked normalised cross-correlation of 2 m relief (zmax-zmin) images.
Writes existing_located.json"""
import json, os, glob, sys, time
import numpy as np
from scipy.signal import fftconvolve
from align import load, level

ROOT = "E:/Subiaco Change Detection point cloud data/Pairs"
CELL = 2.0
fps = {os.path.basename(f)[:-4]: np.load(f) for f in glob.glob('footprints/*.npz')}
LOOPMAP = {'1845':'Loop1','1852':'Loop3','1855':'Loop4','1902':'Loop6','1940':'Loop17'}
EPOCH = os.environ.get('EPOCH', '2023')


def map_images(k):
    d = fps[k]; x0, y0 = int(d['cx'].min()), int(d['cy'].min())
    W, H = int(d['cx'].max()) - x0 + 1, int(d['cy'].max()) - y0 + 1
    R = np.zeros((H, W)); Q = np.zeros((H, W))
    R[d['cy'] - y0, d['cx'] - x0] = np.minimum(d['zmax'] - d['zmin'], 20.0); Q[d['cy'] - y0, d['cx'] - x0] = 1
    return R, Q, x0, y0


def template(P, yaw_deg):
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    xy = P[:, :2] @ np.array([[c, -s], [s, c]]).T
    ix = np.floor(xy[:, 0] / CELL).astype(int); iy = np.floor(xy[:, 1] / CELL).astype(int)
    ix -= ix.min(); iy -= iy.min()
    nx, ny = ix.max() + 1, iy.max() + 1
    zmin = np.full((ny, nx), np.inf); zmax = np.full((ny, nx), -np.inf)
    np.minimum.at(zmin, (iy, ix), P[:, 2]); np.maximum.at(zmax, (iy, ix), P[:, 2])
    M = np.isfinite(zmin).astype(float)
    T = np.where(M > 0, np.minimum(zmax - zmin, 20.0), 0.0)
    return T, M, xy.mean(0) - np.array([ix.min() * CELL, iy.min() * CELL])


def masked_ncc(R, Q, T, M):
    """Padfield masked NCC of template T (mask M) over image R (mask Q); returns correlation map (same size as R)."""
    Tf, Mf = T[::-1, ::-1], M[::-1, ::-1]
    conv = lambda a, b: fftconvolve(a, b, mode='same')
    n = conv(Q, Mf); n = np.maximum(n, 1.0)
    sR = conv(R * Q, Mf); sT = conv(Q, (T * M)[::-1, ::-1])
    sRT = conv(R * Q, (T * M)[::-1, ::-1])
    sRR = conv(R * R * Q, Mf); sTT = conv(Q, (T * T * M)[::-1, ::-1])
    num = sRT - sR * sT / n
    den = np.sqrt(np.maximum(sRR - sR ** 2 / n, 1e-9) * np.maximum(sTT - sT ** 2 / n, 1e-9))
    cc = num / den
    cc[n < 0.5 * M.sum()] = -1          # require at least half the template overlapping map data
    return cc


def search(R, Q, P, yaws):
    best = (-2, None)
    for yw in yaws:
        T, M, _ = template(P, yw)
        cc = masked_ncc(R, Q, T, M)
        iy, ix = np.unravel_index(np.argmax(cc), cc.shape)
        if cc[iy, ix] > best[0]:
            best = (float(cc[iy, ix]), (yw, int(ix), int(iy), T.shape))
    return best


out = {}
pairs = sys.argv[1:] or [f'Pair{i}' for i in range(1, 9)]
for pr in pairs:
    t0 = time.time()
    f23 = [f for f in os.listdir(f'{ROOT}/{pr}') if EPOCH in f and 'Cloud' in f][0]
    tag = f23.replace('Upper City', '')[:13]; k = ('M2023_' + tag) if EPOCH == '2023' else LOOPMAP[tag[9:13]]
    P = load(f'{ROOT}/{pr}/{f23}')
    P = P - P.mean(0)
    Pl, _, tilt, _, _ = level(P)
    R, Q, x0, y0 = map_images(k)
    sc, (yw, ix, iy, shp) = search(R, Q, Pl, np.arange(0, 360, 3))
    sc, (yw, ix, iy, shp) = search(R, Q, Pl, np.arange(yw - 3, yw + 3.01, 0.5))
    sc2, (yw2, ix2, iy2, shp2) = search(R, Q, Pl, [yw])
    # template centre lands at (ix, iy) in map image coords ('same' mode centres the kernel)
    T, M, cen = template(Pl, yw)
    cx = (x0 + ix) * CELL + CELL / 2; cy = (y0 + iy) * CELL + CELL / 2     # map coords of template centre cell
    # second-best peak for ambiguity check
    cc = masked_ncc(R, Q, T, M); flat = cc.copy()
    r = 25; flat[max(0, iy - r):iy + r, max(0, ix - r):ix + r] = -1
    sc_2nd = float(flat.max())
    out[pr] = {'map': k, 'yaw_deg': float(yw), 'score': round(sc, 3), 'second_peak': round(sc_2nd, 3), 'tilt_levelled_deg': round(float(tilt), 2),
               'centre_new_frame': [round(cx, 1), round(cy, 1)], 'template_cells': list(shp), 'n_pts': int(len(P)),
               'half_size_m': [round(float(np.ptp(Pl[:, 0]) / 2), 1), round(float(np.ptp(Pl[:, 1]) / 2), 1)], 'seconds': round(time.time() - t0, 1)}
    print(pr, json.dumps(out[pr]), flush=True)
json.dump(out, open(f'existing_located{"" if EPOCH == "2023" else "_2025"}.json', 'w'), indent=1)
