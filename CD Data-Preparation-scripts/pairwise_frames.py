"""Every 2023 map is in its own SLAM frame and so is every 2025 loop.  For each (map, loop) combination estimate the
2D rigid transform  p_map = Rot(yaw) . p_loop + t  by yaw-searched masked NCC of 4 m relief images.
Writes pairwise_frames.json with score, ambiguity (best score at a yaw > 6 deg away), yaw, t, overlap cells."""
import numpy as np, glob, os, json, time, sys
from scipy.signal import fftconvolve
from multiprocessing import Pool

CELL = 4.0
fps = {os.path.basename(f)[:-4]: np.load(f) for f in glob.glob('footprints/*.npz')}
MAPS = sorted(k for k in fps if k.startswith('M2023')); LOOPS = sorted((k for k in fps if k.startswith('Loop')), key=lambda k: int(k[4:]))


def pts(k):
    d = fps[k]; return d['cx'] * 2.0 + 1, d['cy'] * 2.0 + 1, np.minimum(d['zmax'] - d['zmin'], 20.0)


def raster(x, y, rel, yaw=0.0):
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    xr, yr = c * x - s * y, s * x + c * y
    ix = np.floor(xr / CELL).astype(int); iy = np.floor(yr / CELL).astype(int); ox, oy = ix.min(), iy.min(); ix -= ox; iy -= oy
    R = np.zeros((iy.max() + 1, ix.max() + 1)); Q = np.zeros_like(R)
    np.maximum.at(R, (iy, ix), rel); Q[iy, ix] = 1
    return R, Q, ox, oy


def mncc(R, Q, T, M, min_overlap):
    conv = lambda a, b: fftconvolve(a, b, mode='full')
    Mf = M[::-1, ::-1]; n = conv(Q, Mf); nn = np.maximum(n, 1)
    sR = conv(R * Q, Mf); sT = conv(Q, (T * M)[::-1, ::-1]); sRT = conv(R * Q, (T * M)[::-1, ::-1])
    sRR = conv(R * R * Q, Mf); sTT = conv(Q, (T * T * M)[::-1, ::-1])
    cc = (sRT - sR * sT / nn) / np.sqrt(np.maximum(sRR - sR ** 2 / nn, 1e-9) * np.maximum(sTT - sT ** 2 / nn, 1e-9))
    cc[n < min_overlap] = -1
    return cc, n


def solve(args):
    m, l = args
    x23, y23, r23 = pts(m); x25, y25, r25 = pts(l)
    R, Q, ox, oy = raster(x23, y23, r23)
    min_ov = 700                                             # >= 700 cells of 4 m = 1.1 ha overlapping data
    res = []
    for yaw in np.arange(0, 360, 3.0):
        T, M, tx, ty = raster(x25, y25, r25, yaw)
        cc, n = mncc(R, Q, T, M, min_ov)
        iy, ix = np.unravel_index(np.argmax(cc), cc.shape)
        res.append((float(cc[iy, ix]), float(yaw), int(ix), int(iy), T.shape[1], T.shape[0], tx, ty, int(n[iy, ix])))
    res.sort(reverse=True)
    best = res[0]
    # refine yaw at 0.5 deg around best
    fine = []
    for yaw in np.arange(best[1] - 3, best[1] + 3.01, 0.5):
        T, M, tx, ty = raster(x25, y25, r25, yaw)
        cc, n = mncc(R, Q, T, M, min_ov)
        iy, ix = np.unravel_index(np.argmax(cc), cc.shape)
        fine.append((float(cc[iy, ix]), float(yaw), int(ix), int(iy), T.shape[1], T.shape[0], tx, ty, int(n[iy, ix])))
    fine.sort(reverse=True); sc, yaw, ix, iy, Tw, Th, tx, ty, nov = fine[0]
    amb = max([r[0] for r in res if min(abs(r[1] - yaw), 360 - abs(r[1] - yaw)) > 6] + [-1])
    t = [float((ix - (Tw - 1) + ox - tx) * CELL), float((iy - (Th - 1) + oy - ty) * CELL)]
    out = {'map': m, 'loop': l, 'score': round(sc, 3), 'ambiguity': round(amb, 3), 'yaw_deg': yaw, 't': t, 'overlap_cells4m': nov, 'overlap_ha': round(nov * 16 / 1e4, 1)}
    print(json.dumps(out), flush=True)
    return out


if __name__ == '__main__':
    combos = [(m, l) for m in MAPS for l in LOOPS]
    t0 = time.time()
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as p:
        out = p.map(solve, combos, chunksize=1)
    json.dump(out, open('pairwise_frames.json', 'w'), indent=1)
    print(f'done {len(out)} combos in {time.time()-t0:.0f}s')
