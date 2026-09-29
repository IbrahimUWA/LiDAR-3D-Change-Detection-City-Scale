"""Level, centre and co-register a 2023/2025 crop pair.
Usage: python align.py raw2023.ply raw2025.ply out_dir [tag]
Writes out_dir/2023.ply, out_dir/2025.ply, out_dir/alignment.json"""
import sys, os, json, time
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

def header_len(path):
    n = 0
    with open(path, 'rb') as f:
        for line in f:
            n += 1
            if line.strip() == b'end_header': return n

def load(p):
    return pd.read_csv(p, sep=r'\s+', header=None, skiprows=header_len(p), usecols=[0,1,2],
                       dtype=np.float64, engine='c').to_numpy()

def write_ply(path, pts, comments=()):
    with open(path, 'w', newline='\n') as f:
        f.write('ply\nformat ascii 1.0\n')
        for c in comments: f.write(f'comment {c}\n')
        f.write(f'element vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n'
                'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        np.savetxt(f, pts, fmt='%.4f %.4f %.4f 0 0 0')

def ground_points(P, cell=1.0):
    """lowest point per 1 m XY cell -> candidate ground samples"""
    ij = np.floor(P[:, :2] / cell).astype(np.int64)
    key = (ij[:, 0] - ij[:, 0].min()) * 100000 + (ij[:, 1] - ij[:, 1].min())
    order = np.lexsort((P[:, 2], key))
    key_s = key[order]
    first = np.r_[True, key_s[1:] != key_s[:-1]]
    return P[order[first]]

def ransac_plane(G, iters=400, thr=0.15, rng=np.random.default_rng(0)):
    best_n, best_d, best_inl = None, None, -1
    for _ in range(iters):
        s = G[rng.choice(len(G), 3, replace=False)]
        n = np.cross(s[1]-s[0], s[2]-s[0]); nn = np.linalg.norm(n)
        if nn < 1e-9: continue
        n /= nn
        if n[2] < 0: n = -n
        if n[2] < 0.7: continue                       # reject near-vertical planes (walls)
        d = -n @ s[0]
        dist = np.abs(G @ n + d)
        inl = (dist < thr).sum()
        if inl > best_inl: best_n, best_d, best_inl = n, d, inl
    # refine with least squares on inliers
    m = np.abs(G @ best_n + best_d) < thr
    c = G[m].mean(0)
    _, _, vt = np.linalg.svd(G[m] - c)
    n = vt[2]; n = n if n[2] > 0 else -n
    return n, -n @ c, best_inl / len(G)

def rot_to_z(n):
    """rotation matrix taking unit vector n to +z"""
    z = np.array([0, 0, 1.0]); v = np.cross(n, z); s = np.linalg.norm(v); c = n @ z
    if s < 1e-12: return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)

def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    U, S, Vt = np.linalg.svd((A-ca).T @ (B-cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca

def icp(A, B, iters=60, trim=0.7, max_d=3.0, seed=0):
    """trimmed point-to-point ICP aligning A onto B; returns R,t (applied as A@R.T+t) and final median NN dist"""
    rng = np.random.default_rng(seed)
    sa = A[rng.choice(len(A), min(250000, len(A)), replace=False)]
    sb = B[rng.choice(len(B), min(400000, len(B)), replace=False)]
    tree = cKDTree(sb)
    R, t = np.eye(3), np.zeros(3)
    prev = np.inf
    for it in range(iters):
        cur = sa @ R.T + t
        d, idx = tree.query(cur, workers=-1)
        m = d < max_d
        k = int(trim * m.sum())
        sel = np.argsort(np.where(m, d, np.inf))[:k]
        Ri, ti = kabsch(cur[sel], sb[idx[sel]])
        R, t = Ri @ R, Ri @ t + ti
        err = np.median(d[sel])
        if abs(prev - err) < 1e-5: break
        prev = err
    cur = sa @ R.T + t
    d, _ = tree.query(cur, workers=-1)
    return R, t, float(np.median(d)), float(np.percentile(d, 90)), it + 1

def dsm(P, cell, x0, y0, nx, ny, zmax=15.0):
    ix = np.floor((P[:, 0] - x0) / cell).astype(int); iy = np.floor((P[:, 1] - y0) / cell).astype(int)
    m = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (P[:, 2] > -1) & (P[:, 2] < zmax)
    img = np.full((ny, nx), -np.inf)
    np.maximum.at(img, (iy[m], ix[m]), P[m, 2])
    occ = np.isfinite(img); img[~occ] = 0
    return img, occ

def coarse_xy(A, B, cell=0.5, search=25.0):
    """XY shift (dx,dy) to apply to A so it best overlays B: cross-correlation of clipped DSMs"""
    from scipy.signal import fftconvolve
    lo = np.minimum(A[:, :2].min(0), B[:, :2].min(0)) - search; hi = np.maximum(A[:, :2].max(0), B[:, :2].max(0)) + search
    nx, ny = int(np.ceil((hi[0]-lo[0]) / cell)), int(np.ceil((hi[1]-lo[1]) / cell))
    ia, oa = dsm(A, cell, lo[0], lo[1], nx, ny); ib, ob = dsm(B, cell, lo[0], lo[1], nx, ny)
    ia = (ia - ia[oa].mean()) * oa; ib = (ib - ib[ob].mean()) * ob
    cc = fftconvolve(ib, ia[::-1, ::-1], mode='same')
    ov = fftconvolve(ob.astype(float), oa[::-1, ::-1].astype(float), mode='same')
    cc = cc / np.maximum(ov, 0.05 * ov.max())
    cy, cx = np.array(cc.shape) // 2
    r = int(search / cell)
    win = cc[cy-r:cy+r+1, cx-r:cx+r+1]
    py, px = np.unravel_index(np.argmax(win), win.shape)
    return (px - r) * cell, (py - r) * cell

def level(P):
    G = ground_points(P)
    n, d, frac = ransac_plane(G)
    R = rot_to_z(n)
    p0 = np.array([0.0, 0.0, -d / n[2]])            # plane point under the XY origin
    Q = (P - p0) @ R.T
    tilt = np.degrees(np.arccos(np.clip(n[2], -1, 1)))
    return Q, R, tilt, 0.0, frac

def main(f23, f25, out, tag='', pre=None, clip=False):
    """pre=(yaw_deg, [tx, ty]): 2D rigid transform p_map = Rot(yaw).p_loop + t linking the loop frame to the map frame;
    the 2025 crop is first brought into the 2023 map frame with its inverse.  clip=True keeps only 2025 points inside the
    XY bounding box of the aligned 2023 crop, so both clouds cover the same footprint."""
    t0 = time.time()
    A = load(f23); B = load(f25)
    if pre is not None:
        yaw, tt = pre; c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        Rz = np.array([[c, -s], [s, c]])
        B[:, :2] = (B[:, :2] - np.asarray(tt)) @ Rz          # inverse rotation: (p - t) @ R  == R^T (p - t)
    ctr = (np.r_[A[:, :2].min(0), 0] + np.r_[A[:, :2].max(0), 0]) / 2   # XY centre of the crop box (2023 box)
    A = A - ctr; B = B - ctr
    Bl, RB, tiltB, gzB, fracB = level(B); Bl[:, 2] -= gzB
    Al, RA, tiltA, gzA, fracA = level(A); Al[:, 2] -= gzA
    # centre XY of the levelled clouds on the 2023 crop
    cB = np.r_[(Al[:, :2].min(0) + Al[:, :2].max(0)) / 2, 0]
    Bl -= cB; Al -= cB
    dx, dy = coarse_xy(Al, Bl)
    Al = Al + np.array([dx, dy, 0.0])
    R, t = np.eye(3), np.zeros(3); nit = 0
    for md in (4.0, 2.0, 1.0, 0.5):
        Ri, ti, med, p90, ni = icp(Al @ R.T + t, Bl, iters=30, max_d=md)
        R, t = Ri @ R, Ri @ t + ti; nit += ni
    t = t + R @ np.array([dx, dy, 0.0])          # fold the coarse shift into t (Al was shifted before)
    Al = Al - np.array([dx, dy, 0.0])
    Af = Al @ R.T + t
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    rot = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    n_b_raw = len(B)
    if clip:
        lo, hi = Af[:, :2].min(0), Af[:, :2].max(0)
        keep = (Bl[:, 0] >= lo[0]) & (Bl[:, 0] <= hi[0]) & (Bl[:, 1] >= lo[1]) & (Bl[:, 1] <= hi[1])
        Bl = Bl[keep]
    # reverse check
    d_ba, _ = cKDTree(Af).query(Bl[np.random.default_rng(1).choice(len(Bl), min(200000, len(Bl)), replace=False)], workers=-1)
    os.makedirs(out, exist_ok=True)
    cm = [f'Subiaco change-detection pair {tag}'.strip(), 'Generated 2026-09-27 by align.py (level by RANSAC ground plane, XY-centre, trimmed ICP 2023->2025)']
    write_ply(os.path.join(out, '2023.ply'), Af, cm + [f'source {os.path.basename(f23)}'])
    write_ply(os.path.join(out, '2025.ply'), Bl, cm + [f'source {os.path.basename(f25)}'])
    info = {'tag': tag, 'src2023': os.path.basename(f23), 'src2025': os.path.basename(f25),
            'n2023': int(len(A)), 'n2025': int(len(Bl)), 'n2025_raw_crop': int(n_b_raw), 'clipped_to_2023_footprint': bool(clip),
            'pre_transform_loop_to_map': None if pre is None else {'yaw_deg': float(pre[0]), 't': [float(v) for v in pre[1]]},
            'xy_centre_raw': ctr[:2].round(3).tolist(),
            'tilt2023_deg': round(float(tiltA), 3), 'tilt2025_deg': round(float(tiltB), 3),
            'ground_inlier_frac_2023': round(float(fracA), 3), 'ground_inlier_frac_2025': round(float(fracB), 3),
            'icp_rot_deg': round(float(rot), 3), 'icp_yaw_deg': round(float(yaw), 3), 'icp_t': t.round(3).tolist(), 'icp_iters': nit, 'coarse_xy_shift': [round(float(dx),2), round(float(dy),2)],
            'nn_median_2023to2025_m': round(med, 3), 'nn_p90_2023to2025_m': round(p90, 3),
            'nn_median_2025to2023_m': round(float(np.median(d_ba)), 3), 'nn_p90_2025to2023_m': round(float(np.percentile(d_ba, 90)), 3),
            'bbox2023': [Af.min(0).round(2).tolist(), Af.max(0).round(2).tolist()],
            'bbox2025': [Bl.min(0).round(2).tolist(), Bl.max(0).round(2).tolist()], 'seconds': round(time.time() - t0, 1)}
    json.dump(info, open(os.path.join(out, 'alignment.json'), 'w'), indent=1)
    print(json.dumps(info), flush=True)
    return info

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else '')
