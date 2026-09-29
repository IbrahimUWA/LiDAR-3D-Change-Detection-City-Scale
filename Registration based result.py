"""
Experiments on the Subiaco Pairs data

Implements a headless version of the paper's geometric pipeline:
  voxel downsample -> SOR -> normals -> point-to-plane ICP refinement ->
  LoD95 grid (roughness MAD + sigma_reg from ICP Hessian) ->
  LoD-gated cross-epoch change detection -> DBSCAN object layer.

"""
import os, sys, io, json, time, zipfile, hashlib
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

ZIP_PATH = r"E:\KSA project\Change Detection Revision\Pairs.zip"
OUT_DIR  = r"E:\KSA project\Change Detection Revision\FInal Paper, Revision\Pair results"
CACHE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

PAIRS = [f"Pair{i}" for i in range(1, 9)]

DEFAULTS = dict(leaf=0.25, sor_k=20, sor_std=2.0, r_n=0.8, theta=1.2,
                icp_iters=8, icp_max_corr=1.0, icp_sample=200_000,
                dbscan_eps=1.0, dbscan_min=30, min_obj_pts=200)

# ----------------------------------------------------------------------------- IO
def load_ply_xyz(zf, name):
    """Parse ASCII ply (x y z r g b) -> Nx3 float32."""
    key = hashlib.md5(name.encode()).hexdigest()[:16]
    npy = os.path.join(CACHE, key + ".npy")
    if os.path.exists(npy):
        return np.load(npy)
    with zf.open(name) as f:
        header = b""
        while b"end_header" not in header:
            header += f.readline()
        data = np.loadtxt(io.BytesIO(f.read()), dtype=np.float32, usecols=(0, 1, 2))
    np.save(npy, data)
    return data

# ------------------------------------------------------------------- preprocessing
def voxel_down(pts, leaf):
    ijk = np.floor(pts / leaf).astype(np.int64)
    # unique voxel -> centroid
    order = np.lexsort((ijk[:, 2], ijk[:, 1], ijk[:, 0]))
    ijk_s, pts_s = ijk[order], pts[order]
    change = np.any(np.diff(ijk_s, axis=0) != 0, axis=1)
    starts = np.concatenate(([0], np.nonzero(change)[0] + 1, [len(pts_s)]))
    out = np.add.reduceat(pts_s.astype(np.float64), starts[:-1], axis=0)
    counts = np.diff(starts)
    return (out / counts[:, None]).astype(np.float32)

def sor_filter(pts, k, std_ratio):
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=k + 1, workers=-1)
    mean_d = d[:, 1:].mean(axis=1)
    thr = mean_d.mean() + std_ratio * mean_d.std()
    return pts[mean_d < thr]

def estimate_normals(pts, r_n, k=24):
    """PCA normals in radius-capped kNN neighborhoods.
    Also returns sigma_n = sqrt(lambda_3): the local surface roughness along the
    normal direction (the paper's cylindrical-neighborhood roughness)."""
    tree = cKDTree(pts)
    d, idx = tree.query(pts, k=k, workers=-1)
    mask = d < r_n                       # cap neighborhood by radius
    nb = pts[idx]                        # N,k,3
    w = mask.astype(np.float32)[..., None]
    cnt = np.maximum(w.sum(axis=1), 3)
    mu = (nb * w).sum(axis=1) / cnt
    diff = (nb - mu[:, None, :]) * w
    cov = np.einsum("nki,nkj->nij", diff, diff) / cnt[..., None]
    evals, evecs = np.linalg.eigh(cov)
    n = evecs[:, :, 0]
    n *= np.where(n[:, 2:3] < 0, -1.0, 1.0)  # orient upward-ish
    sigma_n = np.sqrt(np.maximum(evals[:, 0], 0)).astype(np.float32)
    return n, sigma_n

# --------------------------------------------------------------------------- ICP
def icp_point_to_plane(src, dst, dst_n, iters, max_corr, sample):
    """Refine alignment of src onto dst. Returns T(4x4), rmse, inlier_ratio, sigma_reg."""
    rng = np.random.default_rng(0)
    tree = cKDTree(dst)
    T = np.eye(4)
    src_c = src.copy()
    rmse, inlier_ratio, sigma_reg = np.nan, np.nan, np.nan
    for it in range(iters):
        sel = rng.choice(len(src_c), min(sample, len(src_c)), replace=False)
        p = src_c[sel].astype(np.float64)
        d, j = tree.query(p, k=1, distance_upper_bound=max_corr, workers=-1)
        ok = np.isfinite(d)
        p, q, n = p[ok], dst[j[ok]].astype(np.float64), dst_n[j[ok]].astype(np.float64)
        r = np.einsum("ij,ij->i", n, p - q)
        # Tukey weights (tau = 0.3 m)
        tau = 0.3
        w = np.where(np.abs(r) < tau, (1 - (r / tau) ** 2) ** 2, 0.0)
        A = np.hstack([np.cross(p, n), n])           # M x 6
        Aw = A * w[:, None]
        H = Aw.T @ A
        g = Aw.T @ r
        try:
            x = np.linalg.solve(H + 1e-9 * np.eye(6), -g)
        except np.linalg.LinAlgError:
            break
        alpha, beta, gamma_, tx, ty, tz = x
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        cg, sg = np.cos(gamma_), np.sin(gamma_)
        R = np.array([[cg * cb, cg * sb * sa - sg * ca, cg * sb * ca + sg * sa],
                      [sg * cb, sg * sb * sa + cg * ca, sg * sb * ca - cg * sa],
                      [-sb, cb * sa, cb * ca]])
        Ti = np.eye(4); Ti[:3, :3] = R; Ti[:3, 3] = [tx, ty, tz]
        src_c = (Ti[:3, :3] @ src_c.T).T + Ti[:3, 3]
        T = Ti @ T
        # stats on the last iteration
        inl = np.abs(r) < tau
        rmse = float(np.sqrt(np.mean(r[inl] ** 2))) if inl.any() else float("nan")
        inlier_ratio = float(inl.mean())
        if it == iters - 1:
            s2 = np.sum(w * r ** 2) / max(np.sum(w > 0) - 6, 1)
            try:
                cov = s2 * np.linalg.inv(H + 1e-9 * np.eye(6))
                sigma_reg = float(np.sqrt(np.trace(cov[3:, 3:]) / 3.0))  # mean transl. std
            except np.linalg.LinAlgError:
                sigma_reg = float("nan")
    return T, src_c.astype(np.float32), rmse, inlier_ratio, sigma_reg

# --------------------------------------------------------------------------- LoD
def lod_grid(a, b, sigma_reg, cell=5.0):
    """LoD95 per 5m XY cell using plane-residual MAD roughness of both epochs."""
    xy_min = np.minimum(a[:, :2].min(0), b[:, :2].min(0))
    def cell_stats(p):
        ij = np.floor((p[:, :2] - xy_min) / cell).astype(np.int64)
        key = ij[:, 0] * 100000 + ij[:, 1]
        order = np.argsort(key)
        key_s, p_s = key[order], p[order]
        starts = np.concatenate(([0], np.nonzero(np.diff(key_s))[0] + 1, [len(p_s)]))
        stats = {}
        for s, e in zip(starts[:-1], starts[1:]):
            pts = p_s[s:e]
            if len(pts) < 10:
                continue
            c = pts.mean(0)
            q = pts - c
            cov = q.T @ q / len(q)
            evals, evecs = np.linalg.eigh(cov)
            nrm = evecs[:, 0]
            rres = q @ nrm
            mad = 1.4826 * np.median(np.abs(rres - np.median(rres)))
            stats[key_s[s]] = (mad, len(pts))
        return stats
    sa, sb = cell_stats(a), cell_stats(b)
    lod = {}
    for k in set(sa) | set(sb):
        m1, n1 = sa.get(k, (0.15, 10))
        m2, n2 = sb.get(k, (0.15, 10))
        # per-POINT level of detection: do not divide the roughness terms by N
        # (Eq. 3 of the paper gives the cell-mean LoD; gating individual points
        #  must budget the full single-observation noise of both epochs)
        lod[k] = 1.96 * np.sqrt(m1 ** 2 + m2 ** 2 + sigma_reg ** 2)
    def lookup(p):
        ij = np.floor((p[:, :2] - xy_min) / cell).astype(np.int64)
        key = ij[:, 0] * 100000 + ij[:, 1]
        default = float(np.median(list(lod.values()))) if lod else 0.1
        return np.array([lod.get(k, default) for k in key], dtype=np.float32)
    return lod, lookup

# ----------------------------------------------------------------- change detection
def change_detect(a, b, na, nb, sna, snb, sigma_reg, theta, gated=True, fixed_thr=None,
                  dbscan_eps=1.0, dbscan_min=30, min_obj_pts=200, objectify=True):
    """Flag changed points in each epoch; cluster into objects. Returns dict of stats + masks.

    Change evidence is the normal-direction displacement |n·(p−q)| to the nearest
    cross-epoch neighbour (median over k=3 NNs for robustness), which removes the
    tangential sampling-offset component that raw Euclidean NN distance carries
    after voxelization (Eq. 7 of the paper). The gate is the per-point level of
    detection LoD95 = 1.96*sqrt(sigma_n,src^2 + sigma_n,dst(NN)^2 + sigma_reg^2)
    built from the local PCA roughness of both epochs (paper Eq. 3, per-point form)."""
    tree_b, tree_a = cKDTree(b), cKDTree(a)

    def normal_dist(src, src_n, tree, dst, dst_sn):
        d_e, j = tree.query(src, k=3, workers=-1)
        # normal-projected distance to each of the 3 NNs, take the median
        diff = src[:, None, :] - dst[j]                      # N,3,3
        dn = np.abs(np.einsum("nij,ni->nj", diff, src_n))    # N,3
        return np.median(dn, axis=1), d_e[:, 0], dst_sn[j[:, 0]]

    d_ab, de_ab, snb_at_a = normal_dist(a, na, tree_b, b, snb)  # 2023->2025 (removed)
    d_ba, de_ba, sna_at_b = normal_dist(b, nb, tree_a, a, sna)  # 2025->2023 (added)
    # a genuinely missing structure has BOTH large euclidean and large normal dist;
    # take euclidean over when it clearly exceeds grazing-geometry scale
    d_ab = np.where(de_ab > 0.30, np.maximum(d_ab, de_ab), d_ab)
    d_ba = np.where(de_ba > 0.30, np.maximum(d_ba, de_ba), d_ba)
    lod_a = 1.96 * np.sqrt(sna ** 2 + snb_at_a ** 2 + sigma_reg ** 2)
    lod_b = 1.96 * np.sqrt(snb ** 2 + sna_at_b ** 2 + sigma_reg ** 2)
    if gated:
        thr_a = np.maximum(theta * lod_a, 0.10)
        thr_b = np.maximum(theta * lod_b, 0.10)
    else:
        t = fixed_thr if fixed_thr is not None else 0.10
        thr_a = np.full(len(a), t, np.float32)
        thr_b = np.full(len(b), t, np.float32)
    m_a = d_ab > thr_a
    m_b = d_ba > thr_b
    res = dict(raw_flag_frac_2023=float(m_a.mean()), raw_flag_frac_2025=float(m_b.mean()),
               lod_median_m=round(float(np.median(lod_a)), 4),
               lod_p90_m=round(float(np.percentile(lod_a, 90)), 4))
    obj_masks, counts = [], []
    for pts, m in ((a, m_a), (b, m_b)):
        keep = np.zeros(len(pts), bool)
        nobj = 0
        if objectify and m.sum() > dbscan_min:
            sub = pts[m]
            lab = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min, n_jobs=-1).fit(sub).labels_
            for l in range(lab.max() + 1):
                sel = lab == l
                if sel.sum() >= min_obj_pts:
                    idx = np.nonzero(m)[0][sel]
                    keep[idx] = True
                    nobj += 1
        elif not objectify:
            keep = m
            nobj = -1
        obj_masks.append(keep)
        counts.append(nobj)
    res.update(obj_flag_frac_2023=float(obj_masks[0].mean()),
               obj_flag_frac_2025=float(obj_masks[1].mean()),
               n_objects_removed=counts[0], n_objects_added=counts[1])
    return res, obj_masks[0], obj_masks[1]

def stable_cell_fp_rate(a, b, mask_a, cell=2.0, dz_stable=0.05):
    """FP proxy: flagged-point rate inside cells whose median elevation difference
    between epochs is < dz_stable (assumed genuinely unchanged ground/facade)."""
    xy_min = np.minimum(a[:, :2].min(0), b[:, :2].min(0))
    def med_z(p):
        ij = np.floor((p[:, :2] - xy_min) / cell).astype(np.int64)
        key = ij[:, 0] * 100000 + ij[:, 1]
        order = np.argsort(key)
        key_s, z_s = key[order], p[order, 2]
        starts = np.concatenate(([0], np.nonzero(np.diff(key_s))[0] + 1, [len(z_s)]))
        return {key_s[s]: np.median(z_s[s:e]) for s, e in zip(starts[:-1], starts[1:]) if e - s >= 20}
    za, zb = med_z(a), med_z(b)
    stable = {k for k in set(za) & set(zb) if abs(za[k] - zb[k]) < dz_stable}
    ij = np.floor((a[:, :2] - xy_min) / cell).astype(np.int64)
    key = ij[:, 0] * 100000 + ij[:, 1]
    in_stable = np.fromiter((k in stable for k in key), bool, len(key))
    if in_stable.sum() == 0:
        return float("nan"), 0
    return float(mask_a[in_stable].mean()), int(in_stable.sum())

def mask_jaccard(pts, mask1, mask2, leaf=0.5):
    """Agreement of two change masks over the same points, voxelized."""
    if mask1.sum() == 0 and mask2.sum() == 0:
        return 1.0
    v1 = set(map(tuple, np.floor(pts[mask1] / leaf).astype(np.int64)))
    v2 = set(map(tuple, np.floor(pts[mask2] / leaf).astype(np.int64)))
    if not v1 and not v2:
        return 1.0
    return len(v1 & v2) / max(len(v1 | v2), 1)

# --------------------------------------------------------------- semantic split
def unsup_semantics(pts, normals):
    """Very light geometric semantic split: ground / building / vegetation / other."""
    z = pts[:, 2]
    # local ground level on 5 m cells
    xy_min = pts[:, :2].min(0)
    ij = np.floor((pts[:, :2] - xy_min) / 5.0).astype(np.int64)
    key = ij[:, 0] * 100000 + ij[:, 1]
    order = np.argsort(key)
    key_s = key[order]
    starts = np.concatenate(([0], np.nonzero(np.diff(key_s))[0] + 1, [len(key_s)]))
    gz = np.empty(len(pts), np.float32)
    for s, e in zip(starts[:-1], starts[1:]):
        idx = order[s:e]
        gz[idx] = np.percentile(z[idx], 5)
    hag = z - gz                              # height above ground
    vert = np.abs(normals[:, 2])              # |n.ez|
    ground = (hag < 0.3) & (vert > 0.85)
    building = (~ground) & (vert < 0.25) & (hag > 2.0)
    # roughness proxy: local normal variance
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=10, workers=-1)
    nvar = 1.0 - np.linalg.norm(normals[idx].mean(axis=1), axis=1)
    vegetation = (~ground) & (~building) & (nvar > 0.15) & (hag > 0.5)
    other = ~(ground | building | vegetation)
    return dict(ground=float(ground.mean()), building=float(building.mean()),
                vegetation=float(vegetation.mean()), other=float(other.mean()))

# ------------------------------------------------------------------------ runner
def save_ply(path, pts, rgb):
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        arr = np.zeros(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("r", "u1"), ("g", "u1"), ("b", "u1")])
        arr["x"], arr["y"], arr["z"] = pts.T
        arr["r"], arr["g"], arr["b"] = rgb
        arr.tofile(f)

def preprocess(raw, leaf, sor_k, sor_std, r_n):
    p = voxel_down(raw, leaf)
    p = sor_filter(p, sor_k, sor_std)
    n, sn = estimate_normals(p, r_n)
    return p, n, sn

def run_config(raw23, raw25, cfg, tag, keep_masks=False):
    t0 = time.time()
    a, na, sna = preprocess(raw23, cfg["leaf"], cfg["sor_k"], cfg["sor_std"], cfg["r_n"])
    b, nb, snb = preprocess(raw25, cfg["leaf"], cfg["sor_k"], cfg["sor_std"], cfg["r_n"])
    T, a_ref, rmse, inl, sreg = icp_point_to_plane(
        a, b, nb, cfg["icp_iters"], cfg["icp_max_corr"], cfg["icp_sample"])
    if not np.isfinite(sreg):
        sreg = 0.02
    gated = cfg.get("gated", True)
    objectify = cfg.get("objectify", True)
    fixed_thr = cfg.get("fixed_thr")
    # rotate source normals by the ICP rotation before projecting
    na_ref = (T[:3, :3] @ na.T).T.astype(np.float32)
    res, m23, m25 = change_detect(a_ref, b, na_ref, nb, sna, snb, sreg, cfg["theta"],
                                  gated=gated, fixed_thr=fixed_thr,
                                  dbscan_eps=cfg["dbscan_eps"], dbscan_min=cfg["dbscan_min"],
                                  min_obj_pts=cfg["min_obj_pts"], objectify=objectify)
    fp, n_stable = stable_cell_fp_rate(a_ref, b, m23)
    out = dict(tag=tag, n_2023=len(a), n_2025=len(b),
               icp_rmse_m=round(rmse, 4), icp_inlier_ratio=round(inl, 4),
               sigma_reg_m=round(sreg, 4),
               fp_rate_stable_cells=round(fp, 5), n_stable_pts=n_stable,
               runtime_s=round(time.time() - t0, 1), **{k: round(v, 5) if isinstance(v, float) else v for k, v in res.items()})
    extras = dict(a=a_ref, b=b, m23=m23, m25=m25, na=na, nb=nb) if keep_masks else None
    return out, extras

def main():
    zf = zipfile.ZipFile(ZIP_PATH)
    all_rows, sens_rows, abl_rows, sem_rows = [], [], [], []
    for pair in PAIRS:
        print(f"########## {pair} ##########", flush=True)
        raw23 = load_ply_xyz(zf, f"{pair}/2023.ply")
        raw25 = load_ply_xyz(zf, f"{pair}/2025.ply")
        print(f"{pair}: raw 2023={len(raw23):,}  2025={len(raw25):,}", flush=True)

        # ---- default run (also yields masks for PLY export + reference masks)
        cfg = dict(DEFAULTS)
        row, extras = run_config(raw23, raw25, cfg, "default", keep_masks=True)
        row["pair"] = pair
        all_rows.append(row)
        print(json.dumps(row), flush=True)
        a, b, m23_ref, m25_ref = extras["a"], extras["b"], extras["m23"], extras["m25"]

        # export change PLYs (default config)
        pdir = os.path.join(OUT_DIR, "change_maps")
        os.makedirs(pdir, exist_ok=True)
        if m23_ref.sum():
            save_ply(os.path.join(pdir, f"{pair}_removed_2023.ply"), a[m23_ref], (220, 30, 30))
        if m25_ref.sum():
            save_ply(os.path.join(pdir, f"{pair}_added_2025.ply"), b[m25_ref], (30, 160, 30))

        # ---- semantics distribution (default preprocessing)
        sem = unsup_semantics(a, extras["na"])
        sem["pair"] = pair
        sem_rows.append(sem)

        # ---- sensitivity sweep (one-at-a-time)
        sweeps = [("sor_k", [10, 40]), ("sor_std", [1.0, 3.0]), ("r_n", [0.4, 1.6]),
                  ("leaf", [0.15, 0.50]), ("theta", [1.0, 1.5])]
        for pname, vals in sweeps:
            for v in vals:
                c = dict(DEFAULTS); c[pname] = v
                r, ex = run_config(raw23, raw25, c, f"{pname}={v}", keep_masks=True)
                r["pair"] = pair; r["param"] = pname; r["value"] = v
                # mask agreement vs default requires same point set -> use voxel jaccard on 2023 side
                # (point sets differ across configs; voxelized masks are comparable)
                v1 = set(map(tuple, np.floor(a[m23_ref] / 0.5).astype(np.int64)))
                v2 = set(map(tuple, np.floor(ex["a"][ex["m23"]] / 0.5).astype(np.int64)))
                r["mask_jaccard_vs_default"] = round(len(v1 & v2) / max(len(v1 | v2), 1), 4) if (v1 or v2) else 1.0
                sens_rows.append(r)
                print(json.dumps({k: r[k] for k in ("pair", "param", "value", "icp_rmse_m", "obj_flag_frac_2023", "fp_rate_stable_cells", "mask_jaccard_vs_default")}), flush=True)

        # add default as reference row in sensitivity table
        dref = dict(row); dref["param"] = "default"; dref["value"] = "-"; dref["mask_jaccard_vs_default"] = 1.0
        sens_rows.append(dref)

        # ---- ablations
        ablations = [
            ("full", dict()),
            ("no_lod_gating", dict(gated=False, fixed_thr=0.10)),
            ("no_object_layer", dict(objectify=False)),
            ("no_sor", dict(sor_std=1e9)),
        ]
        for name, over in ablations:
            if name == "full":
                r = dict(row); r["variant"] = "full"
            else:
                c = dict(DEFAULTS); c.update(over)
                r, _ = run_config(raw23, raw25, c, name)
                r["pair"] = pair; r["variant"] = name
            abl_rows.append(r)
            print(json.dumps({k: r.get(k) for k in ("pair", "variant", "obj_flag_frac_2023", "raw_flag_frac_2023", "fp_rate_stable_cells", "n_objects_removed", "n_objects_added")}), flush=True)

        # incremental save
        import pandas as pd
        pd.DataFrame(all_rows).to_csv(os.path.join(OUT_DIR, "registration_lod_summary.csv"), index=False)
        pd.DataFrame(sens_rows).to_csv(os.path.join(OUT_DIR, "sensitivity_raw.csv"), index=False)
        pd.DataFrame(abl_rows).to_csv(os.path.join(OUT_DIR, "ablation_raw.csv"), index=False)
        pd.DataFrame(sem_rows).to_csv(os.path.join(OUT_DIR, "segmentation_distribution.csv"), index=False)
    print("ALL DONE", flush=True)

if __name__ == "__main__":
    main()
