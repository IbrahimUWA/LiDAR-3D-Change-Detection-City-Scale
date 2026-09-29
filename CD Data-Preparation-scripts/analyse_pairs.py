"""For each existing pair: bboxes of raw CloudCompare crops, and the rigid transform Rhino applied (Kabsch, same point order)."""
import glob, os, re, json
import numpy as np, polars as pl

ROOT = "E:/Subiaco Change Detection point cloud data/Pairs"

def header_len(path):
    n = 0
    with open(path, 'rb') as f:
        for line in f:
            n += 1
            if line.strip() == b'end_header': return n

def load(path):
    hl = header_len(path)
    import pandas as pd
    df = pd.read_csv(path, sep=r'\s+', header=None, skiprows=hl, usecols=[0,1,2], dtype=np.float64, engine='c')
    return df.to_numpy()

def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    H = (A-ca).T @ (B-cb)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1,1,d])
    R = Vt.T @ D @ U.T
    t = cb - R @ ca
    res = np.linalg.norm((A @ R.T + t) - B, axis=1)
    ang = np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1)))
    yaw = np.degrees(np.arctan2(R[1,0], R[0,0]))
    return R, t, ang, yaw, res.mean(), res.max()

out = {}
for pd_ in sorted(glob.glob(ROOT + "/Pair*"), key=lambda p: int(re.findall(r'\d+', os.path.basename(p))[0])):
    name = os.path.basename(pd_)
    files = [f for f in os.listdir(pd_) if f.endswith('.ply')]
    c23 = [f for f in files if '2023' in f and 'Cloud' in f][0]
    c25 = [f for f in files if '2025' in f and 'Cloud' in f][0]
    rec = {'src2023': c23, 'src2025': c25}
    for yr, cf in (('2023', c23), ('2025', c25)):
        raw = load(os.path.join(pd_, cf)); rh = load(os.path.join(pd_, f'{yr}.ply'))
        rec[yr] = {'n_raw': len(raw), 'n_rhino': len(rh),
                   'bbox_raw': [raw.min(0).round(2).tolist(), raw.max(0).round(2).tolist()],
                   'bbox_rhino': [rh.min(0).round(2).tolist(), rh.max(0).round(2).tolist()]}
        if len(raw) == len(rh):
            R, t, ang, yaw, rm, rx = kabsch(raw, rh)
            rec[yr].update({'rot_deg': round(ang, 4), 'yaw_deg': round(yaw, 4), 't': t.round(3).tolist(),
                            'resid_mean': round(rm, 4), 'resid_max': round(rx, 4)})
    out[name] = rec
    b = rec['2023']['bbox_raw']; b5 = rec['2025']['bbox_raw']
    print(f"{name}: 2023 {c23[:13]} n={rec['2023']['n_raw']:,} xy[{b[0][0]:.0f},{b[1][0]:.0f}]x[{b[0][1]:.0f},{b[1][1]:.0f}] z[{b[0][2]:.1f},{b[1][2]:.1f}] | "
          f"2025 {c25[:13]} n={rec['2025']['n_raw']:,} xy[{b5[0][0]:.0f},{b5[1][0]:.0f}]x[{b5[0][1]:.0f},{b5[1][1]:.0f}] z[{b5[0][2]:.1f},{b5[1][2]:.1f}]")
    for yr in ('2023','2025'):
        r = rec[yr]
        if 'rot_deg' in r:
            print(f"   Rhino {yr}: rot={r['rot_deg']:.3f} deg (yaw {r['yaw_deg']:.3f}) t={r['t']} resid mean={r['resid_mean']} max={r['resid_max']}")
        else:
            print(f"   Rhino {yr}: count mismatch raw={r['n_raw']} rhino={r['n_rhino']} bbox_rhino={r['bbox_rhino']}")
json.dump(out, open('existing_pairs.json','w'), indent=1)
