"""Select N new 80 m boxes.  Each 2023 map and 2025 loop is in its own frame, so:
 1. use the pairwise map<->loop rigid fits (pairwise_frames.json) to predict overlap and generate candidate boxes in the map frame
 2. verify every candidate by a local relief-NCC search of the box template inside the loop (exact yaw + position)
 3. reject duplicates (same physical place) by template-vs-template NCC against existing pairs and already-accepted boxes
Writes selected.json"""
import numpy as np, glob, os, json, sys, time
from scipy.signal import fftconvolve
from pairwise_frames import fps, raster, mncc, CELL as C4

N_NEW = int(sys.argv[1]) if len(sys.argv) > 1 else 12
BOX = 80.0; CELL = 2.0; NB = int(BOX / CELL); STEP = 5
MIN_JOINT = 0.6; MIN_PTS = 300_000; SEP = 130.0
GHOST_Q10 = float(os.environ.get('GHOST_Q10', 1.0)); GHOST_Q25 = float(os.environ.get('GHOST_Q25', 3.0))
MIN_SCORE = 0.70; MIN_MARGIN = 0.15; MAX_PER_LOOP = 3; MAX_PER_MAP = 4
SRC23 = "E:/Subiaco Change Detection point cloud data/Subiaco2023"; SRC25 = "E:/Subiaco Change Detection point cloud data/Subiaco2025"
LOOP_TIME = {1: '1845', 2: '1850', 3: '1852', 4: '1855', 5: '1859', 6: '1902', 7: '1922', 8: '1920', 9: '1918', 10: '1914', 11: '1913', 12: '1910',
             13: '1908', 14: '1904', 15: '2010', 16: '1935', 17: '1940', 18: '2006', 19: '1948', 20: '1958', 21: 'xxxx'}
ANCHORS = {('M2023_20230715_1439', 'Loop3'), ('M2023_20230715_1450', 'Loop4'), ('M2023_20230715_1450', 'Loop6'),
           ('M2023_20230715_1514', 'Loop17'), ('M2023_20230715_1459', 'Loop17')}

pw = json.load(open('pairwise_frames.json'))
# override map<->loop transforms with the ones implied by the located existing pairs (more reliable than the global fit)
_l23 = json.load(open('existing_located.json')); _l25 = json.load(open('existing_located_2025.json'))
_corr = {}
for _pr in _l23:
    if _l23[_pr]['score'] < 0.7 or _l25[_pr]['score'] < 0.7: continue
    _corr.setdefault((_l23[_pr].get('map', _l23[_pr].get('map2023')), _l25[_pr]['map']), []).append((_pr, np.array(_l23[_pr]['centre_new_frame']), np.array(_l25[_pr]['centre_new_frame']),
                                                                     _l23[_pr]['yaw_deg'] - _l25[_pr]['yaw_deg']))
for (_m, _l), _cs in _corr.items():
    if len(_cs) >= 2:
        A_ = np.array([c[1] for c in _cs]); B_ = np.array([c[2] for c in _cs])      # A_ = map, B_ = loop ; p_map = R p_loop + t
        ca, cb = A_.mean(0), B_.mean(0); H = (B_ - cb).T @ (A_ - ca); U, S, Vt = np.linalg.svd(H); R_ = Vt.T @ U.T
        if np.linalg.det(R_) < 0: Vt[1] *= -1; R_ = Vt.T @ U.T
        yaw_ = float(np.degrees(np.arctan2(R_[1, 0], R_[0, 0]))); t_ = ca - R_ @ cb
    else:
        yaw_ = float((_cs[0][3] + 180) % 360 - 180); c_, s_ = np.cos(np.radians(yaw_)), np.sin(np.radians(yaw_))
        t_ = _cs[0][1] - np.array([[c_, -s_], [s_, c_]]) @ _cs[0][2]
    for o in pw:
        if o['map'] == _m and o['loop'] == _l:
            print(f"override {_m[6:]}<->{_l}: pairwise yaw {o['yaw_deg']} t {o['t']} score {o['score']}  ->  anchor yaw {yaw_:.2f} t {t_.round(1).tolist()} from {[c[0] for c in _cs]}")
            o['yaw_deg'] = yaw_; o['t'] = t_.tolist(); o['score'] = max(o['score'], 0.9); o['overlap_ha'] = max(o['overlap_ha'], 3.0); o['anchor'] = [c[0] for c in _cs]
combos = [o for o in pw if (o['score'] >= 0.5 and o['overlap_ha'] >= 3.0) or (o['map'], o['loop']) in ANCHORS]
print(f'{len(combos)} map/loop combinations considered')
loc = json.load(open('existing_located.json'))


def rot(yaw):
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw)); return np.array([[c, -s], [s, c]])


def relief_img(k, cell=CELL):
    d = fps[k]; x0, y0 = int(d['cx'].min()), int(d['cy'].min()); W, H = int(d['cx'].max()) - x0 + 1, int(d['cy'].max()) - y0 + 1
    R = np.zeros((H, W)); Q = np.zeros((H, W)); R[d['cy'] - y0, d['cx'] - x0] = np.minimum(d['zmax'] - d['zmin'], 20.0); Q[d['cy'] - y0, d['cx'] - x0] = 1
    return R, Q, x0, y0


LOOP_IMG = {}


def loop_img(l):
    if l not in LOOP_IMG: LOOP_IMG[l] = relief_img(l)
    return LOOP_IMG[l]


def box_cells(m, xmin, ymin):
    d = fps[m]; x = d['cx'] * 2.0 + 1; y = d['cy'] * 2.0 + 1
    sel = (x >= xmin) & (x < xmin + BOX) & (y >= ymin) & (y < ymin + BOX)
    return x[sel], y[sel], np.minimum(d['zmax'][sel] - d['zmin'][sel], 20.0), int(d['n'][sel].sum())


def template(x, y, rel, yaw):
    xy = np.c_[x, y] @ rot(yaw).T
    ix = np.floor(xy[:, 0] / CELL).astype(int); iy = np.floor(xy[:, 1] / CELL).astype(int); ox, oy = ix.min(), iy.min(); ix -= ox; iy -= oy
    T = np.zeros((iy.max() + 1, ix.max() + 1)); M = np.zeros_like(T); np.maximum.at(T, (iy, ix), rel); M[iy, ix] = 1
    return T, M, ox, oy


def ncc_search(R, Q, x, y, rel, yaws, min_ov):
    best = (-2, None)
    for yaw in yaws:
        T, M, ox, oy = template(x, y, rel, yaw)
        cc, n = mncc(R, Q, T, M, min_ov)
        iy, ix = np.unravel_index(np.argmax(cc), cc.shape)
        if cc[iy, ix] > best[0]:
            best = (float(cc[iy, ix]), (float(yaw), ix, iy, T.shape, ox, oy, cc))
    return best


def verify(m, l, xmin, ymin, yaw0):
    """template = 2023 box relief; image = whole loop relief. returns dict or None"""
    x, y, rel, n23 = box_cells(m, xmin, ymin)
    if len(x) < 0.3 * NB * NB: return None
    R, Q, lx0, ly0 = loop_img(l)
    min_ov = int(0.5 * len(x))
    sc, (yaw, ix, iy, shp, ox, oy, cc) = ncc_search(R, Q, x, y, rel, np.arange(yaw0 - 12, yaw0 + 12.1, 1.0), min_ov)
    sc, (yaw, ix, iy, shp, ox, oy, cc) = ncc_search(R, Q, x, y, rel, np.arange(yaw - 1, yaw + 1.01, 0.25), min_ov)
    flat = cc.copy(); r = 30; flat[max(0, iy - r):iy + r, max(0, ix - r):ix + r] = -1; second = float(flat.max())
    # template origin (rotated-box coords ox*CELL, oy*CELL) sits at loop image cell (ix-(Tw-1), iy-(Th-1))
    Tw, Th = shp[1], shp[0]
    t = np.array([(ix - (Tw - 1) + lx0 - ox) * CELL, (iy - (Th - 1) + ly0 - oy) * CELL])
    ctr_map = np.array([xmin + BOX / 2, ymin + BOX / 2]); ctr_loop = rot(yaw) @ ctr_map + t
    corners = np.array([[xmin, ymin], [xmin + BOX, ymin], [xmin, ymin + BOX], [xmin + BOX, ymin + BOX]]) @ rot(yaw).T + t
    return {'score': round(sc, 3), 'second': round(second, 3), 'yaw_deg': yaw, 't': t.round(2).tolist(), 'centre_loop': ctr_loop.round(1).tolist(),
            'loop_bbox': [corners.min(0).round(1).tolist(), corners.max(0).round(1).tolist()], 'n2023': n23, 'template': (x, y, rel)}


def tt_ncc(a, b):
    """max masked NCC between two box templates over all yaws (duplicate test)"""
    xa, ya, ra = a; xb, yb, rb = b
    Ta, Ma, _, _ = template(xa, ya, ra, 0.0)
    P = 45; R = np.pad(Ta, P); Q = np.pad(Ma, P)
    best = -1
    for yaw in np.arange(0, 360, 5.0):
        T, M, _, _ = template(xb, yb, rb, yaw)
        cc, n = mncc(R, Q, T, M, int(0.4 * min(Ma.sum(), M.sum())))
        best = max(best, float(cc.max()))
    return best


# existing pair templates (raw 2023 crops -> 2 m relief cells), for the duplicate test
from align import load, level
EXIST = {}
for pr in sorted(loc):
    d = f"E:/Subiaco Change Detection point cloud data/Pairs/{pr}"
    f = [f for f in os.listdir(d) if '2023' in f and 'Cloud' in f][0]
    P = load(f'{d}/{f}'); P = P - P.mean(0); Pl = level(P)[0]
    ix = np.floor(Pl[:, 0] / CELL).astype(int); iy = np.floor(Pl[:, 1] / CELL).astype(int)
    key = (ix - ix.min()) * 100000 + (iy - iy.min()); u, inv = np.unique(key, return_inverse=True)
    zmin = np.full(len(u), np.inf); zmax = np.full(len(u), -np.inf); np.minimum.at(zmin, inv, Pl[:, 2]); np.maximum.at(zmax, inv, Pl[:, 2])
    cx = (u // 100000 + ix.min()) * CELL + 1; cy = (u % 100000 + iy.min()) * CELL + 1
    EXIST[pr] = (cx, cy, np.minimum(zmax - zmin, 20.0))
print('existing templates ready')

PW = {(o['map'], o['loop']): o for o in pw}


def same_place_dist(a, b):
    """smallest distance (m) between two boxes evaluated in every frame they can both be expressed in"""
    ds = []
    if a['map'] == b['map']: ds.append(np.hypot(*(a['pm'] - b['pm'])))
    if a['loop'] == b['loop']: ds.append(np.hypot(*(a['pl'] - b['pl'])))
    for x, y in ((a, b), (b, a)):                       # x's map frame <-> y's loop frame via a pairwise fit:  p_map = R p_loop + t
        o = PW.get((x['map'], y['loop']))
        if o and (o['score'] >= 0.5 or 'anchor' in o):
            ds.append(np.hypot(*(x['pm'] - (rot(o['yaw_deg']) @ y['pl'] + np.array(o['t'])))))
    return min(ds) if ds else 9999.0


# candidate generation
cands = []
for o in combos:
    m, l = o['map'], o['loop']; yaw, t = o['yaw_deg'], np.array(o['t'])
    dm = fps[m]; x0, y0 = int(dm['cx'].min()), int(dm['cy'].min()); W, H = int(dm['cx'].max()) - x0 + 1, int(dm['cy'].max()) - y0 + 1
    occm = np.zeros((H, W), bool); occm[dm['cy'] - y0, dm['cx'] - x0] = True
    cntm = np.zeros((H, W)); cntm[dm['cy'] - y0, dm['cx'] - x0] = dm['n']
    dl = fps[l]; pl = np.c_[dl['cx'] * 2.0 + 1, dl['cy'] * 2.0 + 1] @ rot(yaw).T + t
    lx = np.floor(pl[:, 0] / CELL).astype(int) - x0; ly = np.floor(pl[:, 1] / CELL).astype(int) - y0
    ok = (lx >= 0) & (lx < W) & (ly >= 0) & (ly < H)
    occl = np.zeros((H, W), bool); occl[ly[ok], lx[ok]] = True
    cntl = np.zeros((H, W)); np.add.at(cntl, (ly[ok], lx[ok]), dl['n'][ok])
    # relief (zmax - zmin per 2 m cell) grids for the ghost-layer test: a clean box has many near-flat cells (roads, open ground)
    relm = np.full((H, W), np.nan); relm[dm['cy'] - y0, dm['cx'] - x0] = dm['zmax'] - dm['zmin']
    rell = np.full((H, W), np.inf); np.minimum.at(rell, (ly[ok], lx[ok]), (dl['zmax'] - dl['zmin'])[ok]); rell[~np.isfinite(rell)] = np.nan
    def integ(g): I = np.zeros((H + 1, W + 1)); I[1:, 1:] = np.cumsum(np.cumsum(g, 0), 1); return I
    Ij, Im, Il = integ((occm & occl).astype(float)), integ(cntm), integ(cntl)
    for yc in range(0, H - NB + 1, STEP):
        for xc in range(0, W - NB + 1, STEP):
            bs = lambda I: I[yc + NB, xc + NB] - I[yc, xc + NB] - I[yc + NB, xc] + I[yc, xc]
            J = bs(Ij) / (NB * NB)
            if J < MIN_JOINT: continue
            nm, nl = bs(Im), bs(Il)
            if nm < MIN_PTS or nl < MIN_PTS: continue
            rm_ = relm[yc:yc + NB, xc:xc + NB]; rl_ = rell[yc:yc + NB, xc:xc + NB]
            qm = np.nanpercentile(rm_, [10, 25]); ql = np.nanpercentile(rl_, [10, 25])
            if qm[0] > GHOST_Q10 or qm[1] > GHOST_Q25 or ql[0] > GHOST_Q10 or ql[1] > GHOST_Q25: continue
            cands.append({'map': m, 'loop': l, 'xmin': (xc + x0) * CELL, 'ymin': (yc + y0) * CELL, 'joint': float(J), 'n_map': int(nm), 'n_loop_pred': int(nl), 'yaw0': yaw, 'relief_q10_q25_2023': qm.round(2).tolist(), 'relief_q10_q25_2025': ql.round(2).tolist()})
print(f'{len(cands)} candidates')
cands.sort(key=lambda c: -c['joint'])

sel = []; tried = 0; t0 = time.time()
per_loop = {}; per_map = {}
for c in cands:
    if len(sel) >= N_NEW: break
    m, l = c['map'], c['loop']
    if per_loop.get(l, 0) >= MAX_PER_LOOP or per_map.get(m, 0) >= MAX_PER_MAP: continue
    cx, cy = c['xmin'] + BOX / 2, c['ymin'] + BOX / 2
    if any(s['map'] == m and np.hypot(cx - s['cx'], cy - s['cy']) < SEP for s in sel): continue
    if any(e['map2023'] == m and np.hypot(cx - e['centre_new_frame'][0], cy - e['centre_new_frame'][1]) < max(e['half_size_m']) * 1.1 + BOX / 2 + 6 for e in loc.values()): continue
    tried += 1
    v = verify(m, l, c['xmin'], c['ymin'], -c['yaw0'])
    if v is None: continue
    t0_ = np.array(pw[[i for i, o in enumerate(pw) if o['map'] == m and o['loop'] == l][0]]['t'])
    pred = rot(c['yaw0']).T @ (np.array([cx, cy]) - t0_)          # predicted box centre in the loop frame
    far = float(np.hypot(*(np.array(v['centre_loop']) - pred)))
    ok = v['score'] >= MIN_SCORE and v['score'] - v['second'] >= MIN_MARGIN and far < 150
    here = {'map': m, 'loop': l, 'pm': np.array([cx, cy]), 'pl': np.array(v['centre_loop'])}
    others = [{'map': e.get('map', e.get('map2023')), 'loop': _l25[p]['map'], 'pm': np.array(e['centre_new_frame']), 'pl': np.array(_l25[p]['centre_new_frame'])}
              for p, e in loc.items()] + [{'map': s['map'], 'loop': s['loop'], 'pm': np.array([s['cx'], s['cy']]), 'pl': np.array(s['centre_loop'])} for s in sel]
    dup = min([same_place_dist(here, o) for o in others] + [9999.0]) if ok else 9999.0
    status = 'ACCEPT' if ok and dup >= SEP else ('DUPLICATE' if ok else 'REJECT')
    print(f"[{tried}] {m[6:]}+{l} box({c['xmin']:.0f},{c['ymin']:.0f}) joint={c['joint']:.2f} -> score {v['score']:.2f}/{v['second']:.2f} yaw {v['yaw_deg']:.2f} "
          f"pred-err {far:.0f} m nearest-other {dup:.0f} m {status}  [{time.time()-t0:.0f}s]", flush=True)
    if status != 'ACCEPT': continue
    per_loop[l] = per_loop.get(l, 0) + 1; per_map[m] = per_map.get(m, 0) + 1
    s = {'pair': f'Pair{9 + len(sel)}', 'map': m, 'loop': l, 'cx': cx, 'cy': cy, 'xmin': c['xmin'], 'xmax': c['xmin'] + BOX, 'ymin': c['ymin'], 'ymax': c['ymin'] + BOX,
         'joint': c['joint'], 'n2023': v['n2023'], 'n2025_pred': c['n_loop_pred'], 'yaw_deg': v['yaw_deg'], 't': v['t'], 'centre_loop': v['centre_loop'],
         'loop_bbox': v['loop_bbox'], 'score': v['score'], 'second': v['second'], 'pred_err_m': round(far, 1), 'dup_ncc': round(dup, 3),
         'relief_q10_q25_2023': c['relief_q10_q25_2023'], 'relief_q10_q25_2025': c['relief_q10_q25_2025'],
         'src2023': glob.glob(f"{SRC23}/{m[6:19]}*.ply")[0], 'src2025': f"{SRC25}/{l}.ply", 'time2025': LOOP_TIME[int(l[4:])], 'template': v['template']}
    sel.append(s)
for s in sel: s.pop('template')
json.dump(sel, open('selected.json', 'w'), indent=1)
print(f'selected {len(sel)} boxes after {tried} verifications')
