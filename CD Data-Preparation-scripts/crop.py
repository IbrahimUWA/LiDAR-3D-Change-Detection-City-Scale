"""Crop several XY boxes from one huge ASCII PLY in a single streaming pass.
Usage: python crop.py boxes.json    (boxes.json: list of {name, src, xmin, xmax, ymin, ymax, out})"""
import sys, os, json, time
import polars as pl, numpy as np

def header_len(path):
    n = 0
    with open(path, 'rb') as f:
        for line in f:
            n += 1
            if line.strip() == b'end_header': return n

def write_ply(path, df, comments):
    pts = df.select(['x', 'y', 'z']).to_numpy()
    rgb = df.select(['r', 'g', 'b']).to_numpy()
    with open(path, 'w', newline='\n') as f:
        f.write('ply\nformat ascii 1.0\n')
        for c in comments: f.write(f'comment {c}\n')
        f.write(f'element vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n'
                'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        np.savetxt(f, np.c_[pts, rgb], fmt='%.6f %.6f %.6f %d %d %d')

def run(boxes):
    by_src = {}
    for b in boxes: by_src.setdefault(b['src'], []).append(b)
    for src, bl in by_src.items():
        t0 = time.time(); hl = header_len(src)
        lf = pl.scan_csv(src, separator=' ', has_header=False, skip_rows=hl, new_columns=['x','y','z','r','g','b'],
                         schema_overrides={'x': pl.Float64, 'y': pl.Float64, 'z': pl.Float64, 'r': pl.UInt8, 'g': pl.UInt8, 'b': pl.UInt8},
                         low_memory=True)
        preds = [(pl.col('x') >= b['xmin']) & (pl.col('x') < b['xmax']) & (pl.col('y') >= b['ymin']) & (pl.col('y') < b['ymax']) for b in bl]
        anyp = preds[0]
        for p in preds[1:]: anyp = anyp | p
        tag = pl.lit(None, dtype=pl.Int32)
        for i, p in reversed(list(enumerate(preds))):
            tag = pl.when(p).then(pl.lit(i, dtype=pl.Int32)).otherwise(tag)
        df = lf.filter(anyp).with_columns(tag.alias('box')).collect(engine='streaming')
        for i, b in enumerate(bl):
            sub = df.filter(pl.col('box') == i)
            os.makedirs(os.path.dirname(b['out']), exist_ok=True)
            write_ply(b['out'], sub, [f"Subiaco change-detection raw crop {b['name']} (source-map coordinates, uncorrected)",
                                      f"source {os.path.basename(src)}",
                                      f"crop box x[{b['xmin']},{b['xmax']}) y[{b['ymin']},{b['ymax']}) all z",
                                      'Generated 2026-09-27 by crop.py'])
            print(f"{b['name']}: {os.path.basename(src)} -> {len(sub):,} pts -> {b['out']}", flush=True)
        print(f"  pass over {os.path.basename(src)}: {time.time()-t0:.1f}s", flush=True)

if __name__ == '__main__':
    run(json.load(open(sys.argv[1])))
