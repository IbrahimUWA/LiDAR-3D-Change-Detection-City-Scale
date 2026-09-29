"""Stream a huge ASCII PLY (x y z r g b) and build a 2 m occupancy grid: count, zmin, zmax, zsum per cell."""
import sys, time, os
import polars as pl
import numpy as np

CELL = 2.0

def header_len(path):
    n = 0
    with open(path, 'rb') as f:
        for line in f:
            n += 1
            if line.strip() == b'end_header':
                return n
    raise RuntimeError('no end_header')

def scan(path, out_npz):
    t0 = time.time()
    hl = header_len(path)
    lf = (pl.scan_csv(path, separator=' ', has_header=False, skip_rows=hl,
                      new_columns=['x','y','z','r','g','b'],
                      schema_overrides={'x': pl.Float32, 'y': pl.Float32, 'z': pl.Float32,
                                        'r': pl.UInt8, 'g': pl.UInt8, 'b': pl.UInt8},
                      low_memory=True)
          .select([(pl.col('x')/CELL).floor().cast(pl.Int32).alias('cx'),
                   (pl.col('y')/CELL).floor().cast(pl.Int32).alias('cy'),
                   pl.col('z')])
          .group_by(['cx','cy'])
          .agg([pl.len().alias('n'), pl.col('z').min().alias('zmin'),
                pl.col('z').max().alias('zmax'), pl.col('z').sum().alias('zsum')]))
    df = lf.collect(engine='streaming')
    np.savez(out_npz, cx=df['cx'].to_numpy(), cy=df['cy'].to_numpy(), n=df['n'].to_numpy(),
             zmin=df['zmin'].to_numpy(), zmax=df['zmax'].to_numpy(), zsum=df['zsum'].to_numpy(), cell=CELL)
    npts = int(df['n'].sum())
    print(f"{os.path.basename(path)}: {npts:,} pts, {len(df):,} cells, {time.time()-t0:.1f}s", flush=True)

if __name__ == '__main__':
    scan(sys.argv[1], sys.argv[2])
