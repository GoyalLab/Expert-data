"""Per-phase timing of the materialize inner loop on one real store block."""
import sys, time, json
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import numpy as np, h5py, tempfile, os

STORE = Path(os.path.join(os.environ.get('EXPERT_STORE_DIR', 'data/store'), os.environ.get('EXPERT_STORE_HASH', '27dab279946b4ea6')))
ds = sys.argv[1] if len(sys.argv) > 1 else 'ReplogleWeissman2022_rpe1'
xp = STORE / 'cells' / f'{ds}.h5ad'
from _requires import requires  # noqa: E402
requires(xp, what='a canonical store block')
print(f'block: {ds}\n  X: {xp}  ({xp.stat().st_size/1024**3:.2f} GB on disk)')

ROW_CHUNK = 20_000
t_read = t_route = t_write = 0.0
nnz_tot = 0
rows_tot = 0

with h5py.File(str(xp), 'r') as f:
    grp = f['X'] if 'X' in f else f
    dset_d, dset_i, dset_p = grp['data'], grp['indices'], grp['indptr']
    print(f'  filters: data={dset_d.compression} indices={dset_i.compression}'
          f'  chunks={dset_d.chunks}')
    n_obs = dset_p.shape[0] - 1
    n_genes = 38638
    indptr = np.asarray(dset_p)
    rng = np.random.default_rng(0)
    # emulate the three routes: most cells -> pert, some ctrl, few zs
    route = {'pert': rng.random(n_obs) < 0.90,
             'ctrl': rng.random(n_obs) < 0.05,
             'zs':   rng.random(n_obs) < 0.035}
    col_map = np.arange(n_genes, dtype=np.int32)

    out = tempfile.NamedTemporaryFile(suffix='.h5', delete=False); out.close()
    fo = h5py.File(out.name, 'w')
    od = fo.create_dataset('data', (0,), maxshape=(None,), dtype=np.int16,
                           chunks=(1_000_000,), compression='gzip',
                           compression_opts=int(os.environ.get('GZ', '1')))
    oi = fo.create_dataset('indices', (0,), maxshape=(None,), dtype=np.int32,
                           chunks=(1_000_000,), compression='gzip',
                           compression_opts=int(os.environ.get('GZ', '1')))

    LIMIT = 200_000   # cells to profile
    t_all = time.time()
    for s in range(0, min(n_obs, LIMIT), ROW_CHUNK):
        e = min(s + ROW_CHUNK, n_obs, LIMIT)
        a, b = int(indptr[s]), int(indptr[e])
        t0 = time.time()
        d = np.asarray(dset_d[a:b]); idx = np.asarray(dset_i[a:b])
        t_read += time.time() - t0
        nnz_tot += b - a; rows_tot += e - s

        t0 = time.time()
        lengths = np.diff(indptr[s:e+1]).astype(np.intp)
        mapped = col_map[idx]
        valid = mapped >= 0
        cv = np.empty(valid.size + 1, dtype=np.int64); cv[0] = 0
        np.cumsum(valid, out=cv[1:])
        bounds = indptr[s:e+1] - a
        valid_len = cv[bounds[1:]] - cv[bounds[:-1]]
        bufs = []
        for key in ('pert', 'ctrl', 'zs'):
            sel = route[key][s:e]
            if not sel.any(): continue
            row_nnz = valid_len[sel]
            keep = valid & np.repeat(sel, lengths)
            bufs.append((d[keep], mapped[keep]))
        t_route += time.time() - t0

        t0 = time.time()
        for dd, mm in bufs:
            n = od.shape[0]
            od.resize((n + dd.size,)); od[n:] = dd
            oi.resize((n + mm.size,)); oi[n:] = mm
        t_write += time.time() - t0
    tot = time.time() - t_all
    fo.close(); sz = os.path.getsize(out.name); os.unlink(out.name)

raw = nnz_tot * 6 / 1024**2
print(f'\n  profiled {rows_tot:,} cells, {nnz_tot:,} nnz ({raw:.0f} MB raw)')
print(f'\n  {"phase":10s} {"sec":>8s} {"%":>7s} {"MB/s raw":>10s}')
for name, t in (('read', t_read), ('route', t_route), ('write', t_write)):
    print(f'  {name:10s} {t:8.1f} {100*t/tot:6.1f}% {raw/t:10.1f}')
print(f'  {"TOTAL":10s} {tot:8.1f} {100.0:6.1f}% {raw/tot:10.1f}')
print(f'\n  extrapolated to 31.8M cells: {tot * 31.8e6/rows_tot/3600:.1f} h')
