"""Old vs new materialize inner loop: bit-identity + speedup, on a real store block.

The new loop drops the per-row lexsort.  That is only correct if col_map_global is
monotone over its valid entries -- asserted directly here against the real recipe.
"""
import sys, time
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import os
import numpy as np, h5py, json

STORE = Path(os.path.join(os.environ.get('EXPERT_STORE_DIR', 'data/store'), os.environ.get('EXPERT_STORE_HASH', '27dab279946b4ea6')))
OUT = Path(os.path.join(os.environ.get('EXPERT_OUTPUT_DIR', 'data/store_out'), os.environ.get('EXPERT_DB_HASH', '963044872df9d6a8')))
ds = sys.argv[1] if len(sys.argv) > 1 else 'ReplogleWeissman2022_rpe1'

from _requires import requires  # noqa: E402
requires(STORE / 'cells' / f'{ds}.h5ad', OUT / 'recipe.npz',
         what='a canonical store block and a materialised DB recipe')

col_idx = np.asarray(np.load(OUT / 'recipe.npz', allow_pickle=False)['col_idx'],
                     dtype=np.int64)
print(f'recipe: {len(col_idx):,} output columns')
assert np.all(np.diff(col_idx) > 0), 'col_idx NOT strictly ascending -- sort IS needed'
print('  col_idx strictly ascending -> col_map_global monotone over valid entries  OK')

with h5py.File(str(STORE / 'cells' / f'{ds}.h5ad')) as f:
    X = f['X']; n_obs, n_var = (int(v) for v in X.attrs['shape'])
    src_indptr = X['indptr'][:]
    LIMIT = 120_000
    a, b = 0, int(src_indptr[min(LIMIT, n_obs)])
    d = np.asarray(X['data'][a:b]); idx = np.asarray(X['indices'][a:b])
n_rows = min(LIMIT, n_obs)
src_indptr = src_indptr[:n_rows + 1]
col_map = np.full(n_var, -1, dtype=np.int32)
col_map[col_idx] = np.arange(len(col_idx), dtype=np.int32)
print(f'block {ds}: {n_rows:,} rows, {b:,} nnz')

rng = np.random.default_rng(0)
route = {'pert': rng.random(n_rows) < 0.90, 'ctrl': rng.random(n_rows) < 0.05,
         'zs': rng.random(n_rows) < 0.035}
ROW_CHUNK = 20_000

def old(s, e):
    a, b = int(src_indptr[s]), int(src_indptr[e])
    lengths = (src_indptr[s+1:e+1] - src_indptr[s:e]).astype(np.intp)
    dc, ic = d[a:b], idx[a:b]
    row_of = np.repeat(np.arange(e-s, dtype=np.int64), lengths)
    mapped = col_map[ic]
    out = {}
    for key in ('pert','ctrl','zs'):
        sel = route[key][s:e]; nk = int(sel.sum())
        if nk == 0: continue
        keep = (mapped >= 0) & sel[row_of]
        dd, mm, rr = dc[keep], mapped[keep], row_of[keep]
        local = (np.cumsum(sel)-1)[rr] if dd.size else np.zeros(0, np.int64)
        rn = np.bincount(local, minlength=nk) if dd.size else np.zeros(nk, np.int64)
        order = np.lexsort((mm, local)) if dd.size else np.zeros(0, np.intp)
        out[key] = (dd[order], mm[order], rn)
    return out

def new(s, e):
    a, b = int(src_indptr[s]), int(src_indptr[e])
    lengths = (src_indptr[s+1:e+1] - src_indptr[s:e]).astype(np.intp)
    dc, ic = d[a:b], idx[a:b]
    mapped = col_map[ic]
    valid = mapped >= 0
    cv = np.empty(valid.size+1, dtype=np.int64); cv[0] = 0
    np.cumsum(valid, out=cv[1:])
    bounds = src_indptr[s:e+1] - a
    valid_len = cv[bounds[1:]] - cv[bounds[:-1]]
    out = {}
    for key in ('pert','ctrl','zs'):
        sel = route[key][s:e]; nk = int(sel.sum())
        if nk == 0: continue
        rn = valid_len[sel]
        keep = valid & np.repeat(sel, lengths)
        out[key] = (dc[keep], mapped[keep], rn)
    return out

t_old = t_new = 0.0
for s in range(0, n_rows, ROW_CHUNK):
    e = min(s+ROW_CHUNK, n_rows)
    t0=time.time(); o=old(s,e); t_old+=time.time()-t0
    t0=time.time(); n=new(s,e); t_new+=time.time()-t0
    assert o.keys()==n.keys(), f'chunk {s}: route keys differ'
    for k in o:
        for j,nm in enumerate(('data','indices','row_nnz')):
            assert np.array_equal(o[k][j], n[k][j]), f'chunk {s} route {k}: {nm} DIFFERS'
print(f'\n  bit-identical across all {n_rows//ROW_CHUNK+1} chunks x 3 routes  OK')
print(f'\n  old routing {t_old:7.1f}s\n  new routing {t_new:7.1f}s   -> {t_old/t_new:.2f}x')
