"""Check the keep-all DB: X is NOT masked, and the panel rides along in varm."""
import sys
from pathlib import Path
import h5py, numpy as np

if len(sys.argv) < 2:
    raise SystemExit(f'usage: {sys.argv[0]} <materialised DB dir>')
O = sys.argv[1]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _requires import requires  # noqa: E402
requires(O + '/recipe.npz', what='a materialised DB')
z = np.load(O + '/recipe.npz', allow_pickle=False)
cm = {k[len('colmask__'):]: z[k] for k in z.files if k.startswith('colmask__')}
print('  recipe zero_padded flag:', bool(z['zero_padded'][0]) if 'zero_padded' in z.files else 'absent')


def col(f, name):
    d = f['obs'][name]
    if isinstance(d, h5py.Group):
        return np.asarray(d['categories']).astype(str)[np.asarray(d['codes'])]
    a = np.asarray(d)
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


with h5py.File(O + '/database_perturbed.h5ad') as f:
    print('  varm layers:', list(f['varm'].keys()))
    dsv = col(f, 'dataset'); ip = np.asarray(f['X/indptr']); idx = f['X/indices']
    n_genes = int(f['X'].attrs['shape'][1])
    fm = np.asarray(f['varm/feature_mask_per_dataset'])
    pm = (np.asarray(f['varm/panel_membership_per_dataset'])
          if 'panel_membership_per_dataset' in f['varm'] else None)
    d2i = {k: int(np.asarray(f['uns/dataset_to_mask_idx'][k]))
           for k in f['uns/dataset_to_mask_idx']}
    rng = np.random.default_rng(0)
    print(f'\n  {"dataset":34s} {"cols seen":>10s} {"panel":>7s} {"measured":>9s}')
    beyond = 0
    for ds in sorted(set(dsv)):
        rows = np.flatnonzero(dsv == ds)
        take = np.sort(rng.choice(rows, min(400, rows.size), replace=False))
        seen = set()
        for rr in take:
            a, b = int(ip[rr]), int(ip[rr + 1])
            if b > a:
                seen.update(np.asarray(idx[a:b]).tolist())
        n_panel = int(cm[ds].sum())
        n_meas = int((fm[:, d2i[ds]] > 0).sum())
        outside = len(seen - set(np.flatnonzero(cm[ds]).tolist()))
        beyond += outside
        print(f'  {ds:34s} {len(seen):>10,} {n_panel:>7,} {n_meas:>9,}'
              f'   outside panel: {outside:,}')
    print(f'\n  genes seen OUTSIDE the panel: {beyond:,}  '
          f'{"GOOD - X is unmasked" if beyond > 0 else "BAD - X still masked"}')
    if pm is not None:
        ok = all(np.array_equal(pm[:, j] > 0, cm[k]) for k, j in d2i.items())
        print(f'  varm/panel_membership == recipe col_masks: {ok}')
        print(f'  panel density {(pm > 0).mean():.1%}   feature_mask density {(fm > 0).mean():.1%}')
