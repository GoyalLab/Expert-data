"""Integrity check for a materialised DB: row order, zero-padding, feature mask."""
import sys
import h5py, numpy as np

O = sys.argv[1]
z = np.load(O + '/recipe.npz', allow_pickle=False)
cm = {k[len('colmask__'):]: z[k] for k in z.files if k.startswith('colmask__')}


def col(f, name):
    d = f['obs'][name]
    if isinstance(d, h5py.Group):
        return np.asarray(d['categories']).astype(str)[np.asarray(d['codes'])]
    a = np.asarray(d)
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


with h5py.File(O + '/database_perturbed.h5ad') as f:
    dsv = col(f, 'dataset')
    ip = np.asarray(f['X/indptr']); idx = f['X/indices']
    fm = np.asarray(f['varm/feature_mask_per_dataset'])
    d2i = {k: int(np.asarray(f['uns/dataset_to_mask_idx'][k]))
           for k in f['uns/dataset_to_mask_idx']}
    rng = np.random.default_rng(0); bad = 0; n = 0
    for ds in sorted(set(dsv)):
        rows = np.flatnonzero(dsv == ds)
        for rr in np.sort(rng.choice(rows, min(50, rows.size), replace=False)):
            a, b = int(ip[rr]), int(ip[rr + 1])
            if b > a and not cm[ds][np.asarray(idx[a:b])].all():
                bad += 1
            n += 1
    first = [dsv[0]]
    for x, y in zip(dsv[:-1], dsv[1:]):
        if x != y:
            first.append(y)
    print('  blocks:', len(first), ' order == sorted():', first == sorted(set(dsv)))
    print(f'  zero-padding: {n:,} rows sampled, {bad} violations',
          'PASS' if bad == 0 else 'FAIL')
    print('  feature_mask == col_masks:',
          all(np.array_equal(fm[:, j] > 0, cm[k]) for k, j in d2i.items()))
    print('  compression:', f['X/data'].compression, ' dtype:', f['X/data'].dtype)
