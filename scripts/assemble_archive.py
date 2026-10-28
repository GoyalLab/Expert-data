#!/usr/bin/env python
"""Turn an archive of per-sample matrices into the single .h5ad the pipeline expects.

Some sources ship a tar of CellRanger outputs rather than one assembled object -- GEO
GSE295214 (Zhuang2026) is 85 `filtered_feature_bc_matrix.h5` files.  `download_dataset`
can only fetch bytes, so without this step those datasets have to be assembled by hand,
which is exactly the provenance gap that made them unreproducible after the raw cache
was purged (3 of 22 datasets had no download provenance at all).

Per member:
    Gene Expression      -> X, keyed on Ensembl id (the canonical store keys on Ensembl,
                            not symbol: 313 ids carry conflicting symbols)
    CRISPR Guide Capture -> collapsed into obs columns (guide, target, guide_umi,
                            guide_frac, n_guides), NOT a second matrix, so the output
                            stays a plain counts AnnData like every other raw input

Guide assignment is a CHOICE, not a recovered ground truth: a cell is called for its
dominant guide when that guide has >= --min-guide-umi UMIs and >= --min-guide-frac of
the cell's guide counts, else `unassigned`.  Targets are parsed as the leading field of
the guide id (`GTF3C5_-_135906397.23-P1P2` -> GTF3C5); --ntc-pattern names the
non-targeting guides, which become the control label.  Different thresholds give
different labels, so the chosen values are written to uns/assembly for the record.

Members are extracted and discarded one at a time, and samples are concatenated on disk,
so peak memory is one sample rather than the whole dataset.

    python scripts/assemble_archive.py --archive raw/Zhuang2026.tar --out raw/Zhuang2026.h5ad
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MATRIX_SUFFIXES = ('.h5', '.h5ad')
GEX, GUIDE = 'Gene Expression', 'CRISPR Guide Capture'


def _members(archive: Path) -> list[str]:
    if archive.suffix == '.zip':
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
    else:
        with tarfile.open(archive, 'r:*') as t:
            names = [m.name for m in t.getmembers() if m.isfile()]
    keep = [n for n in names if n.endswith(MATRIX_SUFFIXES)
            and not Path(n).name.startswith('.')]
    if not keep:
        raise SystemExit(f'{archive}: no .h5/.h5ad members found')
    return sorted(keep)


def _extract(archive: Path, member: str, dest: Path) -> Path:
    if archive.suffix == '.zip':
        with zipfile.ZipFile(archive) as z, open(dest / Path(member).name, 'wb') as f:
            shutil.copyfileobj(z.open(member), f)
    else:
        with tarfile.open(archive, 'r:*') as t:
            src = t.extractfile(member)
            if src is None:
                raise SystemExit(f'{archive}: cannot read member {member}')
            with open(dest / Path(member).name, 'wb') as f:
                shutil.copyfileobj(src, f, length=32 * 1024 * 1024)
    return dest / Path(member).name


def _sample_name(member: str) -> str:
    """`GSM8943713_jan_10_GEX_A7_sg_H6_filtered_feature_bc_matrix.h5` -> the informative
    prefix, so the sample stays traceable to its GEO accession."""
    stem = Path(member).name
    for tail in ('_filtered_feature_bc_matrix.h5', '_raw_feature_bc_matrix.h5',
                 '.h5ad', '.h5'):
        if stem.endswith(tail):
            stem = stem[: -len(tail)]
            break
    return stem


def _assign_guides(guide_X, guide_ids, ntc_re, min_umi, min_frac):
    """Dominant-guide call per cell -> (guide, target, umi, frac, n_detected)."""
    import numpy as np
    import scipy.sparse as sp

    G = sp.csr_matrix(guide_X)
    total = np.asarray(G.sum(1)).ravel()
    n_det = np.asarray((G > 0).sum(1)).ravel()
    top_i = np.zeros(G.shape[0], dtype=np.int64)
    top_v = np.zeros(G.shape[0], dtype=np.float64)
    for r in range(G.shape[0]):
        s, e = G.indptr[r], G.indptr[r + 1]
        if e > s:
            k = s + int(np.argmax(G.data[s:e]))
            top_i[r], top_v[r] = G.indices[k], G.data[k]
    frac = np.divide(top_v, total, out=np.zeros_like(top_v), where=total > 0)
    ok = (top_v >= min_umi) & (frac >= min_frac) & (total > 0)
    guide = np.where(ok, guide_ids[top_i], 'unassigned').astype(object)
    target = np.array([g.split('_')[0] if g != 'unassigned' else 'unassigned'
                       for g in guide], dtype=object)
    is_ntc = np.array([bool(ntc_re.search(g)) for g in guide])
    target[is_ntc] = 'control'
    return guide, target, top_v, frac, n_det, total


def _read_member(path: Path, sample: str, ntc_re, min_umi, min_frac):
    import anndata as ad
    import numpy as np
    import scanpy as sc

    if path.suffix == '.h5ad':
        a = ad.read_h5ad(path)
    else:
        a = sc.read_10x_h5(path, gex_only=False)
    a.var_names_make_unique()
    ftype = a.var['feature_types'].astype(str).to_numpy() if 'feature_types' in a.var \
        else np.full(a.n_vars, GEX)

    gmask = ftype == GUIDE
    if gmask.any():
        gids = a.var_names.to_numpy()[gmask].astype(str)
        guide, target, umi, frac, n_det, tot = _assign_guides(
            a[:, gmask].X, gids, ntc_re, min_umi, min_frac)
    else:
        n = a.n_obs
        guide = target = np.full(n, 'unassigned', dtype=object)
        umi = frac = np.zeros(n)
        n_det = tot = np.zeros(n, dtype=int)

    xmask = ftype == GEX if (ftype == GEX).any() else ~gmask
    g = a[:, xmask].copy()
    # Ensembl ids as the axis; symbols kept alongside (canonical-store keying)
    if 'gene_ids' in g.var:
        g.var['gene_symbol'] = g.var_names.astype(str)
        g.var_names = g.var['gene_ids'].astype(str)
    g.var_names_make_unique()
    g.obs_names = [f'{sample}_{b}' for b in g.obs_names]
    g.obs['sample'] = sample
    g.obs['guide'] = guide
    g.obs['target'] = target
    g.obs['guide_umi'] = umi.astype('float32')
    g.obs['guide_frac'] = frac.astype('float32')
    g.obs['n_guides'] = n_det.astype('int32')
    g.obs['guide_total'] = np.asarray(tot, dtype='float32')
    for c in ('sample', 'guide', 'target'):
        g.obs[c] = g.obs[c].astype(str).astype('category')
    g.var = g.var.drop(columns=[c for c in ('feature_types', 'genome') if c in g.var])
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tmp-dir', default=None, help='staging dir (default: beside --out)')
    ap.add_argument('--min-guide-umi', type=float, default=3)
    ap.add_argument('--min-guide-frac', type=float, default=0.5)
    ap.add_argument('--ntc-pattern', default=r'(?i)non.?target|^NTC|safe.?harbou?r|AAVS1')
    ap.add_argument('--max-samples', type=int, default=0, help='>0 = smoke test')
    ap.add_argument('--keep-samples', action='store_true',
                    help='keep per-sample h5ads instead of deleting them')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(message)s')
    import anndata as ad
    from anndata.experimental import concat_on_disk

    archive, out = Path(args.archive), Path(args.out)
    ntc_re = re.compile(args.ntc_pattern)
    members = _members(archive)
    if args.max_samples:
        members = members[: args.max_samples]
    logging.info(f'{archive.name}: {len(members)} member(s) -> {out}')

    stage = Path(args.tmp_dir) if args.tmp_dir else out.parent / f'.{out.stem}.assemble'
    stage.mkdir(parents=True, exist_ok=True)
    parts, n_cells, counted = [], 0, {}
    try:
        with tempfile.TemporaryDirectory(dir=str(stage)) as scratch:
            for i, m in enumerate(members, 1):
                sample = _sample_name(m)
                part = stage / f'{sample}.part.h5ad'
                if not part.exists():
                    src = _extract(archive, m, Path(scratch))
                    a = _read_member(src, sample, ntc_re, args.min_guide_umi,
                                     args.min_guide_frac)
                    a.write_h5ad(part)          # uncompressed: concat reads it right back
                    src.unlink(missing_ok=True)
                    counted[sample] = int(a.n_obs)
                    assigned = int((a.obs['target'].astype(str) != 'unassigned').sum())
                    logging.info(f'[{i}/{len(members)}] {sample}: {a.n_obs:,} cells, '
                                 f'{a.n_vars:,} genes, {assigned:,} assigned '
                                 f'({100 * assigned / max(a.n_obs, 1):.1f}%)')
                    del a
                parts.append(str(part))
            out.parent.mkdir(parents=True, exist_ok=True)
            # suffix must stay .h5ad: concat_on_disk dispatches on it (else it tries zarr)
            tmp_out = out.with_name(out.name + '.tmp.h5ad')
            concat_on_disk(parts, str(tmp_out), axis=0, join='outer', label='sample_id',
                           index_unique=None)
            tmp_out.replace(out)
    finally:
        if not args.keep_samples:
            shutil.rmtree(stage, ignore_errors=True)

    a = ad.read_h5ad(out, backed='r')
    n_cells = a.n_obs
    tgt = a.obs['target'].astype(str)
    logging.info(f'wrote {out}: {n_cells:,} cells x {a.n_vars:,} genes | '
                 f'{(tgt != "unassigned").sum():,} assigned, '
                 f'{(tgt == "control").sum():,} control, '
                 f'{tgt.nunique() - 1:,} targets')
    a.file.close()
    meta = {'archive': str(archive), 'members': len(members),
            'min_guide_umi': args.min_guide_umi, 'min_guide_frac': args.min_guide_frac,
            'ntc_pattern': args.ntc_pattern, 'cells_per_sample': counted}
    Path(str(out) + '.assembly.json').write_text(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
