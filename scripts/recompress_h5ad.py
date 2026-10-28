"""Recompress an h5ad file via in-memory load + write_h5ad.

Requires enough RAM to hold the full dataset in memory.

Usage:
    python scripts/recompress_h5ad.py input.h5ad output.h5ad
    python scripts/recompress_h5ad.py input.h5ad  # overwrites in place
"""
import argparse
import anndata as ad


def main():
    parser = argparse.ArgumentParser(description='Recompress h5ad file')
    parser.add_argument('input', help='Input h5ad file')
    parser.add_argument('output', nargs='?', default=None,
                        help='Output path (default: overwrite input)')
    args = parser.parse_args()

    out = args.output or args.input
    print(f'Reading {args.input}...')
    adata = ad.read_h5ad(args.input)
    print(f'Loaded: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes, '
          f'nnz={adata.X.nnz:,}')
    print(f'Writing {out}...')
    adata.write_h5ad(out, compression='gzip')
    print('Done.')


if __name__ == '__main__':
    main()
