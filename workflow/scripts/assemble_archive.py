"""Snakemake entry point for scripts/assemble_archive.py (see that file for the method)."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import setup_logger  # noqa: E402


def main(archive, out, tmp_dir, min_guide_umi, min_guide_frac, ntc_pattern):
    import runpy

    sys.argv = ['assemble_archive.py', '--archive', str(archive), '--out', str(out),
                '--tmp-dir', str(tmp_dir), '--min-guide-umi', str(min_guide_umi),
                '--min-guide-frac', str(min_guide_frac), '--ntc-pattern', ntc_pattern]
    runpy.run_path(str(REPO_ROOT / 'scripts' / 'assemble_archive.py'),
                   run_name='__main__')


if __name__ == '__main__':
    try:
        setup_logger(snakemake.log[0])
        main(archive=snakemake.input.archive, out=snakemake.output.raw,
             tmp_dir=snakemake.params.tmp_dir,
             min_guide_umi=snakemake.params.min_guide_umi,
             min_guide_frac=snakemake.params.min_guide_frac,
             ntc_pattern=snakemake.params.ntc_pattern)
    except NameError:
        print('This script is meant to be run through Snakemake.')
