"""Skip guard for the tests that need real data.

Most of these are validation gates against a canonical store block or a materialised
DB -- neither ships with the repo, and neither can be faked: the whole point of e.g.
test_materialize_fastpath is bit-identity against a REAL block.  Without the data the
honest outcome is SKIP, not a NameError three frames deep.

Exit code stays 0 so a `for f in tests/*.py` sweep on a fresh clone is green.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def requires(*paths, what: str = 'test data') -> None:
    """Exit 0 with a SKIP line if any of `paths` is absent."""
    missing = [str(p) for p in paths if p is not None and not Path(p).exists()]
    if missing:
        print(f'SKIP {Path(sys.argv[0]).name}: needs {what}; not found:')
        for m in missing:
            print(f'  {m}')
        print('  set the EXPERT_* paths (see config/site.env.example) or pass the paths '
              'as arguments')
        raise SystemExit(0)


def requires_env(*names: str, what: str = 'test data') -> None:
    """Exit 0 with a SKIP line if any of the env vars is unset."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        print(f'SKIP {Path(sys.argv[0]).name}: needs {what}; unset: {", ".join(missing)}')
        raise SystemExit(0)
