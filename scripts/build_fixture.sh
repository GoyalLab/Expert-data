#!/bin/bash
#SBATCH --job-name ExPert-fixture
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 48GB
#SBATCH --time 8:00:00
#SBATCH --output logs/fixture/%x-%j.log

# Build the row-subset fixtures.  Login nodes have a 6 GB cgroup cap, so this must
# run on a compute node.  Pass --dry-run to only report selection counts.
#
#   sbatch scripts/build_fixture.sh --dry-run
#   sbatch scripts/build_fixture.sh

mkdir -p logs/fixture

# NOTE: no `set -u` -- /etc/bashrc references an unbound BASHRCSOURCED and would abort.
set -eo pipefail

module purge
source ~/.bashrc
module load mamba
mamba activate "${EXPERT_ENV:-ExPert}"
set -eo pipefail

echo "Host: $(hostname)  Args: $*"
python scripts/build_fixture.py "$@"
echo "Done."
