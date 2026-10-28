#!/bin/bash
#SBATCH --job-name upload_database
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 10
#SBATCH --mem 20GB
#SBATCH --time 4:00:00
#SBATCH --output logs/upload_zs_%j.log
#SBATCH --verbose

# Specify the run directory
SRC_PATH="${EXPERT_PUBLISH_DIR:?set EXPERT_PUBLISH_DIR}/v1/database_zeroshot.h5ad"
DRUG="${EXPERT_INFERENCE_DIR:?set EXPERT_INFERENCE_DIR}/tahoe100m/tahoe100m_ood_subset.h5ad"
# Drive specifics
DRIVE_BASE='nvidia'
DRIVE_LOC='updated_data/'

# Activate env
module purge
source ~/.bashrc
module load mamba
mamba activate basic_toolkit

set -eo pipefail
# Upload the split indices
rclone copy "${SRC_PATH}" "${DRIVE_BASE}:${DRIVE_LOC}" -P
rclone copy "${DRUG}" "${DRIVE_BASE}:${DRIVE_LOC}" -P
echo "done."
