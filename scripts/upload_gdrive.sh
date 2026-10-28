#!/bin/bash
#SBATCH --job-name upload_database
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 10
#SBATCH --mem 20GB
#SBATCH --time 4:00:00
#SBATCH --output logs/upload_%j.log
#SBATCH --mail-user=leon.schwartz@northwestern.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --verbose

# Specify the run directory
ID='1e6bafc04836e06a'
SRC_BASE="${EXPERT_PUBLISH_DIR:?set EXPERT_PUBLISH_DIR}"
SRC_PATH="${SRC_BASE}/${ID}"
# Drive specifics
DRIVE_BASE='nvidia'
DRIVE_LOC="database/${ID}"

# Activate env
module purge
source ~/.bashrc
module load mamba
mamba activate basic_toolkit

# Upload the split indices
rclone copy "${SRC_PATH}" "${DRIVE_BASE}:${DRIVE_LOC}" -P
echo "done."
