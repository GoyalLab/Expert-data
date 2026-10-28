#!/bin/bash
#SBATCH --job-name ExPert-merge
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem 10GB
#SBATCH --time 48:00:00
#SBATCH --output logs/merge/%j/snakemake.log
#SBATCH --verbose

echo "Setting up pipeline environment..."
module purge
source ~/.bashrc
module load mamba
mamba activate "${EXPERT_ENV:-ExPert}"
echo "Starting pipeline..."

snakemake \
    --cores 1 \
    --verbose \
    --configfile "config/merge.yaml" \
    --profile "workflow/profiles/slurm" \
    --rerun-triggers mtime \
    --use-conda

echo "Finished pipeline"
conda deactivate
