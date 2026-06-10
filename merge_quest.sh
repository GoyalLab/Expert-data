#!/bin/bash
#SBATCH --account b1042
#SBATCH --partition genomicslong
#SBATCH --job-name ExPert-merge
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem 10GB
#SBATCH --time 10-00:00:00
#SBATCH --output logs/merge/%j/snakemake.log
#SBATCH --verbose
#SBATCH --mail-user=leon.schwartz@northwestern.edu
#SBATCH --mail-type=END,FAIL

echo "Setting up pipeline environment..."
module purge
source ~/.bashrc
module load mamba
mamba activate ExPert
echo "Starting pipeline..."

snakemake \
    --cores 1 \
    --verbose \
    --configfile "config/merge.yaml" \
    --profile "workflow/profiles/slurm" \
    --use-conda

echo "Finished pipeline"
conda deactivate
