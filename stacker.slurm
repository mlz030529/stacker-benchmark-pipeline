#!/bin/bash
#SBATCH -J stacker_align
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=256G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
mkdir -p logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}


# required
export STACKER_DATA_DIR="/path/to/simulation"
export STACKER_OUT_DIR="/path/to/simulation/aligned_out_stacker"
export STACKER_WEIGHTS="/path/to/stacker/checkpoints/variables/variables"

export STACKER_PAIRS="cell_metadata_S2R2.csv,cell_metadata_simulation.csv,cell_by_gene_S2R2.csv.gz,cell_by_gene_simulation.csv"

# optional
export STACKER_XCOL="center_x"
export STACKER_YCOL="center_y"
export STACKER_MAX_REG_DIM=512
export STACKER_DENSITY_SIZE=512
export STACKER_SIGMA_K=5

python run_stacker_pairs.py
