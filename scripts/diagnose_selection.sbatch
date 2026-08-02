#!/bin/bash
#SBATCH --job-name=ucad-diag-sel
#SBATCH --account=plglscclass26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

set -uo pipefail

BASE=$SCRATCH/pp
cd "$BASE"
RUN_DIR=$BASE/runs/diag_selection
mkdir -p "$RUN_DIR"

module load GCCcore/13.3.0
module load Python/3.12.3

export HF_HOME=$BASE/cache/hf
export TORCH_HOME=$BASE/cache/torch
export XDG_CACHE_HOME=$BASE/cache/xdg
export TMPDIR=$BASE/cache/tmp
export HF_HUB_OFFLINE=1
export PYTHONPATH=$BASE/code/pyCLAD/src
export PYTHONUNBUFFERED=1
export MVTEC_ROOT=$BASE/data/mvtec2d
export OMP_NUM_THREADS=2

$BASE/.venv/bin/python $BASE/code/pyCLAD/scripts/diagnose_task_selection.py 2>&1 |
    tee "$RUN_DIR/diag_$SLURM_JOB_ID.log" |
    grep --line-buffered -E "SELECT |PODSUMOWANIE|klucz gotowy|Traceback|CUDA error"

echo "===DONE==="
