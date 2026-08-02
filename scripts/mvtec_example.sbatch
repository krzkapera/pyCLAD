#!/bin/bash
#SBATCH --job-name=ucad-mvtec
#SBATCH --account=plglscclass26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -uo pipefail

BASE=$SCRATCH/pp
cd "$BASE"
RUN_DIR=$BASE/runs/$SLURM_JOB_ID
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
export MVTEC_MASKS_ROOT=$BASE/data/mvtec2d-sam2
export OMP_NUM_THREADS=2
export UCAD_OUTPUT=$RUN_DIR/ucad_mvtec.json

LOG="$RUN_DIR/mvtec.log"

$BASE/.venv/bin/python $BASE/code/pyCLAD/examples/ucad_mvtec_example.py 2>&1 |
    tee "$LOG" |
    grep --line-buffered -E "Starting (training|evaluation)|Traceback|CUDA error|MemoryError"

echo "===TRAINING_DONE=== $UCAD_OUTPUT"
