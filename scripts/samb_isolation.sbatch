#!/bin/bash
#SBATCH --job-name=ucad-samb-isolation
#SBATCH --account=plglscclass26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
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
export SAMB_ROOT=$BASE/data/mvtec2d-sam-b
export ISOLATION_CATEGORY=${ISOLATION_CATEGORY:-screw}
export ISOLATION_PATCHSIZE=${ISOLATION_PATCHSIZE:-3}
export OMP_NUM_THREADS=2

LOG="$RUN_DIR/samb_isolation.log"

$BASE/.venv/bin/python $BASE/code/pyCLAD/scripts/ucad_samb_isolation_test.py 2>&1 |
    tee "$LOG" |
    grep --line-buffered -E "EPOCH |RESULT |Task .* added|Traceback|CUDA error|MemoryError"

echo "===TRAINING_DONE==="
