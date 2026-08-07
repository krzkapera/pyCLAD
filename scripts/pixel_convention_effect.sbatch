#!/bin/bash
#SBATCH --job-name=ucad-convention
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

# The account and partition come from the submission, so one script serves both clusters:
#   Athena  sbatch -A plglscclass26-gpu-a100 -p plgrid-gpu-a100 ...
#   Ares    sbatch -A plglscclass26-gpu      -p plgrid-gpu-v100 ...
# The configuration arrives through --export, e.g.
#   --export=ALL,UCAD_NAME=arm_refmasks_s0,UCAD_SEED=0,UCAD_MASKS_DIR=$SCRATCH/pp/data/visa-sam2-grid14
# See scripts/pixel_convention_effect.py for the variables it reads.

set -uo pipefail

BASE=$SCRATCH/pp
cd "$BASE"
RUN_DIR=$BASE/runs/convention_${UCAD_NAME}_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"

module load GCCcore/13.3.0 Python/3.12.3 2>/dev/null || module load python/3.12.3-gcccore-13.3.0

export HF_HOME=$BASE/cache/hf
export TORCH_HOME=$BASE/cache/torch
export XDG_CACHE_HOME=$BASE/cache/xdg
export TMPDIR=$BASE/cache/tmp
export HF_HUB_OFFLINE=1
export PYTHONPATH=$BASE/code/pyCLAD/src
export PYTHONUNBUFFERED=1
export MVTEC_ROOT=$BASE/data/mvtec2d
export MVTEC_MASKS_ROOT=$BASE/data/mvtec2d-sam-b
export VISA_ROOT=$BASE/data/visa
export VISA_MASKS_ROOT=$BASE/data/visa-sam2
export OMP_NUM_THREADS=2

$BASE/.venv/bin/python $BASE/code/pyCLAD/scripts/pixel_convention_effect.py 2>&1 |
    tee "$RUN_DIR/${UCAD_NAME}.log" |
    grep --line-buffered -aE "CONVENTION|Traceback|CUDA error|MemoryError" |
    sed "s/^/[$UCAD_NAME] /"
STATUS=${PIPESTATUS[0]}

echo "===CONVENTION_DONE=== $UCAD_NAME exit=$STATUS dir=$RUN_DIR"
