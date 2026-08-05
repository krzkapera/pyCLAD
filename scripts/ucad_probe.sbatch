#!/bin/bash
#SBATCH --job-name=ucad-probe
#SBATCH --account=plglscclass26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

# One configuration per submission; the configuration itself arrives through --export, e.g.
#   sbatch --export=ALL,UCAD_NAME=visa_ep1_rw3,UCAD_EPOCHS=1,UCAD_BATCH_SIZE=8,UCAD_REWEIGHTING=3 \
#          code/pyCLAD/scripts/ucad_probe.sbatch
# See scripts/ucad_probe.py for the variables the probe reads.

set -uo pipefail

BASE=$SCRATCH/pp
cd "$BASE"
RUN_DIR=$BASE/runs/probe_${UCAD_NAME}_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"

# Athena names the module Python/3.12.3, Ares python/3.12.3-gcccore-13.3.0; the account and
# partition come from the submission (Ares uses -A plglscclass26-gpu -p plgrid-gpu-v100).
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
export UCAD_OUTPUT=$RUN_DIR/${UCAD_NAME}.json

$BASE/.venv/bin/python $BASE/code/pyCLAD/scripts/ucad_probe.py 2>&1 |
    tee "$RUN_DIR/${UCAD_NAME}.log" |
    grep --line-buffered -E "PROBE |evaluated on|Traceback|CUDA error|MemoryError" |
    sed "s/^/[$UCAD_NAME] /"
STATUS=${PIPESTATUS[0]}

echo "===PROBE_DONE=== $UCAD_NAME exit=$STATUS output=$UCAD_OUTPUT"
