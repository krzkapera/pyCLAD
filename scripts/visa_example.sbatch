#!/bin/bash
#SBATCH --job-name=ucad-visa
#SBATCH --account=plglscclass26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

# 12 concepts, 9381 training images, 25 epochs each: ~3 h of prompt tuning plus a 12x12
# evaluation matrix over 1440 test images. Memory is sized for the exact coreset of the
# largest concept (986 images x 196 patches x 1024 dims, held on host and on the GPU).

set -uo pipefail

BASE=$SCRATCH/pp
cd "$BASE"
RUN_DIR=$BASE/runs/visa_$SLURM_JOB_ID
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
export VISA_ROOT=$BASE/data/visa
export VISA_MASKS_ROOT=$BASE/data/visa-sam2
export OMP_NUM_THREADS=4
export UCAD_OUTPUT=$RUN_DIR/ucad_visa.json

LOG="$RUN_DIR/visa.log"

$BASE/.venv/bin/python $BASE/code/pyCLAD/examples/ucad_visa_example.py 2>&1 |
    tee "$LOG" |
    grep --line-buffered -E "evaluated on|Starting training on concept|added to memory bank|Epoch 25 Loss|Traceback|CUDA error|MemoryError"
STATUS=${PIPESTATUS[0]}

echo "===TRAINING_DONE=== exit=$STATUS output=$UCAD_OUTPUT"
