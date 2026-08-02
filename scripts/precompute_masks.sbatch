#!/bin/bash
#SBATCH --job-name=ucad-precompute-masks
#SBATCH --account=plglscclass26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%A_%a.out

set -uo pipefail

BASE=$SCRATCH/pp
cd "$BASE"
RUN_DIR=$BASE/runs/$SLURM_ARRAY_JOB_ID
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
export VISA_ROOT=$BASE/data/visa
export OMP_NUM_THREADS=2

NUM_SHARDS=8
if [ "$SLURM_ARRAY_TASK_ID" -lt "$NUM_SHARDS" ]; then
    DATASET=mvtec
    SHARD=$SLURM_ARRAY_TASK_ID
    MASKS_ROOT=$BASE/data/mvtec2d-sam2
else
    DATASET=visa
    SHARD=$((SLURM_ARRAY_TASK_ID - NUM_SHARDS))
    MASKS_ROOT=$BASE/data/visa-sam2
fi
LOG="$RUN_DIR/${DATASET}_shard${SHARD}.log"

echo ">>> task=$SLURM_ARRAY_TASK_ID dataset=$DATASET shard=$SHARD/$NUM_SHARDS gpu=$CUDA_VISIBLE_DEVICES"

$BASE/.venv/bin/python $BASE/code/pyCLAD/scripts/precompute_sam2_masks.py \
    --dataset "$DATASET" --masks-root "$MASKS_ROOT" --shard-index "$SHARD" --num-shards "$NUM_SHARDS" 2>&1 |
    tee "$LOG" |
    grep --line-buffered -E "PROGRESS|DONE|Traceback|CUDA error|MemoryError"

echo "===TRAINING_DONE==="
