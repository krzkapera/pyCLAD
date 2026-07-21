#!/bin/bash
#SBATCH --job-name=ucad-sam2-sweep
#SBATCH --account=plglscclass26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:30:00
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
export OMP_NUM_THREADS=2

SAM_CONFIGS=(${ONLY_SAM_CONFIG:-fast16 balanced32 sam2_defaults loose32})
SEEDS=(${ONLY_SEEDS:-0 1})
CFG=${SAM_CONFIGS[$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))]}
LOG="$RUN_DIR/${CFG}_s${SEED}.log"

echo ">>> task=$SLURM_ARRAY_TASK_ID sam_config=$CFG seed=$SEED gpu=$CUDA_VISIBLE_DEVICES"

$BASE/.venv/bin/python $BASE/code/pyCLAD/examples/ucad_sam2_sweep.py \
    --sam-config "$CFG" --seed "$SEED" 2>&1 |
    tee "$LOG" |
    grep --line-buffered -E "EPOCH |RESULT |Traceback|CUDA error|MemoryError"

echo "===TRAINING_DONE==="
