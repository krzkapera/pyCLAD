#!/bin/bash
#SBATCH --job-name=ucad-visa-smoke
#SBATCH --account=plglscclass26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out

set -uo pipefail

BASE=$SCRATCH/pp
cd "$BASE"
RUN_DIR=$BASE/runs/visa_smoke_$SLURM_JOB_ID
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
export UCAD_OUTPUT=$RUN_DIR/ucad_visa_smoke.json

$BASE/.venv/bin/python $BASE/code/pyCLAD/scripts/visa_smoke.py 2>&1 | tee "$RUN_DIR/smoke.log" | tail -40
STATUS=${PIPESTATUS[0]}

echo "===SMOKE_DONE=== exit=$STATUS"
