#!/bin/bash
#SBATCH --job-name=ucad-visa-probe
#SBATCH --account=plglscclass26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%A_%a.out

set -uo pipefail

BASE=$SCRATCH/pp
cd "$BASE"
RUN_DIR=$BASE/runs/visa_probe_$SLURM_ARRAY_JOB_ID
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

# Task 3 repeats the full run's configuration; 4 uses the reference implementation's batch size.
case $SLURM_ARRAY_TASK_ID in
    0) NAME=ep01_b08; export UCAD_EPOCHS=1;  export UCAD_BATCH_SIZE=8  ;;
    1) NAME=ep05_b08; export UCAD_EPOCHS=5;  export UCAD_BATCH_SIZE=8  ;;
    2) NAME=ep10_b08; export UCAD_EPOCHS=10; export UCAD_BATCH_SIZE=8  ;;
    3) NAME=ep25_b08; export UCAD_EPOCHS=25; export UCAD_BATCH_SIZE=8  ;;
    4) NAME=ep25_b24; export UCAD_EPOCHS=25; export UCAD_BATCH_SIZE=24 ;;
esac

export UCAD_OUTPUT=$RUN_DIR/$NAME.json

$BASE/.venv/bin/python $BASE/code/pyCLAD/scripts/visa_probe.py 2>&1 |
    tee "$RUN_DIR/$NAME.log" |
    grep --line-buffered -E "PROBE |evaluated on|Traceback|CUDA error|MemoryError" |
    sed "s/^/[$NAME] /"
STATUS=${PIPESTATUS[0]}

echo "===PROBE_DONE=== $NAME exit=$STATUS"
