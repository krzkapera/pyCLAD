#!/bin/bash
#SBATCH --job-name=ucad-hyp2
#SBATCH --account=plglscclass26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
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
export MVTEC_MASKS_ROOT=$BASE/data/mvtec2d-sam-b
export MVTEC_PATCHSIZE=1
export MVTEC_CORESET_MODE=exact
export MVTEC_BLUR_SIGMA=3.0
export OMP_NUM_THREADS=2

# Bez ewaluacji co-epokowej: raportujemy tylko protokol 'ours' (last-epoch), ~4x szybciej,
# co pozwala zmiescic sie w limicie czasu. Baseline 'ours' do porownania: 0.797 / 0.307.
export MVTEC_TRACK_AUTHORS=0

case $SLURM_ARRAY_TASK_ID in
    0) NAME=r_baseline;           export MVTEC_PROMPT_LENGTH=5; export MVTEC_RESET_PROMPT=0 ;;
    1) NAME=r_resetprompt;        export MVTEC_PROMPT_LENGTH=5; export MVTEC_RESET_PROMPT=1 ;;
    2) NAME=r_plen1;              export MVTEC_PROMPT_LENGTH=1; export MVTEC_RESET_PROMPT=0 ;;
    3) NAME=r_plen1_resetprompt;  export MVTEC_PROMPT_LENGTH=1; export MVTEC_RESET_PROMPT=1 ;;
esac

LOG="$RUN_DIR/${NAME}.log"
echo ">>> task=$SLURM_ARRAY_TASK_ID variant=$NAME prompt_length=$MVTEC_PROMPT_LENGTH reset=$MVTEC_RESET_PROMPT"

$BASE/.venv/bin/python $BASE/code/pyCLAD/examples/ucad_mvtec_example.py 2>&1 |
    tee "$LOG" |
    grep --line-buffered -E "CONFIG |RESULT |Task [0-9]+ added|Traceback|CUDA error|MemoryError" |
    sed "s/^/[$NAME] /"

echo "===TRAINING_DONE=== $NAME"
