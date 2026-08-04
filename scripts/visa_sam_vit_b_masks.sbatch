#!/bin/bash
#SBATCH --job-name=visa-sam-vitb
#SBATCH --account=plglscclass26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%A_%a.out

# One category per array task: SAM ViT-B over ~800 training images each. The masks reproduce the
# authors' recipe, so they are the missing half of the VisA comparison - the paper's VisA numbers
# come from ViT-B masks that were never released, while our runs so far used SAM2.

set -uo pipefail

BASE=$SCRATCH/pp
cd "$BASE/ucad-ref-run"

module load GCCcore/13.3.0
module load Python/3.12.3

export HF_HOME=$BASE/cache/hf
export TORCH_HOME=$BASE/cache/torch
export XDG_CACHE_HOME=$BASE/cache/xdg
export TMPDIR=$BASE/cache/tmp
export PYTHONPATH=$BASE/ucad-ref-run
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

CATEGORIES=(candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum)
CATEGORY=${CATEGORIES[$SLURM_ARRAY_TASK_ID]}

$BASE/.venv/bin/python $BASE/code/pyCLAD/scripts/visa_sam_vit_b_masks.py \
    "$BASE/data/visa" "$BASE/data/visa-sam-vitb" "$BASE/sam_vit_b_01ec64.pth" "$CATEGORY"

echo "===MASKS_DONE=== $CATEGORY"
