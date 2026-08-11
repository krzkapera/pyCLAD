#!/bin/bash
#SBATCH --job-name=ucad-ref-visa
#SBATCH --account=plglscclass26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%j.out

# The reference implementation on VisA: 12 concepts, 9381 training images, 25 epochs, and a test-set
# evaluation after every epoch (it reports the best one). The same run over MVTec's 3629 training
# images took 2h51m, so this needs roughly 8h. run_ucad_resumable.py checkpoints each finished
# concept into UCAD_CKPT_DIR, so a re-submission with the same directory continues where it stopped.

set -uo pipefail

BASE=$SCRATCH/pp
REPO=$BASE/ucad-ref-run
cd "$REPO"

module load GCCcore/13.3.0
module load Python/3.12.3

export HF_HOME=$BASE/cache/hf
export TORCH_HOME=$BASE/cache/torch
export XDG_CACHE_HOME=$BASE/cache/xdg
export TMPDIR=$BASE/cache/tmp
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export UCAD_MEM_LIMIT_GB=40

RUN_DIR=$BASE/runs/ref_single_${REF_SEED:-0}_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"
export UCAD_CKPT_DIR=${UCAD_CKPT_DIR:-$BASE/runs/ref_single_ckpt_${REF_SEED:-0}}
mkdir -p "$UCAD_CKPT_DIR"
LOG="$RUN_DIR/reference_visa.log"

# The reference reads SCL masks from the image path with 'visa' replaced by 'visa-sam-b'.
$BASE/.venv/bin/python $BASE/code/pyCLAD/scripts/visa_sam_b_masks.py \
    "$BASE/data/visa-sam2" "$BASE/data/visa-sam-b" || exit 1

CATEGORIES=${REF_CATEGORIES:-"candle pipe_fryum pcb1"}
D_FLAGS=""
for c in $CATEGORIES; do D_FLAGS="$D_FLAGS -d $c"; done

$BASE/.venv/bin/python launch.py --gpu 0 --seed ${REF_SEED:-0} --memory_size 196 --epochs_num 25 \
    --log_group FULL --log_project UCAD12CLS "$RUN_DIR/results_full" \
    ucad -b wideresnet50 -le layer2 -le layer3 \
    --pretrain_embed_dimension 1024 --target_embed_dimension 1024 \
    --anomaly_scorer_num_nn 1 --patchsize 1 \
    sampler -p 0.1 approx_greedy_coreset \
    dataset --resize 224 --imagesize 224 $D_FLAGS --num_workers 4 mvtec "$BASE/data/visa" \
    2>&1 | tee "$LOG" |
    grep --line-buffered -E "SINGLE_EPOCH|Evaluating dataset|^current task|^Limited|^Nolimlited|CHECKPOINT|WZNOWIENIE|Traceback|CUDA error|MemoryError"
STATUS=${PIPESTATUS[0]}

echo "===REFERENCE_DONE=== exit=$STATUS results=$RUN_DIR/results_full"
