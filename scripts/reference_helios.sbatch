#!/bin/bash -l
#SBATCH --job-name=ucad-ref
#SBATCH --account=plgadmgr2025-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

# The authors' implementation on Helios. ML-bundle must come first: the GPU nodes are aarch64 and the
# venv is built against the Python it provides. REF_DATASET picks the benchmark, REF_CATEGORIES the
# subset, REF_SEED the seed; run_ucad_resumable.py checkpoints each finished concept into
# UCAD_CKPT_DIR, so a re-submission with the same directory continues where it stopped.

set -uo pipefail

ml ML-bundle/25.10

BASE=/net/scratch/hscra/plgrid/plgkrzkapera/pp
REPO=$BASE/ucad-ref-run
cd "$REPO"

export HF_HOME=$BASE/cache/hf
export TORCH_HOME=$BASE/cache/torch
export XDG_CACHE_HOME=$BASE/cache/xdg
export TMPDIR=$BASE/cache/tmp
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export UCAD_MEM_LIMIT_GB=40

REF_DATASET=${REF_DATASET:-visa}
REF_SEED=${REF_SEED:-0}
RUN_DIR=$BASE/runs/ref_${REF_DATASET}_${REF_SEED}_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"
export UCAD_CKPT_DIR=${UCAD_CKPT_DIR:-$BASE/runs/ref_ckpt_${REF_DATASET}_${REF_SEED}}
mkdir -p "$UCAD_CKPT_DIR"

if [ "$REF_DATASET" = "visa" ]; then
    DATA_ROOT=$BASE/data/visa
    DEFAULT_CATEGORIES="candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum"
    # The reference reads its SCL masks from the image path with 'visa' replaced by 'visa-sam-b'.
    $BASE/.venv-ml/bin/python $BASE/code/pyCLAD/scripts/visa_sam_b_masks.py \
        "$BASE/data/visa-sam2" "$BASE/data/visa-sam-b" || exit 1
else
    DATA_ROOT=$BASE/data/mvtec2d
    DEFAULT_CATEGORIES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
fi

CATEGORIES=${REF_CATEGORIES:-$DEFAULT_CATEGORIES}
D_FLAGS=""
for c in $CATEGORIES; do D_FLAGS="$D_FLAGS -d $c"; done

$BASE/.venv-ml/bin/python launch.py --gpu 0 --seed $REF_SEED --memory_size 196 --epochs_num ${REF_EPOCHS:-25} \
    --log_group FULL --log_project UCADREF "$RUN_DIR/results" \
    ucad -b wideresnet50 -le layer2 -le layer3 \
    --pretrain_embed_dimension 1024 --target_embed_dimension 1024 \
    --anomaly_scorer_num_nn 1 --patchsize 1 \
    sampler -p 0.1 approx_greedy_coreset \
    dataset --resize 224 --imagesize 224 $D_FLAGS --num_workers 2 mvtec "$DATA_ROOT" \
    2>&1 | tee "$RUN_DIR/reference.log" |
    grep --line-buffered -E "SINGLE_EPOCH|Evaluating dataset|Traceback|CUDA error|MemoryError"
echo "===REFERENCE_DONE=== exit=${PIPESTATUS[0]} results=$RUN_DIR"
