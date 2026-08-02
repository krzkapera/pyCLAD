#!/bin/bash
#SBATCH --job-name=ucad-repo-query
#SBATCH --account=plglscclass26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out

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
export OMP_NUM_THREADS=2

# Checkpointy przezywaja miedzy uruchomieniami: ukonczone zadania sa wczytywane,
# wiec po TIMEOUT wystarczy zlozyc joba ponownie i dokonczy od miejsca przerwania.
export UCAD_CKPT_DIR=$BASE/runs/repo_query_ckpt
mkdir -p "$UCAD_CKPT_DIR"

RUN_DIR=$BASE/runs/repo_query
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/repo_query_$SLURM_JOB_ID.log"

echo ">>> checkpointy: $UCAD_CKPT_DIR ($(ls -1 $UCAD_CKPT_DIR 2>/dev/null | wc -l) zadan gotowych)"

CATEGORIES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
D_FLAGS=""
for c in $CATEGORIES; do D_FLAGS="$D_FLAGS -d $c"; done

sed -i 's/exec(compile(open("run_ucad.py").read(), "run_ucad.py", "exec"))/exec(compile(open("run_ucad_resumable.py").read(), "run_ucad_resumable.py", "exec"))/' launch.py

$BASE/.venv/bin/python launch.py --gpu 0 --seed 0 --memory_size 196 --epochs_num 25 \
    --log_group QUERY --log_project UCAD15Q "$RUN_DIR/results" \
    ucad -b wideresnet50 -le layer2 -le layer3 \
    --pretrain_embed_dimension 1024 --target_embed_dimension 1024 \
    --anomaly_scorer_num_nn 1 --patchsize 1 \
    sampler -p 0.1 approx_greedy_coreset \
    dataset --resize 224 --imagesize 224 $D_FLAGS --num_workers 2 mvtec "$BASE/data/mvtec2d" \
    2>&1 | tee "$LOG" |
    grep --line-buffered -E "WZNOWIENIE|CHECKPOINT|Evaluating dataset|^current task|Traceback|CUDA error|MemoryError"

echo "===DONE=== ukonczonych zadan: $(ls -1 $UCAD_CKPT_DIR 2>/dev/null | wc -l)/15"
