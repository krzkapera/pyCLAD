#!/bin/bash
#SBATCH --job-name=ucad-ref-dryrun
#SBATCH --account=plglscclass26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%j.out

set -uo pipefail

BASE=$SCRATCH/pp
cd "$BASE/ucad-ref-run"

module load GCCcore/13.3.0
module load Python/3.12.3

export HF_HOME=$BASE/cache/hf
export TORCH_HOME=$BASE/cache/torch
export XDG_CACHE_HOME=$BASE/cache/xdg
export TMPDIR=$BASE/cache/tmp
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2

$BASE/.venv/bin/python launch.py --gpu 0 --seed 0 --memory_size 196 --epochs_num 1 \
    --log_group DRYRUN --log_project DRYRUN15 logs/results_dryrun15 \
    ucad -b wideresnet50 -le layer2 -le layer3 \
    --pretrain_embed_dimension 1024 --target_embed_dimension 1024 \
    --anomaly_scorer_num_nn 1 --patchsize 1 \
    sampler -p 0.1 approx_greedy_coreset \
    dataset --resize 224 --imagesize 224 -d bottle --num_workers 2 mvtec "$BASE/data/mvtec2d"

echo "===DRYRUN_DONE==="
