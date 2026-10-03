#!/bin/bash
#SBATCH --job-name=lab_6
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=gpu4090
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Usage: for re-read later
#   sbatch lab6.sbatch vanilla          # Q1
#   sbatch lab6.sbatch improved         # Q2
#   sbatch lab6.sbatch resnet18         # Q4
#   sbatch lab6.sbatch compare          # Q3 (after vanilla + improved finished)
# Extra args are passed through, example  sbatch lab6.sbatch vanilla --seed 1 --epochs 40

set -euo pipefail

SRC=$HOME/deep_learning/src/lab_6
DATA=${DATA:-$HOME/deep_learning/data/realwaste-main/RealWaste}
MODE=${1:?usage: sbatch lab6.sbatch <vanilla|improved|resnet18|compare> [extra args]}
shift

echo "=== Host ==="
echo "Job executed on: $(hostname)"
echo "Job started on: $(date)"

source $HOME/deep_learning/env_dl/bin/activate

echo "=== Python ==="
echo "Python executable location: $(which python)"
echo "Python version: $(python --version)"

echo "=== GPU ==="
nvidia-smi

echo "=== Execution ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
cd $SRC
if [ "$MODE" = "compare" ]; then
    python compare.py --data "$DATA" --workers $SLURM_CPUS_PER_TASK "$@"
else
    python train.py --model "$MODE" --data "$DATA" --workers $SLURM_CPUS_PER_TASK "$@"
fi
echo "Job finished on: $(date)"
