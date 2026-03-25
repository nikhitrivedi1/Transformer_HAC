#!/bin/bash
#SBATCH -J patchtst_single
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 04:00:00

#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# --- env setup ---
source $(conda info --base)/etc/profile.d/conda.sh
conda activate PatchTST

BASE_CONFIG="configs/config_1.yaml"

# ==========================================
# SET YOUR SPECIFIC VALUES HERE
# ==========================================
PATCH_LENGTH=64
STRIDE=64
LR=0.001
DROPOUT=0.2
NUM_LAYERS=5
OPTIMIZER="adamw"
N_HEADS=8
WEIGHT_DECAY=0.03
RUN_TYPE="test"  # 'validation' or 'test'

# Optional: override with command-line arguments
# Usage: sbatch run_job.sh [patch_length] [stride] [lr] [dropout] [num_layers] [optimizer] [n_heads] [weight_decay] [run_type]
if [ $# -ge 1 ]; then PATCH_LENGTH=$1; fi
if [ $# -ge 2 ]; then STRIDE=$2; fi
if [ $# -ge 3 ]; then LR=$3; fi
if [ $# -ge 4 ]; then DROPOUT=$4; fi
if [ $# -ge 5 ]; then NUM_LAYERS=$5; fi
if [ $# -ge 6 ]; then OPTIMIZER=$6; fi
if [ $# -ge 7 ]; then N_HEADS=$7; fi
if [ $# -ge 8 ]; then WEIGHT_DECAY=$8; fi
if [ $# -ge 9 ]; then RUN_TYPE=$9; fi

# ==========================================

# Build run name (make filesystem-friendly)
WD_TAG=${WEIGHT_DECAY//./p}
RUN_NAME="pl${PATCH_LENGTH}_s${STRIDE}_lr${LR}_do${DROPOUT}_L${NUM_LAYERS}_opt${OPTIMIZER}_h${N_HEADS}_wd${WD_TAG}"

echo "=========================================="
echo "Running PatchTST with specific parameters"
echo "=========================================="
echo "Config:       $BASE_CONFIG"
echo "Patch Length: $PATCH_LENGTH"
echo "Stride:       $STRIDE"
echo "Learning Rate: $LR"
echo "Dropout:      $DROPOUT"
echo "Num Layers:   $NUM_LAYERS"
echo "Optimizer:    $OPTIMIZER"
echo "N Heads:      $N_HEADS"
echo "Weight Decay: $WEIGHT_DECAY"
echo "Run Type:     $RUN_TYPE"
echo "Run Name:     $RUN_NAME"
echo "Output Dir:   $RUN_NAME"
echo "=========================================="

python patch_tst.py \
  --config "$BASE_CONFIG" \
  --patch_length "$PATCH_LENGTH" \
  --stride "$STRIDE" \
  --lr "$LR" \
  --t_dropout "$DROPOUT" \
  --t_num_layers "$NUM_LAYERS" \
  --optimizer_type "$OPTIMIZER" \
  --n_heads "$N_HEADS" \
  --weight_decay "$WEIGHT_DECAY" \
  --run_type "$RUN_TYPE" \
  --run_name "$RUN_NAME" \
  --output_dir "$RUN_NAME"

echo "Done!"
