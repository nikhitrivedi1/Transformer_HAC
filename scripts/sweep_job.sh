#!/bin/bash
#SBATCH -J patchtst_grid
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 08:00:00

#SBATCH --array=0-3

#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

# Signal handler: receive USR1 5 minutes before timeout
#SBATCH --signal=B:USR1@300

# Note: Not using 'set -e' because we need to handle training failures gracefully
set -uo pipefail
mkdir -p logs

# ==============================================================================
# CHECKPOINTING & JOB CHAINING SETUP
# ==============================================================================

# Directory for lightweight completion markers (no model weights)
COMPLETION_DIR="completed_runs"
mkdir -p "$COMPLETION_DIR"

# Directory for model weight checkpoints (only saved on interruption)
WEIGHT_CHECKPOINT_DIR="weight_checkpoints"
mkdir -p "$WEIGHT_CHECKPOINT_DIR"

# Flag to track if we received timeout signal
TIMEOUT_SIGNAL_RECEIVED=0

# PID of the currently running training process
TRAIN_PID=""

# Current run name (for checkpoint naming)
CURRENT_RUN_NAME=""

# Signal handler for graceful shutdown before timeout
handle_timeout() {
    echo ""
    echo "=========================================="
    echo "TIMEOUT WARNING: Received USR1 signal"
    echo "Time limit approaching..."
    echo "=========================================="
    TIMEOUT_SIGNAL_RECEIVED=1
    
    # Forward signal to training process if running
    if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "Forwarding USR1 to training process (PID: $TRAIN_PID)"
        echo "Training will save weights to: ${WEIGHT_CHECKPOINT_DIR}/${CURRENT_RUN_NAME}"
        kill -USR1 "$TRAIN_PID" 2>/dev/null || true
    fi
}

# Set up signal trap
trap 'handle_timeout' USR1

# Function to check if a run is already completed (lightweight marker file)
is_run_completed() {
    local run_name=$1
    [[ -f "${COMPLETION_DIR}/${run_name}.done" ]]
}

# Function to mark a run as completed (lightweight - just touches a file)
mark_run_completed() {
    local run_name=$1
    touch "${COMPLETION_DIR}/${run_name}.done"
    echo "  [COMPLETE] Run marked as done: ${run_name}"
}

# Function to check if a weight checkpoint exists (for resuming interrupted runs)
has_weight_checkpoint() {
    local run_name=$1
    [[ -d "${WEIGHT_CHECKPOINT_DIR}/${run_name}" ]] && \
    [[ -f "${WEIGHT_CHECKPOINT_DIR}/${run_name}/checkpoint.pt" ]]
}

# Function to resubmit this job
resubmit_job() {
    echo ""
    echo "=========================================="
    echo "RESUBMITTING JOB"
    echo "=========================================="
    
    # Get the script path
    local script_path="${BASH_SOURCE[0]}"
    if [[ ! -f "$script_path" ]]; then
        script_path="sweep_job.sh"
    fi
    
    # Resubmit with same parameters
    # The new job will read completion markers and skip completed tasks
    local new_job_id
    new_job_id=$(sbatch --parsable "$script_path")
    
    echo "Resubmitted as job: $new_job_id"
    echo "Completion markers in: $COMPLETION_DIR/"
    echo "Completed tasks will be skipped on resume"
    echo "=========================================="
}

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

echo "Setting up environment..."
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo "Date: $(date)"

# Activate conda
if ! source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null; then
    echo "ERROR: Failed to source conda"
    exit 1
fi

if ! conda activate PatchTST; then
    echo "ERROR: Failed to activate PatchTST environment"
    exit 1
fi

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

BASE_CONFIG="configs/config_1.yaml"

# Verify config file exists
if [[ ! -f "$BASE_CONFIG" ]]; then
    echo "ERROR: Config file not found: $BASE_CONFIG"
    exit 1
fi
echo "Config file: $BASE_CONFIG (found)"

# ==============================================================================
# GRID VALUES
# ==============================================================================

PATCH_LENGTHS=(32 64 128)
STRIDES=(8 16 32)
LRS=(0.001)
DROPOUTS=(0.2 0.5 0.7)
LAYERS=(3 5 7)

OPTIMIZERS=("adamw")
HEADS=(4 8 16)

# Weight decay grid
WEIGHT_DECAYS=(0.01 0.03 0.05)

# Classifier Grid
CLASSIFIERS=("mlp")

# Augmentation probabilities
TWARP_PROBS=(0 0.3 0.5)
MWARP_PROBS=(0 0.3 0.5)
SHIFT_PROBS=(0 0.2 0.4)
JITTER_PROBS=(0)

# ==============================================================================
# COMPUTE GRID SIZE
# ==============================================================================

n_patch=${#PATCH_LENGTHS[@]}
n_stride=${#STRIDES[@]}
n_lr=${#LRS[@]}
n_drop=${#DROPOUTS[@]}
n_layers=${#LAYERS[@]}
n_opt=${#OPTIMIZERS[@]}
n_heads=${#HEADS[@]}
n_wd=${#WEIGHT_DECAYS[@]}
n_classifier=${#CLASSIFIERS[@]}
n_jitter_prob=${#JITTER_PROBS[@]}
n_shift_prob=${#SHIFT_PROBS[@]}
n_twarp_prob=${#TWARP_PROBS[@]}
n_mwarp_prob=${#MWARP_PROBS[@]}
TOTAL=$((n_patch * n_stride * n_lr * n_drop * n_layers * n_opt * n_heads * n_wd * n_classifier * n_jitter_prob * n_shift_prob * n_twarp_prob * n_mwarp_prob))
WORKERS=4

# ==============================================================================
# DECODE AND RUN FUNCTION
# ==============================================================================

decode_and_run () {
  local tid=$1
  local t=$tid

  # Decode in reverse order of multiplication
  # Parameters decoded FIRST change FASTEST, decoded LAST change SLOWEST
  local idx_wd=$(( t % n_wd ));         t=$(( t / n_wd ))
  local idx_heads=$(( t % n_heads ));   t=$(( t / n_heads ))
  local idx_opt=$(( t % n_opt ));       t=$(( t / n_opt ))
  local idx_layers=$(( t % n_layers )); t=$(( t / n_layers ))
  local idx_drop=$(( t % n_drop ));     t=$(( t / n_drop ))
  local idx_lr=$(( t % n_lr ));         t=$(( t / n_lr ))
  local idx_stride=$(( t % n_stride )); t=$(( t / n_stride ))
  local idx_patch=$(( t % n_patch ));   t=$(( t / n_patch ))
  local idx_classifier=$(( t % n_classifier )); t=$(( t / n_classifier ))
  # Augmentation probs - jitter_prob decoded LAST so it traverses slowest
  local idx_mwarp_prob=$(( t % n_mwarp_prob )); t=$(( t / n_mwarp_prob ))
  local idx_twarp_prob=$(( t % n_twarp_prob )); t=$(( t / n_twarp_prob ))
  local idx_shift_prob=$(( t % n_shift_prob )); t=$(( t / n_shift_prob ))
  local idx_jitter_prob=$(( t % n_jitter_prob ))  # LAST - changes slowest

  local patch_length=${PATCH_LENGTHS[$idx_patch]}
  local stride=${STRIDES[$idx_stride]}
  local lr=${LRS[$idx_lr]}
  local t_dropout=${DROPOUTS[$idx_drop]}
  local t_num_layers=${LAYERS[$idx_layers]}
  local optimizer_type=${OPTIMIZERS[$idx_opt]}
  local n_heads_val=${HEADS[$idx_heads]}
  local weight_decay=${WEIGHT_DECAYS[$idx_wd]}
  local classifier=${CLASSIFIERS[$idx_classifier]}
  local twarp_prob=${TWARP_PROBS[$idx_twarp_prob]}
  local mwarp_prob=${MWARP_PROBS[$idx_mwarp_prob]}
  local shift_prob=${SHIFT_PROBS[$idx_shift_prob]}
  local jitter_prob=${JITTER_PROBS[$idx_jitter_prob]}

  # Make run_name filesystem-friendly
  local wd_tag=${weight_decay//./p}

  local run_name="pl${patch_length}_s${stride}_lr${lr}_do${t_dropout}_L${t_num_layers}_opt${optimizer_type}_h${n_heads_val}_wd${wd_tag}_classifier${classifier}_jitter_prob${jitter_prob}_shift_prob${shift_prob}_twarp_prob${twarp_prob}_mwarp_prob${mwarp_prob}"

  echo ""
  echo "=========================================="
  echo "tid=$tid -> $run_name"
  echo "Params: patch_length=$patch_length stride=$stride lr=$lr t_dropout=$t_dropout t_num_layers=$t_num_layers optimizer_type=$optimizer_type n_heads=$n_heads_val weight_decay=$weight_decay classifier=$classifier jitter_prob=$jitter_prob shift_prob=$shift_prob twarp_prob=$twarp_prob mwarp_prob=$mwarp_prob"
  echo "=========================================="

  # Set current run name for signal handler
  CURRENT_RUN_NAME="$run_name"
  
  # Build the command arguments
  local cmd_args=(
    python train.py
    --config "$BASE_CONFIG"
    --patch_length "$patch_length"
    --stride "$stride"
    --lr "$lr"
    --t_dropout "$t_dropout"
    --t_num_layers "$t_num_layers"
    --optimizer_type "$optimizer_type"
    --n_heads "$n_heads_val"
    --weight_decay "$weight_decay"
    --run_name "$run_name"
    --classifier "$classifier"
    --jitter_prob "$jitter_prob"
    --shift_prob "$shift_prob"
    --twarp_prob "$twarp_prob"
    --mwarp_prob "$mwarp_prob"
    --checkpoint_dir "$WEIGHT_CHECKPOINT_DIR"
  )
  
  # Check for existing weight checkpoint (interrupted run)
  if has_weight_checkpoint "$run_name"; then
      echo "  [RESUME] Found weight checkpoint - resuming from: ${WEIGHT_CHECKPOINT_DIR}/${run_name}"
      cmd_args+=(--resume_from "${WEIGHT_CHECKPOINT_DIR}/${run_name}/checkpoint.pt")
  fi

  # Run training in background so we can capture PID for signal forwarding
  "${cmd_args[@]}" &
  
  TRAIN_PID=$!
  
  # Wait for training to complete (wait returns exit code of the process)
  local exit_code=0
  wait $TRAIN_PID || exit_code=$?
  
  TRAIN_PID=""
  CURRENT_RUN_NAME=""
  
  return $exit_code
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

RANK=${SLURM_ARRAY_TASK_ID:-0}

echo "=========================================="
echo "SWEEP JOB STARTED"
echo "=========================================="
echo "Array rank: $RANK / $((WORKERS-1))"
echo "TOTAL configs: $TOTAL"
echo "Completion markers: $COMPLETION_DIR/"
echo "Weight checkpoints: $WEIGHT_CHECKPOINT_DIR/ (only on interruption)"
echo "This job will run tids: $RANK, $((RANK+WORKERS)), $((RANK+2*WORKERS)), ..."
echo "=========================================="

# Helper function to get run_name for a tid (needed for completion check)
get_run_name_for_tid() {
    local tid=$1
    local t=$tid
    
    local idx_wd=$(( t % n_wd ));         t=$(( t / n_wd ))
    local idx_heads=$(( t % n_heads ));   t=$(( t / n_heads ))
    local idx_opt=$(( t % n_opt ));       t=$(( t / n_opt ))
    local idx_layers=$(( t % n_layers )); t=$(( t / n_layers ))
    local idx_drop=$(( t % n_drop ));     t=$(( t / n_drop ))
    local idx_lr=$(( t % n_lr ));         t=$(( t / n_lr ))
    local idx_stride=$(( t % n_stride )); t=$(( t / n_stride ))
    local idx_patch=$(( t % n_patch ));   t=$(( t / n_patch ))
    local idx_classifier=$(( t % n_classifier )); t=$(( t / n_classifier ))
    local idx_mwarp_prob=$(( t % n_mwarp_prob )); t=$(( t / n_mwarp_prob ))
    local idx_twarp_prob=$(( t % n_twarp_prob )); t=$(( t / n_twarp_prob ))
    local idx_shift_prob=$(( t % n_shift_prob )); t=$(( t / n_shift_prob ))
    local idx_jitter_prob=$(( t % n_jitter_prob ))
    
    local patch_length=${PATCH_LENGTHS[$idx_patch]}
    local stride=${STRIDES[$idx_stride]}
    local lr=${LRS[$idx_lr]}
    local t_dropout=${DROPOUTS[$idx_drop]}
    local t_num_layers=${LAYERS[$idx_layers]}
    local optimizer_type=${OPTIMIZERS[$idx_opt]}
    local n_heads_val=${HEADS[$idx_heads]}
    local weight_decay=${WEIGHT_DECAYS[$idx_wd]}
    local classifier=${CLASSIFIERS[$idx_classifier]}
    local twarp_prob=${TWARP_PROBS[$idx_twarp_prob]}
    local mwarp_prob=${MWARP_PROBS[$idx_mwarp_prob]}
    local shift_prob=${SHIFT_PROBS[$idx_shift_prob]}
    local jitter_prob=${JITTER_PROBS[$idx_jitter_prob]}
    local wd_tag=${weight_decay//./p}
    
    echo "pl${patch_length}_s${stride}_lr${lr}_do${t_dropout}_L${t_num_layers}_opt${optimizer_type}_h${n_heads_val}_wd${wd_tag}_classifier${classifier}_jitter_prob${jitter_prob}_shift_prob${shift_prob}_twarp_prob${twarp_prob}_mwarp_prob${mwarp_prob}"
}

# Count tasks
echo "Counting tasks for this worker..."
TASKS_FOR_THIS_WORKER=0
TASKS_ALREADY_DONE=0
TASKS_WITH_CHECKPOINT=0
for ((tid=RANK; tid<TOTAL; tid+=WORKERS)); do
    TASKS_FOR_THIS_WORKER=$((TASKS_FOR_THIS_WORKER + 1))
    run_name=$(get_run_name_for_tid "$tid")
    if is_run_completed "$run_name"; then
        TASKS_ALREADY_DONE=$((TASKS_ALREADY_DONE + 1))
    elif has_weight_checkpoint "$run_name"; then
        TASKS_WITH_CHECKPOINT=$((TASKS_WITH_CHECKPOINT + 1))
    fi
done
echo "Tasks assigned to this worker: $TASKS_FOR_THIS_WORKER"
echo "Tasks already completed: $TASKS_ALREADY_DONE"
echo "Tasks with weight checkpoint (will resume): $TASKS_WITH_CHECKPOINT"
echo "Tasks remaining: $((TASKS_FOR_THIS_WORKER - TASKS_ALREADY_DONE))"
echo "=========================================="

# Main loop
echo "Starting main training loop..."
TASKS_COMPLETED_THIS_RUN=0
for ((tid=RANK; tid<TOTAL; tid+=WORKERS)); do
    
    # Check for timeout signal before starting new task
    if [[ $TIMEOUT_SIGNAL_RECEIVED -eq 1 ]]; then
        echo ""
        echo "Timeout signal received - stopping before task $tid"
        break
    fi
    
    # Get run name for this task
    run_name=$(get_run_name_for_tid "$tid")
    
    # Skip if already completed (lightweight marker file exists)
    if is_run_completed "$run_name"; then
        echo "[SKIP] Task $tid already completed: $run_name"
        continue
    fi
    
    # Run the task (will resume from checkpoint if exists)
    if decode_and_run "$tid"; then
        # Mark as completed with lightweight marker (no weights saved)
        mark_run_completed "$run_name"
        TASKS_COMPLETED_THIS_RUN=$((TASKS_COMPLETED_THIS_RUN + 1))
        
        # Clean up weight checkpoint if it existed (training completed successfully)
        if [[ -d "${WEIGHT_CHECKPOINT_DIR}/${run_name}" ]]; then
            echo "  [CLEANUP] Removing weight checkpoint (training completed)"
            rm -rf "${WEIGHT_CHECKPOINT_DIR}/${run_name}"
        fi
    else
        echo "  [WARNING] Task $tid exited with error or was interrupted"
        # Weight checkpoint may have been saved by train.py on USR1
    fi
done

# ==============================================================================
# CLEANUP AND POTENTIAL RESUBMIT
# ==============================================================================

echo ""
echo "=========================================="
echo "SWEEP JOB SUMMARY (Rank $RANK)"
echo "=========================================="
echo "Tasks completed this run: $TASKS_COMPLETED_THIS_RUN"

# Count remaining and interrupted tasks
REMAINING=0
INTERRUPTED=0
for ((tid=RANK; tid<TOTAL; tid+=WORKERS)); do
    run_name=$(get_run_name_for_tid "$tid")
    if ! is_run_completed "$run_name"; then
        REMAINING=$((REMAINING + 1))
        if has_weight_checkpoint "$run_name"; then
            INTERRUPTED=$((INTERRUPTED + 1))
        fi
    fi
done

echo "Tasks remaining: $REMAINING"
echo "Tasks with saved weights (interrupted): $INTERRUPTED"

# Check if we need to resubmit
if [[ $TIMEOUT_SIGNAL_RECEIVED -eq 1 ]] && [[ $REMAINING -gt 0 ]]; then
    resubmit_job
elif [[ $REMAINING -eq 0 ]]; then
    echo "All tasks for rank $RANK completed successfully!"
else
    echo "Some tasks may have failed - check logs"
    echo "Interrupted tasks will resume from weight checkpoints on next run"
fi

echo "=========================================="
echo "Done rank=$RANK"
