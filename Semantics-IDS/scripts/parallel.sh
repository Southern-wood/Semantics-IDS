#!/bin/bash

# --- Basic Configuration ---

# Maximum number of parallel jobs
MAX_CONCURRENT=3
# Delay (in seconds) before starting the *next* job
INITIAL_START_DELAY=45 # Delay before starting jobs for gpu memory allocation
# Maximum number of retries for a failed task
MAX_RETRIES=5
# Delay (in seconds) between retries
RETRY_DELAY=100
# Log directory
LOG_DIR="logs"
mkdir -p "$LOG_DIR" # make sure the log directory exists

# --- End Basic Configuration ---

# --- Color in Terminal ---
HEADER='\033[95m'
BLUE='\033[94m'
GREEN='\033[92m'
RED='\033[93m'
FAIL='\033[91m'
ENDC='\033[0m'
BOLD='\033[1m'
UNDERLINE='\033[4m'

STARING="$BOLD[STARTING]$ENDC"
SUCCESS="$GREEN[SUCCESS]$ENDC"
RETRY="$RED[RETRY]$ENDC"
FAILED="$FAIL[FAILED]$ENDC"


# ------------------------

# --- Read configurations from cross.json using jq ---
if ! command -v jq &> /dev/null
then
    echo "jq command could not be found. Please install jq to parse cross.json."
    exit 1
fi

if [ ! -f "scripts/cross.json" ]; then
    echo "Error: scripts/cross.json not found!"
    exit 1
fi

# Store JSON content in variables
single_types=($(jq -r '.single[]' scripts/cross.json))
mixup_types=($(jq -r '.mix[]' scripts/cross.json))

quality_types=("pure" "${single_types[@]}" "${mixup_types[@]}")
dataset_list=("SWaT" "WADI" "HAI")
modes=("train" "test")

# quality_types=("pure" "noise" "missing" "duplicate" "delay" "mismatch" "mix_1" "mix_2")



# --- Helper function to run a single task with retries ---
run_task() {
    local dataset_name="$1"
    local train_quality="$2" # Quality used for training the model
    local train_level="$3"   # Level used for training
    local mode="$4"          # "train" or "test"
    local retries=0
    local cmd_base="python main.py \
                    --dataset \"$dataset_name\" \
                    --quality_type \"$train_quality\" \
                    --level \"$train_level\" \
                    --mode \"$mode\""
    local cmd=""
    local log_file_suffix=""
    local task_desc_suffix=""

    # --- Command and Log File Construction ---
    if [[ "$train_quality" == "pure" ]]; then
        cmd="$cmd_base" # pure type doesn't use level for training 
        log_file_suffix="${dataset_name}_pure"
        task_desc_suffix="pure ${dataset_name}"
    else
        cmd="$cmd_base --level \"$train_level\""
        log_file_suffix="${dataset_name}_${train_quality}_${train_level}"
        task_desc_suffix="${train_quality} ${train_level} ${dataset_name}"
    fi

    log_file="${LOG_DIR}/${log_file_suffix}.log"
    if [[ "$mode" == "train" ]]; then
        task_desc="Training on $task_desc_suffix"
    else 
        log_file="${LOG_DIR}/${log_file_suffix}.log"
        task_desc="Testing model from $task_desc_suffix on its own data type"
    fi

    cmd="TQDM_DISABLE=1 $cmd"
    time="[$(date '+%Y-%m-%d %H:%M:%S')]"
    echo -e "$time $STARING $task_desc -> $log_file"

    # Run the command with retries
    while [ $retries -lt $MAX_RETRIES ]; do
        if eval "$cmd" > "$log_file" 2>&1; then
            time="[$(date '+%Y-%m-%d %H:%M:%S')]"
            echo -e "$time $SUCCESS $task_desc"
            status=0
            break
        else
            retries=$((retries + 1))
            time="[$(date '+%Y-%m-%d %H:%M:%S')]"
            echo -e "$time $RETRY [$retries/$MAX_RETRIES] $task_desc"
            sleep $RETRY_DELAY
        fi
    done
    
    time="[$(date '+%Y-%m-%d %H:%M:%S')]"
    [ $status -ne 0 ] && echo -e "$time $FAILED $task_desc"
    return $status
}

# Export the function and variables needed by parallel
export -f run_task
# Add STARING, SUCCESS, RETRY, FAILED to the export list
export LOG_DIR MAX_RETRIES RETRY_DELAY STARING SUCCESS RETRY FAILED

# --- Generate  Tasks ---
generate_args() {
    for dataset in "${dataset_list[@]}"; do
        for mode in "${modes[@]}"; do
            for quality in "${quality_types[@]}"; do
                echo "$dataset $quality low $mode"
            done
        done
    done
}

# --- Main Script ---
echo -e "$HEADER======= Parallel Tasks Started ========$ENDC"
echo Job count: $(echo $(generate_args | wc -l))
echo Maximum concurrent jobs: $MAX_CONCURRENT
echo Initial start delay: $INITIAL_START_DELAY seconds
echo Maximum retries: $MAX_RETRIES
echo Retry delay: $RETRY_DELAY seconds
echo "==========================================="
generate_args | parallel --line-buffer \
    --jobs $MAX_CONCURRENT \
    --delay $INITIAL_START_DELAY \
    --colsep ' ' \
    run_task {1} {2} {3} {4}
echo "===== All Tasks Completed ====="