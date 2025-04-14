#!/bin/bash

# --- Configuration ---

# Maximum number of parallel jobs
MAX_CONCURRENT=3
# Delay (in seconds) before starting the *next* job
INITIAL_START_DELAY=30 # Delay before starting jobs for gpu memory allocation
# Maximum number of retries for a failed task
MAX_RETRIES=3
# Delay (in seconds) between retries
RETRY_DELAY=1000
# Log directory
LOG_DIR="logs"
mkdir -p "$LOG_DIR" # make sure the log directory exists


quality_types=("pure" "noise" "missing" "duplicate" "delay" "mismatch")
dataset_list=("HAI" "SWaT" "WADI")
modes=("train" "test")

basic_params="--win_size 105 --patch_size 357 --anormly_ratio 1 --loss_fuc MSE"

# --- Helper function to run a single task with retries ---
run_task() {
    local dataset="$1"
    local quality="$2"
    local level="$3" 
    local mode="$4"  # "train" or "test"
    local retries=0
    local cmd_base="python main.py --dataset \"$dataset\" --quality \"$quality\" --mode \"$mode\""
    local cmd=""
    local log_file=""
    local task_desc=""

    # Channel_num and batch_size based on dataset
    local channel_num batch_size
    case "$dataset" in
        "SWaT") channel_num=46; batch_size=128 ;;
        "WADI") channel_num=98; batch_size=64 ;;
        "HAI") channel_num=50; batch_size=64 ;; 
    esac

    # construct command and log file based on parameters
    local cmd="python main.py \
        $basic_params \
        --dataset $dataset \
        --quality_type $quality\
        --level $level \
        --input_c $channel_num \
        --output_c $channel_num \
        --batch_size $batch_size \
        --mode $mode"
    
   
    # No need to set level for pure quality
    if [ "$quality" != "pure" ]; then
        cmd+=" --level $level"
    fi

    # num_epochs and retrain based on mode
    if [ "$mode" == "train" ]; then
        cmd+=" --num_epochs 3"
    else
        cmd+=" --num_epochs 10" 
    fi

    local log_file
    if [ "$quality" == "pure" ]; then
        log_file="${LOG_DIR}/${dataset}_pure_${mode}.log"
    else
        log_file="${LOG_DIR}/${dataset}_${quality}_${level}_${mode}.log"
    fi

    
    # Create log file if it doesn't exist
    local task_desc="${mode} ${dataset} ${quality}"
    [ "$quality" != "pure" ] && task_desc+=" ${level}"

    
    echo "[START] $task_desc -> $log_file"

    # Run the command with retries
    while [ $retries -lt $MAX_RETRIES ]; do
        if eval "$cmd" > "$log_file" 2>&1; then
            echo "[SUCCESS] $task_desc"
            status=0
            break
        else
            retries=$((retries + 1))
            echo "[RETRY $retries/$MAX_RETRIES] $task_desc"
            sleep $RETRY_DELAY
        fi
    done

    [ $status -ne 0 ] && echo "[FAILED] $task_desc"
    return $status
}

# Export the function and variables needed by parallel
export -f run_task
export LOG_DIR MAX_RETRIES RETRY_DELAY

# --- Generate  Tasks ---
generate_args() {
    for mode in "${modes[@]}"; do
        for dataset in "${dataset_list[@]}"; do
            for quality in "${quality_types[@]}"; do
                if [ "$quality" == "pure" ]; then
                    # pure quality doesn't have a level, actually unused 
                    echo "$dataset $quality low $mode"
                else
                    echo "$dataset $quality low $mode"
                    echo "$dataset $quality high $mode"
                fi
            done
        done
    done
}

# --- Main Script ---
echo "===== Experiment Started ====="
generate_args | parallel --line-buffer \
    --jobs $MAX_CONCURRENT \
    --delay $INITIAL_START_DELAY \
    --colsep ' ' \
    run_task {1} {2} {3} {4}
echo "===== All Tasks Completed ====="