#!/bin/bash

# --- Configuration ---

# Maximum number of parallel jobs
MAX_CONCURRENT=6
# Delay (in seconds) before starting the *next* job
INITIAL_START_DELAY=30 # Delay before starting jobs for gpu memory allocation
# Maximum number of retries for a failed task
MAX_RETRIES=3
# Delay (in seconds) between retries
RETRY_DELAY=1000
# Log directory
LOG_DIR="logs"
mkdir -p "$LOG_DIR" # make sure the log directory exists


quality_type=("pure" "noise" "missing" "duplicate" "delay" "mismatch")
dataset=("SWaT" "WADI" "HAI")


# --- Helper function to run a single task with retries ---
run_task() {
    local dataset="$1"
    local quality="$2"
    local level="$3" # Can be empty string ""
    local mode="$4"  # "train" or "test"
    local retries=0
    local cmd_base="python main.py --dataset \"$dataset\" --quality \"$quality\" --mode \"$mode\""
    local cmd=""
    local log_file=""
    local task_desc=""

    # Construct command and log file based on parameters
    if [[ "$quality" == "pure" ]]; then
        cmd="$cmd_base" # pure type doesn't need level
        if [[ "$mode" == "train" ]]; then
             log_file="${LOG_DIR}/${dataset}_pure_train.log" # 
             task_desc="Training on pure ${dataset} dataset"
        else # test mode
             log_file="${LOG_DIR}/${dataset}_pure_test.log"
             task_desc="Testing on pure ${dataset} dataset"
        fi
    else # Non-pure quality types
        cmd="$cmd_base --level \"$level\"" # None-pure type need a level
         if [[ "$mode" == "train" ]]; then
             cmd="$cmd --retrain" # Add --retrain for training
             log_file="${LOG_DIR}/${dataset}_${quality}_${level}_train.log" # 
             task_desc="Training on ${quality} ${level} ${dataset} dataset"
         else # test mode
             log_file="${LOG_DIR}/${dataset}_${quality}_${level}_test.log"
             task_desc="Testing on ${quality} ${level} ${dataset} dataset"
         fi
    fi

    # Add redirection for all cases
    local full_cmd="$cmd > \"$log_file\" 2>/dev/null" # Redirect stdout to log file and stderr to /dev/null
    echo "[STARTING] $task_desc (Log: $log_file)"


    # Execute with retry logic
    until eval "$full_cmd"; do
        retries=$((retries + 1))
        if [[ $retries -gt $MAX_RETRIES ]]; then
            echo "[FAILED] $task_desc after $MAX_RETRIES retries. Check log: $log_file" >&2
            return 1 # Indicate failure
        fi
        echo "[RETRYING ($retries/$MAX_RETRIES)] $task_desc in ${RETRY_DELAY}s..." >&2
        sleep "$RETRY_DELAY"
    done

    echo "[SUCCESS] $task_desc"
    return 0 # Indicate success
}

# Export the function and variables needed by parallel
export -f run_task
export LOG_DIR MAX_RETRIES RETRY_DELAY

# # --- Generate Training Tasks ---
# echo "--- Generating Training Tasks ---"
# train_tasks_args=()
# for i in "${dataset[@]}"; do
#     for j in "${quality_type[@]}"; do
#         if [ "$j" == "pure" ]; then
#             train_tasks_args+=("$i" "pure" "low" "train") # level is unused
#         else
#             train_tasks_args+=("$i" "$j" "low" "train")
#             train_tasks_args+=("$i" "$j" "high" "train")
#         fi
#     done
# done

# echo "--- Running Training Tasks (Max Parallel: $MAX_CONCURRENT, Delay: $INITIAL_START_DELAY s) ---"
# printf "%s\n" "${train_tasks_args[@]}" |  paste -d ' ' - - - - | parallel --line-buffer --jobs "$MAX_CONCURRENT" --delay "$INITIAL_START_DELAY" --colsep ' ' run_task {1} {2} {3} {4}
# echo "--- Training Tasks Complete ---"

# --- Generate Testing Tasks ---
echo "--- Generating Testing Tasks ---"
test_tasks_args=()
for i in "${dataset[@]}"; do
    for j in "${quality_type[@]}"; do
        if [ "$j" == "pure" ]; then
            test_tasks_args+=("$i" "pure" "low" "test") # level is empty string
        else
            test_tasks_args+=("$i" "$j" "low" "test")
            test_tasks_args+=("$i" "$j" "high" "test")
        fi
    done
done

echo "--- Running Testing Tasks (Max Parallel: $MAX_CONCURRENT, Delay: $INITIAL_START_DELAY s) ---"
printf "%s\n" "${test_tasks_args[@]}" |  paste -d ' ' - - - - | parallel --line-buffer --jobs "$MAX_CONCURRENT" --delay "$INITIAL_START_DELAY" --colsep ' ' run_task {1} {2} {3} {4}
echo "--- Testing Tasks Complete ---"

echo "--- All tasks finished. ---"