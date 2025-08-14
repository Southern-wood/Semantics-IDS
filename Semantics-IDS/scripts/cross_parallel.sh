#!/bin/bash

# --- Basic Configuration ---

# Maximum number of parallel jobs
MAX_CONCURRENT=6
device_list=("0" "1" "2" "3" "4" "5")
MAX_CPU_PER_GPU=2
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
    echo "Try \`conda install -c conda-forge jq\` if you are using conda."
    exit 1
fi

if [ ! -f "scripts/cross.json" ]; then
    echo "Error: scripts/cross.json not found!"
    exit 1
fi

# dataset_list=("SWaT" "WADI" "HAI")
dataset_list=("SWaT")


# --- Helper function to run a single task with retries ---
run_task() {
    local dataset_name="$1"
    local train_quality="$2" # Quality used for training the model
    local train_level="$3"   # Level used for training
    local mode="$4"          # "train" or "test"
    local target_test_data="$5" # Optional

    local retries=0
    local status=1

    local local_device_list_as_string="$device_list_as_string"
    local device_list=() # Initialize as a local array
    # Safely read the string into the array, handles empty string case too
    if [[ -n "$local_device_list_as_string" ]]; then
        IFS=' ' read -r -a device_list <<< "$local_device_list_as_string"
    fi
    
    # Ditribute GPUs based on job slot
    local job_slot=${PARALLEL_JOBSLOT}
    local num_avail_gpus=${#device_list[@]}
    local base_gpus_per_job=$((num_avail_gpus / MAX_CONCURRENT))
    local remainder_gpus=$((num_avail_gpus % MAX_CONCURRENT))

    local gpus_for_this_job=$base_gpus_per_job
    if (( job_slot <= remainder_gpus )); then # Distribute the remainder GPUs
        gpus_for_this_job=$((gpus_for_this_job + 1))
    fi

    local assigned_gpu_indices_for_this_job=()
    local current_gpu_offset=0
    for (( s=1; s<job_slot; s++ )); do # calculate the offset for the current job slot
        local gpus_for_prev_job_slot=$base_gpus_per_job
        if (( s <= remainder_gpus )); then
            gpus_for_prev_job_slot=$((gpus_for_prev_job_slot + 1))
        fi
        current_gpu_offset=$((current_gpu_offset + gpus_for_prev_job_slot))
    done

    for (( i=0; i<gpus_for_this_job; i++ )); do
        # Ensure current_gpu_offset + i is within bounds of device_list
        local target_gpu_index=$((current_gpu_offset + i))
        if (( target_gpu_index < num_avail_gpus )); then
            assigned_gpu_indices_for_this_job+=("${device_list[$target_gpu_index]}")
        else
            time="[$(date '+%Y-%m-%d %H:%M:%S')]"
            echo -e "$time $FAILED Task for dataset $dataset_name, quality $train_quality, mode $mode: GPU index out of bounds. Offset: $current_gpu_offset, i: $i, num_avail_gpus: $num_avail_gpus. Configuration issue."
            return 1 # Exit with error
        fi
    done
    
    local assigned_devices_str=$(IFS=,; echo "${assigned_gpu_indices_for_this_job[*]}")
    local count_nproc=${#assigned_gpu_indices_for_this_job[@]}

    if [ -z "$assigned_devices_str" ] || [ "$count_nproc" -eq 0 ]; then
        time="[$(date '+%Y-%m-%d %H:%M:%S')]"
        echo -e "$time $FAILED Task for dataset $dataset_name, quality $train_quality, mode $mode: No GPUs assigned for job_slot ${job_slot}. num_avail_gpus: $num_avail_gpus. Configuration issue."
        return 1 # Exit with error
    fi
    local master_port=$((29500 + (job_slot - 1)))
    local cuda="CUDA_VISIBLE_DEVICES=${assigned_devices_str}"
    local fsdp_prefix="torchrun \
                    --nproc_per_node=$count_nproc \
                    --master_port=$master_port"
    # Calculate the device index based on job ID
    # Find devices where (device_index + 1) is divisible by the job ID
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $BOLD[INFO]$ENDC: Job slot $job_slot, Using GPUs: $assigned_devices_str"
    local cmd_base="$cuda \
                    $fsdp_prefix \
                    main.py \
                    --dataset \"$dataset_name\" \
                    --quality_type \"$train_quality\" \
                    --level \"$train_level\" \
                    --mode \"$mode\""
    # if there is a target_test_data 
    if [[ -n "$target_test_data" ]]; then
        cmd_base="$cmd_base --target_test_data $target_test_data"
    fi
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
    elif [[ -n "$target_test_data" ]]; then
        log_file="${LOG_DIR}/${log_file_suffix}_test_on_${target_test_data}.log"
        task_desc="Testing model from $task_desc_suffix on $target_test_data"
    else 
        task_desc="Testing model from $task_desc_suffix on its own data type"
    fi


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

# Convert device_list array to a space-separated string for reliable export
device_list_as_string="${device_list[*]}"
export device_list_as_string # Export the string

# Export other necessary variables (remove the array 'device_list' from direct export)
export LOG_DIR MAX_RETRIES RETRY_DELAY STARING SUCCESS RETRY FAILED MAX_CONCURRENT
export OMP_NUM_THREADS=$MAX_CPU_PER_GPU
export TQDM_DISABLE=1


# --- Generate Cross-Category Tasks ---
generate_cross_args() {
    for dataset in "${dataset_list[@]}"; do
        # We need test mode only
        mode="test"
        for mix_x in $(jq -r '.cross_settings | keys[]' scripts/cross.json); do
            # read cross all mix_x
            jq -c ".cross_settings.\"$mix_x\"[]" scripts/cross.json | while read -r pair; do
                add=$(echo "$pair" | jq -r '.add')
                target=$(echo "$pair" | jq -r '.target')
                echo "$dataset $mix_x low $mode ${target}_high"
            done
        done
    done
}

# --- Main Script ---
echo -e "$HEADER======= Parallel Tasks Started ========$ENDC"
echo Job count: $(echo $(generate_cross_args | wc -l))
echo Maximum concurrent jobs: $MAX_CONCURRENT
echo Initial start delay: $INITIAL_START_DELAY seconds
echo Maximum retries: $MAX_RETRIES
echo Retry delay: $RETRY_DELAY seconds
echo "==========================================="
echo "Using GPU count: ${#device_list[@]}"
echo "Device list: ${device_list[*]}"
echo "==========================================="
# Cross Tasks
generate_cross_args | parallel --line-buffer \
    --jobs $MAX_CONCURRENT \
    --delay $INITIAL_START_DELAY \
    --colsep ' ' \
    run_task {1} {2} {3} {4} {5}
echo "===== All Tasks Completed ====="