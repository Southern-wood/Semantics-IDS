#!bin/sh

datasets=("SWaT" "WADI" "HAI")
noise_types=("pure" "noise" "missing" "duplicate" "delay")
for i in {1..10}; do
    noise_types+=("mix_$i")
done
noise_types+=("mix_all")
noise_str=$(printf "%s " "${noise_types[@]}")

for dataset in "${datasets[@]}"; do
    echo "Generating noisy data for $dataset"
    python main.py \
        --dataset "$dataset" \
        --type $noise_str
    echo "Done generating noisy data for $dataset"
    echo "-------------------------"
done