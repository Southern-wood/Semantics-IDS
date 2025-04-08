#!bin/sh

datasets=("SWaT" "WADI" "HAI")
low_quality_types=("pure" "noise" "missing" "duplicate" "delay" "mismatch")

low_quality_str=$(printf "%s " "${low_quality_types[@]}")

for dataset in "${datasets[@]}"; do
    echo "Generating low quality data for $dataset"
    python main.py \
        --dataset "$dataset" \
        --type $low_quality_str
    echo "Done generating low quality data for $dataset"
    echo "-------------------------"
done