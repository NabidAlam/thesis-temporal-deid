#!/bin/bash

# Exit immediately if a command fails
set -e

# Set default method and dataset
METHOD=$1         # "tsp_sam" or "samurai"
DATASET=$2        # "ted", "tragic_talkers", "team_ten"

# Paths
VIDEO_DIR="videos/$DATASET"
OUTPUT_DIR="output/$METHOD/$DATASET"
RESULTS_DIR="results"
CONFIG_DIR="configs"

# Loop through all videos in the dataset
for video in "$VIDEO_DIR"/*.mp4; do
    video_name=$(basename "$video" .mp4)
    echo "Processing $video_name with $METHOD..."

    python pipeline.py \
        --method "$METHOD" \
        --input "$video" \
        --output "$OUTPUT_DIR/$video_name" \
        --config "$CONFIG_DIR/${METHOD}_config.yaml"
done

echo "De-identification pipeline complete for $METHOD on $DATASET."

# Optional: run evaluation
python eval/eval_identity.py --method "$METHOD" --dataset "$DATASET"
python eval/eval_pose.py --method "$METHOD" --dataset "$DATASET"
python eval/eval_runtime.py --method "$METHOD" --dataset "$DATASET"

echo "Evaluation completed."
