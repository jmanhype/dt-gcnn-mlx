#!/bin/bash
# Example training script for DT-GCNN

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Create output directory with timestamp
OUTPUT_DIR="./experiments/$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Copy configuration to output directory
cp configs/default_config.yaml $OUTPUT_DIR/

# Run training with default configuration
python src/training/train.py \
    --data-dir ./data \
    --output-dir $OUTPUT_DIR \
    --config configs/default_config.yaml \
    --verbose \
    2>&1 | tee $OUTPUT_DIR/training.log

# Alternative: Run with command line arguments only
# python src/training/train.py \
#     --data-dir ./data \
#     --output-dir $OUTPUT_DIR \
#     --num-vertices 1723 \
#     --embedding-dim 256 \
#     --batch-size 32 \
#     --num-epochs 100 \
#     --learning-rate 0.001 \
#     --verbose

# Alternative: Resume from checkpoint
# python src/training/train.py \
#     --data-dir ./data \
#     --output-dir $OUTPUT_DIR \
#     --config configs/default_config.yaml \
#     --resume ./experiments/previous_run/checkpoints/best_model.npz \
#     --verbose