#!/bin/bash
# Launch script for multi-GPU training

# Default values
GPUS=${GPUS:-1}
CONFIG=${CONFIG:-""}
MODEL=${MODEL:-"gpt2"}
DATASET=${DATASET:-""}
OUTPUT_DIR=${OUTPUT_DIR:-"./output"}
STRATEGY=${STRATEGY:-"dp"}
MAX_STEPS=${MAX_STEPS:-10000}
PRECISION=${PRECISION:-"fp16"}
LR=${LR:-0.0001}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Noctua Training Launch"
echo "=========================================="
echo "GPUs: $GPUS"
echo "Model: $MODEL"
echo "Strategy: $STRATEGY"
echo "Precision: $PRECISION"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Determine launch command
if [ $GPUS -eq 1 ]; then
    # Single GPU
    echo "Launching single GPU training..."
    python scripts/train.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --output_dir "$OUTPUT_DIR" \
        --strategy "$STRATEGY" \
        --max_steps "$MAX_STEPS" \
        --precision "$PRECISION" \
        --lr "$LR" \
        ${CONFIG:+"--config" "$CONFIG"}
else
    # Multi-GPU with torchrun
    echo "Launching multi-GPU training with torchrun..."
    torchrun \
        --nproc_per_node=$GPUS \
        --nnodes=1 \
        scripts/train.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --output_dir "$OUTPUT_DIR" \
        --strategy "$STRATEGY" \
        --max_steps "$MAX_STEPS" \
        --precision "$PRECISION" \
        --lr "$LR" \
        ${CONFIG:+"--config" "$CONFIG"}
fi

echo "Training complete!"
