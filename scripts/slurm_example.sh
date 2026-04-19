#!/bin/bash
#SBATCH --job-name=noctua-training
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Load required modules
module load cuda/12.1
module load nccl/2.18
module load openmpi/4.1

# Set NCCL settings
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN
export NCCL_NET_GDR_LEVEL=5

# Training configuration
export MODEL_NAME="gpt2-large"
export STRATEGY="zero2"
export GPUS_PER_NODE=8
export TOTAL_NODES=4
export WORLD_SIZE=$((GPUS_PER_NODE * TOTAL_NODES))
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)

# Launch training
echo "========================================"
echo "Noctua Training Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "GPUs: $GPUS_PER_NODE per node"
echo "Total GPUs: $WORLD_SIZE"
echo "Model: $MODEL_NAME"
echo "Strategy: $STRATEGY"
echo "========================================"

srun python scripts/train.py \
    --model "$MODEL_NAME" \
    --strategy "$STRATEGY" \
    --world_size "$WORLD_SIZE" \
    --max_steps 100000 \
    --precision bf16 \
    --batch_size 16 \
    --output_dir "./output/${SLURM_JOB_ID}"

echo "Training complete!"
