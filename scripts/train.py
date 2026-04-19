#!/usr/bin/env python3
"""
Noctua Training Script

Example training script demonstrating various training configurations.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from noctua import NoctuaConfig, NoctuaTrainer
from noctua.core.config import ParallelStrategy, PrecisionType


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Noctua Training")
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (YAML or JSON)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name or path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Dataset path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="dp",
        choices=["dp", "zero1", "zero2", "zero3", "pp"],
        help="Parallel strategy",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="World size (number of GPUs)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
        help="Training precision",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run (no actual training)",
    )
    
    return parser.parse_args()


def create_config(args) -> NoctuaConfig:
    """Create configuration from arguments."""
    if args.config:
        # Load from file
        config = NoctuaConfig.from_yaml(args.config)
    else:
        # Create from arguments
        config = NoctuaConfig()
    
    # Override with command line arguments
    config.model.model_name = args.model
    config.training.output_dir = args.output_dir
    config.data.batch_size = args.batch_size
    config.data.dataset_path = args.dataset
    config.training.max_steps = args.max_steps
    config.training.dry_run = args.dry_run
    
    # Precision
    precision_map = {
        "fp32": PrecisionType.FP32,
        "fp16": PrecisionType.FP16,
        "bf16": PrecisionType.BF16,
    }
    config.training.precision = precision_map.get(args.precision, PrecisionType.FP16)
    
    # Parallel strategy
    strategy_map = {
        "dp": ParallelStrategy.DATA_PARALLEL,
        "zero1": ParallelStrategy.ZERO1,
        "zero2": ParallelStrategy.ZERO2,
        "zero3": ParallelStrategy.ZERO3,
        "pp": ParallelStrategy.PIPELINE_PARALLEL,
    }
    config.parallel.strategy = strategy_map.get(args.strategy, ParallelStrategy.DATA_PARALLEL)
    config.parallel.world_size = args.world_size
    
    # Optimizer
    config.optimizer.learning_rate = args.lr
    
    # Resume
    if args.resume:
        config.training.load_checkpoint = args.resume
    
    return config


def main():
    """Main training function."""
    args = parse_args()
    
    print("=" * 60)
    print("Noctua Distributed LLM Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Strategy: {args.strategy}")
    print(f"World Size: {args.world_size}")
    print(f"Precision: {args.precision}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Output Dir: {args.output_dir}")
    print("=" * 60)
    
    # Create configuration
    config = create_config(args)
    
    # Print configuration
    print("\nConfiguration:")
    print(config)
    
    if args.dry_run:
        print("\nDry run - exiting without training")
        return
    
    # Initialize trainer
    trainer = NoctuaTrainer(config)
    
    # Setup and train
    try:
        trainer.setup()
        trainer.train()
        trainer.save_model()
        print("\nTraining complete!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        raise


if __name__ == "__main__":
    main()
