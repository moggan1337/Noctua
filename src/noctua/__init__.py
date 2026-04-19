"""
Noctua - Distributed LLM Training System

A high-performance distributed training framework for large language models,
supporting DataParallel, PipelineParallel, and ZeRO-style optimizations.
"""

__version__ = "0.1.0"
__author__ = "Noctua Team"

from noctua.core.config import NoctuaConfig
from noctua.core.trainer import NoctuaTrainer
from noctua.core.model_wrapper import ModelWrapper

__all__ = [
    "NoctuaConfig",
    "NoctuaTrainer",
    "ModelWrapper",
]
