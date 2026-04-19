"""
Logging Utilities

Provides comprehensive logging with support for:
- Multi-process logging (rank-aware)
- TensorBoard integration
- Multiple reporting backends
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime

import torch


# Global logger registry
_loggers: Dict[str, logging.Logger] = {}


def setup_logger(
    name: str = "noctua",
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
    rank: Optional[int] = None,
) -> logging.Logger:
    """
    Setup a logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to write logs to
        format_string: Custom format string
        rank: Process rank (for distributed training)
        
    Returns:
        Configured logger
    """
    # Check if already configured
    if name in _loggers:
        return _loggers[name]
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Determine rank
    if rank is None:
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = int(os.environ.get("RANK", 0))
    
    is_main_process = rank == 0
    
    # Default format
    if format_string is None:
        if is_main_process:
            format_string = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
        else:
            format_string = f"[%(asctime)s] [Rank {rank}] [%(levelname)s] [%(name)s] %(message)s"
    
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console handler (only on main process)
    if is_main_process:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None and is_main_process:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Store in registry
    _loggers[name] = logger
    
    return logger


def get_logger(name: str = "noctua") -> logging.Logger:
    """Get an existing logger."""
    if name in _loggers:
        return _loggers[name]
    return setup_logger(name)


class TrainingLogger:
    """
    Training-specific logger with metrics tracking.
    
    Integrates with TensorBoard and supports multiple
    reporting backends.
    
    Example:
        >>> logger = TrainingLogger("output/logs", ["tensorboard", "wandb"])
        >>> logger.log_metrics({"loss": 0.5, "lr": 1e-4}, step=100)
        >>> logger.log_text("Training complete!")
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        report_to: list = None,
        experiment_name: str = "noctua_experiment",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.report_to = report_to or ["tensorboard"]
        self.experiment_name = experiment_name
        
        # Setup tensorboard
        self._tensorboard_writer = None
        if "tensorboard" in self.report_to:
            from torch.utils.tensorboard import SummaryWriter
            self._tensorboard_writer = SummaryWriter(
                log_dir=str(self.log_dir / "tensorboard"),
            )
        
        # Setup wandb
        self._wandb_run = None
        if "wandb" in self.report_to:
            self._setup_wandb(config)
        
        # Console logger
        self.logger = setup_logger("training")
    
    def _setup_wandb(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize Weights & Biases."""
        try:
            import wandb
            
            wandb.init(
                project=self.experiment_name,
                dir=str(self.log_dir),
                config=config,
            )
            self._wandb_run = wandb
        except ImportError:
            self.logger.warning("wandb not installed, skipping...")
    
    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int]],
        step: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        """
        Log metrics to all configured backends.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Training step
            prefix: Prefix for metric names
        """
        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # TensorBoard
        if self._tensorboard_writer is not None and step is not None:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._tensorboard_writer.add_scalar(name, value, step)
        
        # Weights & Biases
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)
        
        # Console
        metric_str = ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                               for k, v in metrics.items())
        self.logger.info(f"Step {step}: {metric_str}")
    
    def log_text(self, text: str, step: Optional[int] = None) -> None:
        """Log text message."""
        if self._tensorboard_writer is not None and step is not None:
            self._tensorboard_writer.add_text("log", text, step)
        
        self.logger.info(text)
    
    def log_histogram(
        self,
        name: str,
        values: Union[torch.Tensor, list],
        step: int,
    ) -> None:
        """Log histogram of values."""
        if self._tensorboard_writer is not None:
            self._tensorboard_writer.add_histogram(name, values, step)
    
    def log_image(
        self,
        name: str,
        image: Union[torch.Tensor, "PIL.Image"],
        step: int,
    ) -> None:
        """Log image."""
        if self._tensorboard_writer is not None:
            self._tensorboard_writer.add_image(name, image, step)
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration."""
        if self._wandb_run is not None:
            self._wandb_run.config.update(config)
    
    def finish(self) -> None:
        """Clean up logging resources."""
        if self._tensorboard_writer is not None:
            self._tensorboard_writer.close()
        
        if self._wandb_run is not None:
            self._wandb_run.finish()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


def log_system_info() -> Dict[str, str]:
    """Log system information."""
    import platform
    import torch
    
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "gpu_count": str(torch.cuda.device_count()),
            "gpu_name": torch.cuda.get_device_name(0),
        })
    
    return info


def rank_aware_print(message: str, rank: int = 0) -> None:
    """Print message only on specified rank."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == rank:
            print(message)
    else:
        print(message)
