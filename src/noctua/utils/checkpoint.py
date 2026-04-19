"""
Checkpoint Utilities

Tools for saving and loading model checkpoints with
support for distributed training and model components.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    
    version: str = "1.0"
    model_name: str = ""
    training_steps: int = 0
    epoch: int = 0
    global_step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    timestamp: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CheckpointMetadata:
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def save_checkpoint(
    model: Union[nn.Module, Dict[str, nn.Module]],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    state: Optional[Dict[str, Any]] = None,
    output_dir: Union[str, Path] = "./checkpoints",
    checkpoint_name: str = "checkpoint",
    save_best: bool = False,
    max_checkpoints: int = 5,
    metadata: Optional[CheckpointMetadata] = None,
    save_model_only: bool = False,
) -> Path:
    """
    Save a training checkpoint.
    
    Args:
        model: Model or dictionary of models to save
        optimizer: Optimizer to save
        scheduler: Learning rate scheduler to save
        state: Training state dictionary
        output_dir: Output directory
        checkpoint_name: Name for checkpoint
        save_best: Only save if better than previous
        max_checkpoints: Maximum checkpoints to keep
        metadata: Checkpoint metadata
        save_model_only: Only save model, not optimizer state
        
    Returns:
        Path to saved checkpoint
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint directory
    checkpoint_dir = output_dir / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    if isinstance(model, nn.Module):
        model_path = checkpoint_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
    elif isinstance(model, dict):
        for name, module in model.items():
            if isinstance(module, nn.Module):
                model_path = checkpoint_dir / f"model_{name}.pt"
                torch.save(module.state_dict(), model_path)
    
    # Save optimizer
    if not save_model_only and optimizer is not None:
        optimizer_path = checkpoint_dir / "optimizer.pt"
        torch.save(optimizer.state_dict(), optimizer_path)
    
    # Save scheduler
    if not save_model_only and scheduler is not None:
        scheduler_path = checkpoint_dir / "scheduler.pt"
        torch.save(scheduler.state_dict(), scheduler_path)
    
    # Save training state
    if state is not None:
        state_path = checkpoint_dir / "trainer_state.pt"
        torch.save(state, state_path)
    
    # Save metadata
    if metadata is not None:
        metadata_path = checkpoint_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
    
    # Manage checkpoint limit
    _manage_checkpoint_limit(output_dir, max_checkpoints)
    
    return checkpoint_dir


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to load tensors to
        strict: Whether to strictly enforce state dict keys match
        
    Returns:
        Dictionary with loaded metadata and state
    """
    checkpoint_path = Path(checkpoint_path)
    
    result = {
        "loaded": False,
        "metadata": None,
        "model_loaded": False,
        "optimizer_loaded": False,
    }
    
    # Load metadata
    metadata_path = checkpoint_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            result["metadata"] = CheckpointMetadata.from_dict(json.load(f))
    
    # Load model
    if model is not None:
        model_path = checkpoint_path / "model.pt"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=strict)
            result["model_loaded"] = True
    
    # Load optimizer
    if optimizer is not None:
        optimizer_path = checkpoint_path / "optimizer.pt"
        if optimizer_path.exists():
            state_dict = torch.load(optimizer_path, map_location=device)
            optimizer.load_state_dict(state_dict)
            result["optimizer_loaded"] = True
    
    # Load scheduler
    if scheduler is not None:
        scheduler_path = checkpoint_path / "scheduler.pt"
        if scheduler_path.exists():
            state_dict = torch.load(scheduler_path, map_location=device)
            scheduler.load_state_dict(state_dict)
    
    # Load training state
    state_path = checkpoint_path / "trainer_state.pt"
    if state_path.exists():
        result["trainer_state"] = torch.load(state_path, map_location=device)
    
    result["loaded"] = True
    return result


def save_best_checkpoint(
    checkpoint_dir: Union[str, Path],
    current_loss: float,
    best_loss: float,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    metadata: Optional[CheckpointMetadata] = None,
) -> tuple[bool, float]:
    """
    Save checkpoint only if it's better than best.
    
    Args:
        checkpoint_dir: Directory for checkpoints
        current_loss: Current loss value
        best_loss: Best loss so far
        model: Model to save
        optimizer: Optimizer to save
        metadata: Checkpoint metadata
        
    Returns:
        Tuple of (saved, new_best_loss)
    """
    if current_loss < best_loss:
        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            output_dir=checkpoint_dir,
            checkpoint_name="checkpoint_best",
            metadata=metadata,
        )
        return True, current_loss
    
    return False, best_loss


def list_checkpoints(
    checkpoint_dir: Union[str, Path],
    pattern: str = "checkpoint_*",
) -> List[Path]:
    """List all checkpoints in a directory."""
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = sorted(
        checkpoint_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    
    return checkpoints


def get_latest_checkpoint(
    checkpoint_dir: Union[str, Path],
) -> Optional[Path]:
    """Get path to latest checkpoint."""
    checkpoints = list_checkpoints(checkpoint_dir)
    return checkpoints[0] if checkpoints else None


def delete_checkpoint(checkpoint_path: Union[str, Path]) -> None:
    """Delete a checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    
    if checkpoint_path.exists() and checkpoint_path.is_dir():
        shutil.rmtree(checkpoint_path)


def _manage_checkpoint_limit(
    checkpoint_dir: Path,
    max_checkpoints: int,
) -> None:
    """Manage checkpoint directory size."""
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    
    # Keep (max_checkpoints - 1) to leave room for new one
    for checkpoint in checkpoints[max_checkpoints - 1:]:
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)


def convert_checkpoint_to_safetensors(
    checkpoint_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Convert PyTorch checkpoint to SafeTensors format.
    
    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Output path for safetensors file
        
    Returns:
        Path to converted checkpoint
    """
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("safetensors required: pip install safetensors")
    
    checkpoint_path = Path(checkpoint_path)
    
    if output_path is None:
        output_path = checkpoint_path.with_suffix(".safetensors")
    else:
        output_path = Path(output_path)
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path / "model.pt", map_location="cpu")
    
    # Save as safetensors
    save_file(state_dict, output_path)
    
    return output_path


def create_pretrained_checkpoint(
    model: nn.Module,
    output_dir: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    tokenizer: Optional[Any] = None,
) -> Path:
    """
    Create a checkpoint suitable for pretrained model loading.
    
    Args:
        model: Model to save
        output_dir: Output directory
        config: Model config
        tokenizer: Tokenizer to save
        
    Returns:
        Path to checkpoint directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / "pytorch_model.bin"
    torch.save(model.state_dict(), model_path)
    
    # Save config
    if config is not None:
        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    
    # Save tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(str(output_dir))
    
    return output_dir
