"""
Noctua Trainer - Main Training Loop

Provides distributed training loop with:
- Multi-GPU/node support via DataParallel/PipelineParallel
- ZeRO optimization
- Mixed precision training
- Comprehensive logging and checkpointing
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from noctua.core.config import NoctuaConfig, ParallelConfig, PrecisionType, ParallelStrategy
from noctua.core.model_wrapper import ModelWrapper
from noctua.core.communication import NCCLCommunicator, ProcessGroup
from noctua.optimizers.zero import ZeroOptimizer, ZeroReduicer
from noctua.utils.logging import setup_logger


@dataclass
class TrainingState:
    """
    Tracks the current state of training.
    
    Used for checkpointing and resuming training.
    """
    
    global_step: int = 0
    epoch: int = 0
    total_loss: float = 0.0
    loss_count: int = 0
    best_loss: float = float("inf")
    last_eval_loss: float = float("inf")
    last_save_time: float = field(default_factory=time.time)
    learning_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "total_loss": self.total_loss,
            "loss_count": self.loss_count,
            "best_loss": self.best_loss,
            "last_eval_loss": self.last_eval_loss,
            "last_save_time": self.last_save_time,
            "learning_rate": self.learning_rate,
        }
    
    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> TrainingState:
        """Create from dictionary."""
        return cls(**{k: v for k, v in state_dict.items() if k in cls.__dataclass_fields__})
    
    @property
    def avg_loss(self) -> float:
        """Get average loss."""
        if self.loss_count == 0:
            return float("inf")
        return self.total_loss / self.loss_count


class CheckpointManager:
    """Manages model checkpointing and loading."""
    
    def __init__(
        self,
        output_dir: str,
        save_total_limit: int = 3,
        save_strategy: str = "steps",
        save_steps: int = 5000,
        model_name: str = "noctua_model",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_total_limit = save_total_limit
        self.save_strategy = save_strategy
        self.save_steps = save_steps
        self.model_name = model_name
        self._saved_checkpoints: List[Path] = []
    
    def save_checkpoint(
        self,
        model: ModelWrapper,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
        state: Optional[TrainingState] = None,
        step: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        is_main_process: bool = True,
    ) -> Optional[Path]:
        """
        Save training checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            scheduler: Optional scheduler to save
            state: Training state
            step: Current training step
            metadata: Additional metadata
            is_main_process: Only save on main process
            
        Returns:
            Path to saved checkpoint or None
        """
        if not is_main_process:
            return None
        
        # Create checkpoint path
        checkpoint_name = f"checkpoint-{step}"
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = checkpoint_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        
        # Save optimizer
        optimizer_path = checkpoint_dir / "optimizer.pt"
        torch.save(optimizer.state_dict(), optimizer_path)
        
        # Save scheduler if provided
        if scheduler is not None:
            scheduler_path = checkpoint_dir / "scheduler.pt"
            torch.save(scheduler.state_dict(), scheduler_path)
        
        # Save training state
        if state is not None:
            state_path = checkpoint_dir / "trainer_state.json"
            import json
            with open(state_path, "w") as f:
                json.dump(state.to_dict(), f, indent=2)
        
        # Save metadata
        if metadata is not None:
            metadata_path = checkpoint_dir / "metadata.json"
            import json
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        
        # Manage checkpoint limit
        self._saved_checkpoints.append(checkpoint_dir)
        if len(self._saved_checkpoints) > self.save_total_limit:
            oldest = self._saved_checkpoints.pop(0)
            if oldest.exists():
                import shutil
                shutil.rmtree(oldest)
        
        return checkpoint_dir
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: ModelWrapper,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> TrainingState:
        """Load checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        # Load model
        model_path = checkpoint_path / "model.pt"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
        
        # Load optimizer
        if optimizer is not None:
            optimizer_path = checkpoint_path / "optimizer.pt"
            if optimizer_path.exists():
                optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
        
        # Load scheduler
        if scheduler is not None:
            scheduler_path = checkpoint_path / "scheduler.pt"
            if scheduler_path.exists():
                scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
        
        # Load training state
        state = TrainingState()
        state_path = checkpoint_path / "trainer_state.json"
        if state_path.exists():
            import json
            with open(state_path) as f:
                state = TrainingState.from_dict(json.load(f))
        
        return state
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        if not self.output_dir.exists():
            return None
        
        checkpoints = sorted(
            self.output_dir.glob("checkpoint-*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        
        return checkpoints[0] if checkpoints else None


class NoctuaTrainer:
    """
    Main trainer class for distributed LLM training.
    
    Features:
    - Multi-GPU and multi-node support
    - DataParallel and PipelineParallel strategies
    - ZeRO-1/2/3 optimization
    - Mixed precision (FP16/BF16) training
    - Comprehensive logging and checkpointing
    - Learning rate scheduling
    
    Example:
        >>> config = NoctuaConfig.from_yaml("config.yaml")
        >>> trainer = NoctuaTrainer(config)
        >>> trainer.setup()
        >>> trainer.train()
    """
    
    def __init__(
        self,
        config: NoctuaConfig,
        model: Optional[ModelWrapper] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
    ):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Communication
        self.communicator: Optional[NCCLCommunicator] = None
        
        # Training components
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.grad_scaler: Optional[torch.cuda.amp.GradScaler] = None
        
        # State tracking
        self.state = TrainingState()
        self.checkpoint_manager: Optional[CheckpointManager] = None
        
        # Logging
        self.logger = setup_logger("noctua")
        self.writer: Optional[SummaryWriter] = None
        
        # Dataloaders
        self.train_dataloader: Optional[DataLoader] = None
        self.eval_dataloader: Optional[DataLoader] = None
        
        # Flags
        self._is_setup = False
        self._is_training = False
        self._should_stop = False
    
    def setup(self) -> None:
        """Initialize all training components."""
        if self._is_setup:
            return
        
        self.logger.info("Setting up Noctua Trainer...")
        
        # Initialize distributed communication
        self._setup_communication()
        
        # Setup model
        self._setup_model()
        
        # Setup data
        self._setup_data()
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Setup scheduler
        self._setup_scheduler()
        
        # Setup mixed precision
        self._setup_mixed_precision()
        
        # Setup checkpointing
        self._setup_checkpointing()
        
        # Setup logging
        self._setup_logging()
        
        # Load checkpoint if specified
        if self.config.training.load_checkpoint:
            self._load_checkpoint(self.config.training.load_checkpoint)
        
        self._is_setup = True
        self.logger.info("Setup complete!")
    
    def _setup_communication(self) -> None:
        """Initialize distributed communication."""
        self.communicator = NCCLCommunicator()
        
        # Check if we should initialize distributed
        if "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1:
            self.communicator.setup_from_env(
                master_addr=self.config.master_addr,
                master_port=self.config.master_port,
                backend=self.config.parallel.backend,
            )
            self.logger.info(
                f"Distributed initialized: rank={self.communicator.rank}, "
                f"world_size={self.communicator.world_size}"
            )
    
    def _setup_model(self) -> None:
        """Setup model for training."""
        if self.model is None:
            self.logger.info(f"Loading model: {self.config.model.model_name}")
            self.model = ModelWrapper.from_pretrained(
                model_name=self.config.model.model_name,
                config=self.config.model,
                precision=self.config.training.precision,
                use_flash_attention=self.config.use_flash_attention,
            )
        
        # Move model to device
        device = self.config.get_device()
        self.model = self.model.to(device)
        
        # Setup training mode
        self.model.train()
        
        # Enable gradient checkpointing if configured
        if self.config.training.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()
        
        # Print parameter info
        if self.config.is_main_process():
            total_params = self.model.get_num_params()
            trainable_params = self.model.get_num_trainable_params()
            self.logger.info(f"Total parameters: {total_params:,}")
            self.logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def _setup_data(self) -> None:
        """Setup dataloaders."""
        from noctua.utils.data import create_dataloader
        
        if self.train_dataset is not None:
            self.train_dataloader = create_dataloader(
                dataset=self.train_dataset,
                batch_size=self.config.data.get_batch_size_per_device(
                    self.config.parallel.world_size
                ),
                num_workers=self.config.data.num_workers,
                shuffle=True,
                pin_memory=self.config.data.pin_memory,
                drop_last=True,
            )
        
        if self.eval_dataset is not None:
            self.eval_dataloader = create_dataloader(
                dataset=self.eval_dataset,
                batch_size=self.config.data.eval_batch_size,
                num_workers=self.config.data.num_workers,
                shuffle=False,
                pin_memory=self.config.data.pin_memory,
                drop_last=False,
            )
    
    def _setup_optimizer(self) -> None:
        """Setup optimizer."""
        optimizer_config = self.config.optimizer
        
        # Create optimizer
        if optimizer_config.optimizer_type.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                **optimizer_config.to_optimizer_kwargs(),
            )
        elif optimizer_config.optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                **optimizer_config.to_optimizer_kwargs(),
            )
        elif optimizer_config.optimizer_type.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=optimizer_config.learning_rate,
                momentum=0.9,
                weight_decay=optimizer_config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_config.optimizer_type}")
        
        # Wrap with ZeRO if configured
        if self.config.parallel.strategy in [
            ParallelStrategy.ZERO1,
            ParallelStrategy.ZERO2,
            ParallelStrategy.ZERO3,
        ]:
            self._setup_zero_optimizer()
    
    def _setup_zero_optimizer(self) -> None:
        """Setup ZeRO optimizer wrapper."""
        from noctua.optimizers.zero import ZeroDistributedOptimizer
        
        zero_stage = self.config.parallel.zero_stage
        
        self.optimizer = ZeroDistributedOptimizer(
            optimizer=self.optimizer,
            named_parameters=self.model.named_parameters(),
            level=zero_stage,
            device=str(self.config.get_device()),
        )
    
    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        scheduler_config = self.config.optimizer
        
        if scheduler_config.lr_scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.max_steps,
                eta_min=scheduler_config.min_lr,
            )
        elif scheduler_config.lr_scheduler_type == "linear":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.config.training.max_steps,
            )
        elif scheduler_config.lr_scheduler_type == "constant":
            self.scheduler = optim.lr_scheduler.ConstantLR(
                self.optimizer,
                total_iters=0,
            )
        elif scheduler_config.lr_scheduler_type == "warmup_cosine":
            # Custom warmup + cosine decay
            warmup_steps = self.config.training.warmup_steps
            max_steps = self.config.training.max_steps
            
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                progress = (step - warmup_steps) / (max_steps - warmup_steps)
                return scheduler_config.min_lr + (1 - scheduler_config.min_lr) * 0.5 * (1 + progress)
            
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, 
                lambda _: 1.0
            )
    
    def _setup_mixed_precision(self) -> None:
        """Setup mixed precision training."""
        if self.config.training.precision == PrecisionType.FP16:
            self.grad_scaler = torch.cuda.amp.GradScaler(
                init_scale=self.config.optimizer.initial_loss_scale,
                growth_factor=self.config.optimizer.loss_scale_factor,
                backoff_factor=0.5,
                growth_interval=self.config.optimizer.loss_scale_window,
            )
        else:
            self.grad_scaler = None
    
    def _setup_checkpointing(self) -> None:
        """Setup checkpoint manager."""
        self.checkpoint_manager = CheckpointManager(
            output_dir=self.config.training.output_dir,
            save_total_limit=self.config.training.save_total_limit,
            save_strategy=self.config.training.save_strategy,
            save_steps=self.config.training.save_steps,
        )
    
    def _setup_logging(self) -> None:
        """Setup logging and tensorboard."""
        if self.config.is_main_process():
            # Setup tensorboard
            log_dir = self.config.training.logging_dir or (
                Path(self.config.training.output_dir) / "logs"
            )
            self.writer = SummaryWriter(log_dir=log_dir)
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Training metrics and final state
        """
        if not self._is_setup:
            self.setup()
        
        self._is_training = True
        self.logger.info("Starting training...")
        
        # Training metrics
        metrics = {
            "total_steps": 0,
            "total_epochs": 0,
            "training_losses": [],
            "evaluation_losses": [],
            "best_loss": float("inf"),
        }
        
        try:
            while not self._should_stop:
                # Training epoch
                epoch_metrics = self._train_epoch()
                metrics["training_losses"].extend(epoch_metrics.get("losses", []))
                
                # Evaluation
                if self.eval_dataloader is not None:
                    eval_metrics = self._evaluate()
                    metrics["evaluation_losses"].append(eval_metrics.get("loss", float("inf")))
                
                # Check stopping conditions
                if self.state.global_step >= self.config.training.max_steps:
                    self.logger.info("Reached max steps, stopping training.")
                    break
                
                if self.config.training.max_epochs is not None:
                    self.state.epoch += 1
                    if self.state.epoch >= self.config.training.max_epochs:
                        self.logger.info("Reached max epochs, stopping training.")
                        break
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user.")
        
        finally:
            self._is_training = False
            self._cleanup()
        
        # Save final checkpoint
        self._save_checkpoint()
        
        return metrics
    
    def _train_epoch(self) -> Dict[str, Any]:
        """Train for one epoch."""
        if self.train_dataloader is None:
            return {"losses": []}
        
        losses = []
        epoch_iterator = self.train_dataloader
        
        if self.config.is_main_process():
            epoch_iterator = tqdm(
                epoch_iterator,
                desc=f"Epoch {self.state.epoch}",
                disable=False,
            )
        
        for batch in epoch_iterator:
            # Training step
            loss = self._training_step(batch)
            losses.append(loss)
            
            # Update state
            self.state.total_loss += loss
            self.state.loss_count += 1
            self.state.global_step += 1
            
            # Logging
            if self.state.global_step % self.config.training.logging_steps == 0:
                self._log_metrics({"train/loss": loss})
            
            # Evaluation
            if (
                self.eval_dataloader is not None
                and self.config.training.do_eval
                and self.state.global_step % self.config.training.eval_steps == 0
            ):
                eval_metrics = self._evaluate()
                self._log_metrics({"eval/loss": eval_metrics.get("loss", 0)})
                
                # Update best loss
                if eval_metrics.get("loss", float("inf")) < self.state.best_loss:
                    self.state.best_loss = eval_metrics.get("loss", float("inf"))
            
            # Checkpointing
            if (
                self.state.global_step % self.config.training.save_steps == 0
                and self.state.global_step > 0
            ):
                self._save_checkpoint()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                self.state.learning_rate = self.scheduler.get_last_lr()[0]
        
        return {"losses": losses}
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        # Move batch to device
        batch = {
            k: v.to(self.config.get_device()) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        
        # Forward pass with mixed precision
        if self.grad_scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch.get("labels"),
                )
            
            loss = outputs["loss"]
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.parallel.gradient_accumulation_steps
            
            # Backward pass
            self.grad_scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (self.state.global_step + 1) % self.config.parallel.gradient_accumulation_steps == 0:
                # Unscale gradients for clipping
                self.grad_scaler.unscale_(self.optimizer)
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.parallel.gradient_clip_norm,
                )
                
                # Optimizer step
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad()
        else:
            # Standard forward/backward
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
            )
            
            loss = outputs["loss"] / self.config.parallel.gradient_accumulation_steps
            loss.backward()
            
            if (self.state.global_step + 1) % self.config.parallel.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.parallel.gradient_clip_norm,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        return loss.item() * self.config.parallel.gradient_accumulation_steps
    
    def _evaluate(self) -> Dict[str, float]:
        """Run evaluation."""
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {
                    k: v.to(self.config.get_device()) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch.get("labels"),
                )
                
                if outputs["loss"] is not None:
                    total_loss += outputs["loss"].item()
                    num_batches += 1
        
        self.model.train()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {"loss": avg_loss}
    
    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to tensorboard and console."""
        # Console logging
        metric_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self.logger.info(f"Step {self.state.global_step}: {metric_str}")
        
        # Tensorboard logging
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, self.state.global_step)
            
            # Log learning rate
            if self.state.learning_rate > 0:
                self.writer.add_scalar(
                    "train/learning_rate", 
                    self.state.learning_rate, 
                    self.state.global_step
                )
    
    def _save_checkpoint(self) -> None:
        """Save current checkpoint."""
        if self.checkpoint_manager is not None:
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                state=self.state,
                step=self.state.global_step,
                is_main_process=self.config.is_main_process(),
            )
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint."""
        if self.checkpoint_manager is not None:
            self.state = self.checkpoint_manager.load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        # Close tensorboard writer
        if self.writer is not None:
            self.writer.close()
        
        # Cleanup distributed
        if self.communicator is not None:
            self.communicator.cleanup()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def stop(self) -> None:
        """Signal training to stop."""
        self._should_stop = True
    
    def save_model(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save model to disk."""
        path = Path(path or self.config.training.output_dir) / "final_model"
        path.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.model.state_dict(), path / "model.pt")
        
        # Save config
        import json
        with open(path / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        self.logger.info(f"Model saved to {path}")
