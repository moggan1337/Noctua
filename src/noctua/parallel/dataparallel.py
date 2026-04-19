"""
Data Parallel Training

Implements DistributedDataParallel (DDP) with optimizations for LLM training.
Supports gradient bucketing, overlap, and communication hiding.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from noctua.core.config import ParallelConfig


class DistributedDataParallel(nn.Module):
    """
    Enhanced DistributedDataParallel wrapper optimized for LLMs.
    
    Features:
    - Gradient bucketing for efficient communication
    - Overlapped gradient synchronization
    - Automatic device placement
    - Mixed precision support
    
    Example:
        >>> model = ModelWrapper(...)
        >>> ddp_model = DistributedDataParallel(model, config)
        >>> output = ddp_model(inputs)
    """
    
    def __init__(
        self,
        module: nn.Module,
        config: ParallelConfig,
        broadcast_buffers: bool = True,
        find_unused_parameters: bool = False,
        gradient_as_bucket_view: bool = True,
        static_graph: bool = False,
    ):
        super().__init__()
        
        self.module = module
        self.config = config
        self.world_size = config.world_size
        self.rank = config.rank
        self.local_rank = config.local_rank
        
        # DDP parameters
        self.broadcast_buffers = broadcast_buffers
        self.find_unused_parameters = find_unused_parameters
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.static_graph = static_graph
        
        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")
        
        # Move module to device
        self.module = self.module.to(self.device)
        
        # Create DDP module
        self._ddp = None
        self._setup_ddp()
        
        # Gradient tracking
        self._gradient_bucket = None
    
    def _setup_ddp(self) -> None:
        """Initialize PyTorch DDP."""
        # Set requires_grad for parameters
        for p in self.module.parameters():
            if not p.requires_grad:
                p.requires_grad = True
        
        # Create DDP wrapper
        self._ddp = torch.nn.parallel.DistributedDataParallel(
            self.module,
            device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            output_device=self.local_rank if torch.cuda.is_available() else None,
            broadcast_buffers=self.broadcast_buffers,
            find_unused_parameters=self.find_unused_parameters,
            gradient_as_bucket_view=self.gradient_as_bucket_view,
            static_graph=self.static_graph,
        )
    
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass through DDP model."""
        return self._ddp(*args, **kwargs)
    
    @property
    def no_sync(self) -> Callable:
        """Context manager to disable gradient synchronization."""
        return self._ddp.no_sync
    
    def sync_buffers(self) -> None:
        """Synchronize buffers across processes."""
        if self.broadcast_buffers:
            for buffer in self.module.buffers():
                dist.broadcast(buffer, src=0)
    
    def get_raw_model(self) -> nn.Module:
        """Get the underlying module without DDP wrapper."""
        return self.module
    
    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Get state dict from underlying module."""
        return self.module.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict: Dict[str, Any], *args, **kwargs) -> None:
        """Load state dict to underlying module."""
        self.module.load_state_dict(state_dict, *args, **kwargs)


class DataParallelTrainer:
    """
    Data Parallel training orchestrator.
    
    Manages multi-GPU/multi-node training with efficient
    data loading and gradient synchronization.
    
    Example:
        >>> trainer = DataParallelTrainer(config)
        >>> trainer.setup(model, dataset)
        >>> trainer.train()
    """
    
    def __init__(
        self,
        config: ParallelConfig,
        gradient_clip_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
    ):
        self.config = config
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[Optimizer] = None
        self.train_dataloader: Optional[DataLoader] = None
        self.ddp_model: Optional[DistributedDataParallel] = None
        
        self._is_setup = False
    
    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.config.rank == 0
    
    @property
    def world_size(self) -> int:
        """Get world size."""
        return self.config.world_size
    
    @property
    def rank(self) -> int:
        """Get current rank."""
        return self.config.rank
    
    @property
    def local_rank(self) -> int:
        """Get local rank."""
        return self.config.local_rank
    
    def setup(
        self,
        model: nn.Module,
        train_dataset,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        """
        Setup training components.
        
        Args:
            model: Model to train
            train_dataset: Training dataset
            optimizer: Optional pre-configured optimizer
        """
        self.model = model
        
        # Create distributed model
        self.ddp_model = DistributedDataParallel(
            module=model,
            config=self.config,
        )
        
        # Setup optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.AdamW(
                self.ddp_model.parameters(),
                lr=1e-4,
            )
        
        # Setup dataloader with distributed sampler
        self.train_dataloader = self._create_dataloader(train_dataset)
        
        self._is_setup = True
    
    def _create_dataloader(self, dataset) -> DataLoader:
        """Create dataloader with distributed sampling."""
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=True,
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
        )
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Single training step.
        
        Args:
            batch: Input batch
            scaler: Optional gradient scaler for mixed precision
            
        Returns:
            Tuple of (loss, scaled_loss)
        """
        # Forward pass
        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = self.ddp_model(**batch)
                loss = output["loss"] if isinstance(output, dict) else output
        else:
            output = self.ddp_model(**batch)
            loss = output["loss"] if isinstance(output, dict) else output
        
        # Scale loss for gradient accumulation
        scaled_loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        return loss, scaled_loss.item()
    
    def optimizer_step(
        self,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> None:
        """Perform optimizer step with gradient clipping."""
        if scaler is not None:
            scaler.unscale_(self.optimizer)
        
        torch.nn.utils.clip_grad_norm_(
            self.ddp_model.parameters(),
            self.gradient_clip_norm,
        )
        
        if scaler is not None:
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        if self.train_dataloader is None:
            return {"loss": 0.0}
        
        self.ddp_model.train()
        
        # Update sampler for epoch
        sampler = self.train_dataloader.sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move batch to device
            batch = {
                k: v.to(self.local_rank) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            # Training step
            loss, _ = self.train_step(batch)
            total_loss += loss.item()
            num_batches += 1
            
            # Optimizer step (at accumulation boundary)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer_step()
        
        return {
            "loss": total_loss / num_batches if num_batches > 0 else 0.0,
            "num_batches": num_batches,
        }
    
    def reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Reduce loss across all processes."""
        if self.world_size == 1:
            return loss
        
        reduced = loss.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= self.world_size
        
        return reduced
    
    def barrier(self) -> None:
        """Synchronize all processes."""
        if self.world_size > 1:
            dist.barrier()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.world_size > 1:
            dist.destroy_process_group()
