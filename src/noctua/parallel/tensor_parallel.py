"""
Tensor Parallel Training

Implements tensor parallelism for efficient distributed computation
across multiple GPUs within a node. Uses row and column parallelism
for linear layers.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from functools import wraps

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor


class TensorParallel(nn.Module):
    """
    Tensor Parallel wrapper for model layers.
    
    Splits tensors across multiple GPUs for parallel computation.
    Uses collective communication for all-reduce operations.
    
    Example:
        >>> tp = TensorParallel(model, tensor_parallel_size=4)
        >>> tp.setup()
        >>> output = tp(input)
    """
    
    def __init__(
        self,
        module: nn.Module,
        tensor_parallel_size: int = 1,
        find_unused_parameters: bool = False,
    ):
        super().__init__()
        
        self.module = module
        self.tensor_parallel_size = tensor_parallel_size
        self.find_unused_parameters = find_unused_parameters
        
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        self._is_sharded = False
    
    def setup(self) -> None:
        """Setup tensor parallel components."""
        self._shard_parameters()
        self._is_sharded = True
    
    def _shard_parameters(self) -> None:
        """Shard model parameters across tensor parallel group."""
        for name, module in self.module.named_modules():
            if isinstance(module, nn.Linear):
                self._shard_linear(module, name)
    
    def _shard_linear(self, module: nn.Linear, name: str) -> None:
        """Shard a linear layer."""
        if "attention.query" in name or "mlp.fc1" in name:
            # Column parallel - split output features
            self._column_parallel_split(module)
        elif "attention.output" in name or "mlp.fc2" in name:
            # Row parallel - split input features
            self._row_parallel_split(module)
    
    def _column_parallel_split(self, module: nn.Linear) -> None:
        """Split linear layer in column dimension (output)."""
        if module.out_features % self.tensor_parallel_size != 0:
            raise ValueError(
                f"Cannot split {module.out_features} features across "
                f"{self.tensor_parallel_size} GPUs"
            )
        
        # Split output dimension
        split_size = module.out_features // self.tensor_parallel_size
        start_idx = self.rank * split_size
        end_idx = start_idx + split_size
        
        # Create sharded weight
        sharded_weight = module.weight[start_idx:end_idx].clone()
        
        # Replace with sharded module
        new_module = nn.Linear(
            module.in_features,
            split_size,
            bias=module.bias is not None,
        )
        new_module.weight.data = sharded_weight
        if module.bias is not None:
            new_module.bias.data = module.bias[start_idx:end_idx].clone()
        
        self._replace_module(name, new_module)
    
    def _row_parallel_split(self, module: nn.Linear) -> None:
        """Split linear layer in row dimension (input)."""
        if module.in_features % self.tensor_parallel_size != 0:
            raise ValueError(
                f"Cannot split {module.in_features} features across "
                f"{self.tensor_parallel_size} GPUs"
            )
        
        # Split input dimension
        split_size = module.in_features // self.tensor_parallel_size
        start_idx = self.rank * split_size
        end_idx = start_idx + split_size
        
        # Create sharded weight
        sharded_weight = module.weight[:, start_idx:end_idx].clone()
        
        # Replace with sharded module
        new_module = nn.Linear(
            split_size,
            module.out_features,
            bias=False,
        )
        new_module.weight.data = sharded_weight
        
        self._replace_module(name, new_module)
    
    def _replace_module(self, name: str, new_module: nn.Module) -> None:
        """Replace a module in the model."""
        parts = name.split(".")
        parent = self.module
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)
    
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass with tensor parallel computation."""
        return self.module(*args, **kwargs)
    
    def all_reduce(self, input_: Tensor) -> Tensor:
        """All-reduce tensor across tensor parallel group."""
        if self.tensor_parallel_size == 1:
            return input_
        
        # All-reduce in-place
        dist.all_reduce(input_, op=dist.ReduceOp.SUM)
        return input_
    
    def all_gather(self, input_: Tensor, dim: int = -1) -> Tensor:
        """All-gather tensor across tensor parallel group."""
        if self.tensor_parallel_size == 1:
            return input_
        
        world_size = self.tensor_parallel_size
        rank = self.rank
        
        # Gather sizes
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        dist.all_gather(tensor_list, input_)
        
        # Concatenate along specified dimension
        output = torch.cat(tensor_list, dim=dim)
        return output
    
    def reduce_from_tensor_parallel(self, input_: Tensor) -> Tensor:
        """Reduce tensor to single rank in tensor parallel group."""
        if self.tensor_parallel_size == 1:
            return input_
        
        dist.all_reduce(input_, op=dist.ReduceOp.SUM)
        return input_


class ColParallel(nn.Linear):
    """
    Column-parallel linear layer.
    
    Splits the output features across tensor parallel group.
    Input is replicated, output is partitioned.
    
    Example:
        >>> layer = ColParallel(512, 1024, tp_size=4)
        >>> output = layer(input)  # output shape: [*, 1024/4]
        >>> output = layer.gather(output)  # output shape: [*, 1024]
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int = 1,
        bias: bool = True,
        gather_output: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.tp_size = tp_size
        self.gather_output = gather_output
        
        # Calculate local output size
        local_out_features = out_features // tp_size
        
        super().__init__(
            in_features,
            local_out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
    
    def forward(self, input_: Tensor) -> Tensor:
        """Forward pass."""
        output = super().forward(input_)
        
        if self.gather_output:
            return self.gather(output)
        
        return output
    
    def gather(self, input_: Tensor) -> Tensor:
        """Gather output from all tensor parallel ranks."""
        if self.tp_size == 1:
            return input_
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # All-gather
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        dist.all_gather(tensor_list, input_.contiguous())
        
        output = torch.cat(tensor_list, dim=-1)
        return output


class RowParallel(nn.Linear):
    """
    Row-parallel linear layer.
    
    Splits the input features across tensor parallel group.
    Output is computed locally, then all-reduced.
    
    Example:
        >>> layer = RowParallel(1024, 512, tp_size=4)
        >>> output = layer(input)  # output shape: [*, 512]
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int = 1,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.tp_size = tp_size
        
        # Calculate local input size
        local_in_features = in_features // tp_size
        
        super().__init__(
            local_in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
    
    def forward(self, input_: Tensor) -> Tensor:
        """Forward pass with all-reduce."""
        # Linear operation (local)
        output = super().forward(input_)
        
        # All-reduce across tensor parallel group
        if self.tp_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        
        return output


class ParallelLinear(nn.Module):
    """
    Generic parallel linear layer supporting both column and row parallelism.
    
    Can be configured for different parallelism strategies.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        parallel_type: str = "col",  # "col" or "row"
        tp_size: int = 1,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        
        if parallel_type == "col":
            self.linear = ColParallel(
                in_features,
                out_features,
                tp_size=tp_size,
                bias=bias,
                **kwargs,
            )
        else:
            self.linear = RowParallel(
                in_features,
                out_features,
                tp_size=tp_size,
                bias=bias,
                **kwargs,
            )
        
        self.tp_size = tp_size
    
    def forward(self, input_: Tensor) -> Tensor:
        """Forward pass."""
        return self.linear(input_)


# Utility functions
def tensor_parallel_reset() -> None:
    """Reset tensor parallel state."""
    pass


def copy_tensor_parallel_grads(source: Tensor, target: Tensor) -> None:
    """Copy gradients from source to target with tensor parallel handling."""
    if dist.is_initialized():
        dist.all_reduce(target, op=dist.ReduceOp.SUM)
        target.div_(dist.get_world_size())


def get_tensor_model_parallel_world_size() -> int:
    """Get tensor parallel world size from environment."""
    return int(os.environ.get("TP_WORLD_SIZE", 1))


def get_tensor_model_parallel_rank() -> int:
    """Get tensor parallel rank from environment."""
    return int(os.environ.get("TP_RANK", 0))


def get_tensor_model_parallel_group() -> Optional[dist.ProcessGroup]:
    """Get tensor parallel process group."""
    if dist.is_initialized() and hasattr(dist, "_tensor_parallel_group"):
        return dist._tensor_parallel_group
    return None
