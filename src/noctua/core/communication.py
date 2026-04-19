"""
Distributed Communication Layer

Provides process group management and communication primitives for
NCCL, GLOO, and MPI backends with unified interface.
"""

from __future__ import annotations

import os
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn

from noctua.core.config import BackendType, ParallelConfig


T = TypeVar("T")


@dataclass
class ProcessGroup:
    """
    Wrapper for torch.distributed.ProcessGroup with additional utilities.
    
    Provides unified interface for different distributed backends
    with automatic initialization and cleanup.
    """
    
    name: str
    backend: BackendType
    world_size: int
    rank: int
    local_rank: int
    _group: Optional[dist.ProcessGroup] = None
    
    @property
    def is_initialized(self) -> bool:
        """Check if process group is initialized."""
        return self._group is not None and dist.is_initialized()
    
    @property
    def group(self) -> dist.ProcessGroup:
        """Get the underlying process group."""
        if self._group is None:
            raise RuntimeError("Process group not initialized")
        return self._group
    
    def barrier(self) -> None:
        """Synchronize all processes in the group."""
        if self.is_initialized:
            dist.barrier(group=self.group)
    
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.rank == 0
    
    def local_rank_to_global_rank(self, local_rank: int) -> int:
        """Convert local rank to global rank."""
        return local_rank
    
    def global_rank_to_local_rank(self, global_rank: int) -> int:
        """Convert global rank to local rank."""
        return global_rank


class NCCLCommunicator:
    """
    NCCL-based distributed communication for GPU clusters.
    
    Optimized for NVIDIA GPUs with high-bandwidth interconnects.
    Supports custom kernels and fused operations.
    
    Example:
        >>> comm = NCCLCommunicator()
        >>> comm.setup_from_env()
        >>> tensors = comm.all_reduce([torch.randn(1024, 1024) for _ in range(4)])
        >>> comm.cleanup()
    """
    
    def __init__(
        self,
        backend: BackendType = BackendType.NCCL,
        init_method: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.backend = backend
        self.init_method = init_method
        self.timeout = timeout
        self._is_initialized = False
        self._process_groups: Dict[str, ProcessGroup] = {}
        
        # NCCL-specific settings
        self._nccl_config = {
            "NCCL_SOCKET_IFNAME": os.environ.get("NCCL_SOCKET_IFNAME", ""),
            "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
            "NCCL_IB_TIMEOUT": os.environ.get("NCCL_IB_TIMEOUT", "1"),
            "NCCL_IB_RETRY_CNT": os.environ.get("NCCL_IB_RETRY_CNT", "7"),
        }
    
    @property
    def is_initialized(self) -> bool:
        """Check if distributed is initialized."""
        return dist.is_initialized()
    
    @property
    def world_size(self) -> int:
        """Get total number of processes."""
        if not self.is_initialized:
            return 1
        return dist.get_world_size()
    
    @property
    def rank(self) -> int:
        """Get current process rank."""
        if not self.is_initialized:
            return 0
        return dist.get_rank()
    
    @property
    def local_rank(self) -> int:
        """Get local rank within node."""
        return int(os.environ.get("LOCAL_RANK", 0))
    
    @property
    def device(self) -> torch.device:
        """Get device for current process."""
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device("cpu")
    
    def setup_from_env(
        self,
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None,
        backend: BackendType = BackendType.NCCL,
    ) -> None:
        """
        Initialize distributed from environment variables.
        
        Expected environment variables:
        - WORLD_SIZE: Total number of processes
        - RANK: Current process rank (0-indexed)
        - LOCAL_RANK: Local rank within the node
        - MASTER_ADDR: Master process address
        - MASTER_PORT: Master process port
        """
        if self.is_initialized:
            return
        
        # Get distributed parameters from environment
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        master_addr = master_addr or os.environ.get("MASTER_ADDR", "localhost")
        master_port = master_port or int(os.environ.get("MASTER_PORT", "29500"))
        
        if world_size == 1:
            # Single process mode - no distributed init needed
            self._is_initialized = True
            return
        
        # Set NCCL config
        for key, value in self._nccl_config.items():
            os.environ[key] = value
        
        # Initialize distributed
        init_method = self.init_method or f"tcp://{master_addr}:{master_port}"
        backend_str = backend.value
        
        dist.init_process_group(
            backend=backend_str,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            timeout=self.timeout,
        )
        
        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        
        self._is_initialized = True
        
        # Create default process group
        default_group = ProcessGroup(
            name="default",
            backend=backend,
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            _group=dist.group.WORLD,
        )
        self._process_groups["default"] = default_group
    
    def setup_from_slurm(self, backend: BackendType = BackendType.NCCL) -> None:
        """
        Initialize distributed from SLURM environment variables.
        
        Expected SLURM variables:
        - SLURM_PROCID
        - SLURM_NTASKS
        - SLURM_LOCALID
        - SLURM_JOB_NODELIST
        - SLURM_STEP_NODELIST
        """
        if self.is_initialized:
            return
        
        # Get SLURM parameters
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        
        # Get master address from nodelist
        hostnames = os.environ.get("SLURM_STEP_NODELIST", os.environ.get("SLURM_JOB_NODELIST", "localhost"))
        
        # Extract first hostname for master
        master_addr = hostnames.split(",")[0].split("-")[0] if "," in hostnames else hostnames
        
        # Use any available port
        master_port = int(os.environ.get("SLURM_GPU_RESOLUTION_PORT", 29500)) + rank
        
        self.setup_from_env(
            master_addr=master_addr,
            master_port=master_port,
            backend=backend,
        )
    
    def create_process_group(
        self,
        name: str,
        ranks: List[int],
        backend: BackendType = BackendType.NCCL,
    ) -> ProcessGroup:
        """
        Create a new process group for subset of processes.
        
        Args:
            name: Name for the process group
            ranks: List of global ranks to include
            backend: Backend to use
            
        Returns:
            ProcessGroup object
        """
        if not self.is_initialized:
            raise RuntimeError("Distributed not initialized")
        
        pg = dist.new_group(ranks=ranks, backend=backend.value)
        
        process_group = ProcessGroup(
            name=name,
            backend=backend,
            world_size=len(ranks),
            rank=ranks[self.rank] if self.rank in ranks else -1,
            local_rank=0,
            _group=pg,
        )
        
        self._process_groups[name] = process_group
        return process_group
    
    def barrier(self, group_name: str = "default") -> None:
        """Synchronize all processes in a group."""
        if not self.is_initialized or self.world_size == 1:
            return
        
        if group_name not in self._process_groups:
            dist.barrier()
        else:
            dist.barrier(group=self._process_groups[group_name].group)
    
    def all_reduce(
        self,
        tensors: List[torch.Tensor],
        op: dist.ReduceOp = dist.ReduceOp.SUM,
        group_name: str = "default",
    ) -> List[torch.Tensor]:
        """
        Perform all-reduce operation across all processes.
        
        Args:
            tensors: List of tensors to reduce
            op: Reduction operation
            group_name: Process group name
            
        Returns:
            Reduced tensors (same object, modified in-place)
        """
        if not self.is_initialized or self.world_size == 1:
            return tensors
        
        group = self._get_group(group_name)
        
        for tensor in tensors:
            dist.all_reduce(tensor, op=op, group=group)
        
        return tensors
    
    def reduce(
        self,
        tensor: torch.Tensor,
        dst: int = 0,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
        group_name: str = "default",
    ) -> torch.Tensor:
        """Reduce tensor to single process."""
        if not self.is_initialized or self.world_size == 1:
            return tensor
        
        group = self._get_group(group_name)
        dist.reduce(tensor, dst=dst, op=op, group=group)
        return tensor
    
    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int = 0,
        group_name: str = "default",
    ) -> torch.Tensor:
        """Broadcast tensor from source to all processes."""
        if not self.is_initialized or self.world_size == 1:
            return tensor
        
        group = self._get_group(group_name)
        dist.broadcast(tensor, src=src, group=group)
        return tensor
    
    def all_gather(
        self,
        tensor: torch.Tensor,
        group_name: str = "default",
    ) -> List[torch.Tensor]:
        """Gather tensors from all processes."""
        if not self.is_initialized or self.world_size == 1:
            return [tensor]
        
        group = self._get_group(group_name)
        world_size = group.world_size if group else self.world_size
        
        output_tensors = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output_tensors, tensor, group=group)
        
        return output_tensors
    
    def gather(
        self,
        tensor: torch.Tensor,
        dst: int = 0,
        group_name: str = "default",
    ) -> List[torch.Tensor]:
        """Gather tensors to single process."""
        if not self.is_initialized or self.world_size == 1:
            return [tensor]
        
        group = self._get_group(group_name)
        world_size = group.world_size if group else self.world_size
        
        if self.rank == dst:
            output_tensors = [torch.empty_like(tensor) for _ in range(world_size)]
            dist.gather(tensor, output_tensors, dst=dst, group=group)
            return output_tensors
        else:
            dist.gather(tensor, dst=dst, group=group)
            return []
    
    def scatter(
        self,
        tensors: List[torch.Tensor],
        src: int = 0,
        group_name: str = "default",
    ) -> torch.Tensor:
        """Scatter tensors to all processes."""
        if not self.is_initialized or self.world_size == 1:
            return tensors[0] if tensors else torch.empty(0)
        
        group = self._get_group(group_name)
        output_tensor = torch.empty_like(tensors[0])
        dist.scatter(output_tensor, tensors, src=src, group=group)
        return output_tensor
    
    def send(self, tensor: torch.Tensor, dst: int, tag: int = 0) -> None:
        """Send tensor to destination process."""
        if self.is_initialized:
            dist.send(tensor, dst=dst, tag=tag)
    
    def recv(self, tensor: torch.Tensor, src: Optional[int] = None, tag: int = 0) -> torch.Tensor:
        """Receive tensor from source process."""
        if self.is_initialized:
            if src is None:
                src = dist.get_rank()
            dist.recv(tensor, src=src, tag=tag)
        return tensor
    
    def _get_group(self, name: str) -> Optional[dist.ProcessGroup]:
        """Get process group by name."""
        return self._process_groups.get(name, dist.group.WORLD)
    
    def cleanup(self) -> None:
        """Clean up distributed training."""
        if self.is_initialized:
            dist.destroy_process_group()
        self._is_initialized = False
        self._process_groups.clear()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass


class MPICommunicator:
    """
    MPI-based distributed communication for multi-node clusters.
    
    Uses mpi4py for process management and communication.
    Compatible with most HPC systems and job schedulers.
    
    Example:
        >>> comm = MPICommunicator()
        >>> comm.initialize()
        >>> tensor = comm.bcast(model_weights, root=0)
        >>> comm.finalize()
    """
    
    def __init__(self):
        self._comm = None
        self._is_initialized = False
    
    def initialize(self) -> None:
        """Initialize MPI communicator."""
        try:
            from mpi4py import MPI
            
            self._comm = MPI.COMM_WORLD
            self._is_initialized = True
        except ImportError:
            raise ImportError("mpi4py is required for MPI communicator")
    
    @property
    def world_size(self) -> int:
        """Get total number of processes."""
        if not self._is_initialized:
            return 1
        return self._comm.Get_size()
    
    @property
    def rank(self) -> int:
        """Get current process rank."""
        if not self._is_initialized:
            return 0
        return self._comm.Get_rank()
    
    @property
    def local_rank(self) -> int:
        """Get local rank within node."""
        # Get local rank via MPI info
        try:
            return int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
        except Exception:
            return 0
    
    def barrier(self) -> None:
        """Synchronize all processes."""
        if self._is_initialized:
            self._comm.Barrier()
    
    def allreduce(
        self,
        data: Any,
        op: str = "SUM",
    ) -> Any:
        """Perform allreduce operation."""
        if not self._is_initialized:
            return data
        
        from mpi4py import MPI
        
        if isinstance(data, torch.Tensor):
            # Handle PyTorch tensors
            if data.device.type == "cuda":
                data = data.cpu()
            
            op_map = {
                "SUM": MPI.SUM,
                "MAX": MPI.MAX,
                "MIN": MPI.MIN,
                "PROD": MPI.PROD,
            }
            
            result = data.clone()
            self._comm.Allreduce(MPI.IN_PLACE, result, op=op_map.get(op, MPI.SUM))
            return result
        else:
            return self._comm.allreduce(data, op=getattr(MPI, op))
    
    def bcast(self, data: Any, root: int = 0) -> Any:
        """Broadcast data from root to all processes."""
        if not self._is_initialized:
            return data
        return self._comm.bcast(data, root=root)
    
    def send(self, data: Any, dest: int, tag: int = 0) -> None:
        """Send data to destination."""
        if self._is_initialized:
            self._comm.send(data, dest=dest, tag=tag)
    
    def recv(self, source: int = 0, tag: int = 0) -> Any:
        """Receive data from source."""
        if self._is_initialized:
            return self._comm.recv(source=source, tag=tag)
        return None
    
    def finalize(self) -> None:
        """Finalize MPI."""
        if self._is_initialized:
            from mpi4py import MPI
            MPI.Finalize()
        self._is_initialized = False


def distributed_required(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to ensure distributed is initialized."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not dist.is_initialized():
            raise RuntimeError(
                f"{func.__name__} requires distributed training to be initialized. "
                "Call communicator.setup_from_env() first."
            )
        return func(*args, **kwargs)
    return wrapper


def main_process_only(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to run function only on main process."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if dist.is_initialized() and dist.get_rank() != 0:
            return None
        return func(*args, **kwargs)
    return wrapper


def local_process_only(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to run function only on local rank 0."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank != 0:
            return None
        return func(*args, **kwargs)
    return wrapper
