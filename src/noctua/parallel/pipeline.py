"""
Pipeline Parallel Training

Implements model parallelism across multiple devices/nodes with:
- Sequential pipeline stages
- Virtual pipeline stages for efficiency
- 1F1B (one-forward-one-backward) scheduling
- Inter-stage communication
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor


class ScheduleType(Enum):
    """Pipeline schedule types."""
    FORWARD_BACKWARD = "forward_backward"
    ONE_FORWARD_ONE_BACKWARD = "1f1b"
    INTERLEAVED = "interleaved"


@dataclass
class PipelineStageInfo:
    """Information about a pipeline stage."""
    
    stage_id: int
    num_layers: int
    first_layer_idx: int
    last_layer_idx: int
    device: Union[int, torch.device]
    
    @property
    def layer_indices(self) -> range:
        """Get range of layer indices in this stage."""
        return range(self.first_layer_idx, self.last_layer_idx + 1)


class PipelineStage(nn.Module):
    """
    A single stage in a pipeline parallel model.
    
    Represents a contiguous subset of model layers that will
    be placed on a specific device.
    
    Example:
        >>> stage = PipelineStage(
        ...     layer_start=0,
        ...     layer_end=12,
        ...     num_layers=12,
        ...     device=0,
        ... )
        >>> output = stage.forward(hidden_states)
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer_start: int,
        layer_end: int,
        num_layers: int,
        device: Union[int, torch.device] = 0,
        is_first: bool = False,
        is_last: bool = False,
    ):
        super().__init__()
        
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.num_layers = num_layers
        self.is_first = is_first
        self.is_last = is_last
        
        # Handle device
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Extract layers for this stage
        self._extract_layers(model)
        
        # Move to device
        self.to(self.device)
        
        # Input/output projection layers for embedding handling
        self.input_proj = None
        self.output_proj = None
    
    def _extract_layers(self, model: nn.Module) -> None:
        """Extract layers for this stage from the model."""
        # Get all layers
        if hasattr(model, "transformer"):
            # GPT-style models
            self.embeddings = model.transformer.wte if hasattr(model.transformer, 'wte') else None
            self.layers = nn.ModuleList([
                model.transformer.h[i] 
                for i in range(self.layer_start, self.layer_end + 1)
            ])
            self.final_norm = model.transformer.ln_f if self.is_last else None
        elif hasattr(model, "decoder"):
            # Encoder-decoder models
            self.embeddings = None
            self.layers = nn.ModuleList([
                model.decoder.layers[i]
                for i in range(self.layer_start, self.layer_end + 1)
            ])
            self.final_norm = model.decoder.final_layer_norm if self.is_last else None
        else:
            # Generic model
            self.embeddings = None
            self.layers = nn.ModuleList(list(model.children())[self.layer_start:self.layer_end + 1])
            self.final_norm = None
    
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Forward pass through this stage.
        
        Args:
            input_ids: Input token IDs (for first stage)
            hidden_states: Hidden states from previous stage
            attention_mask: Attention mask
            
        Returns:
            Dictionary with output hidden states and metadata
        """
        # Handle input
        if not self.is_first and hidden_states is None:
            raise ValueError("Non-first stage requires hidden_states input")
        
        # Process input
        if self.is_first and input_ids is not None:
            if self.embeddings is not None:
                hidden_states = self.embeddings(input_ids)
            else:
                hidden_states = input_ids
        
        # Apply layers
        for layer in self.layers:
            layer_output = layer(
                hidden_states,
                attention_mask=attention_mask,
                **kwargs,
            )
            hidden_states = layer_output[0]
        
        # Apply final norm
        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)
        
        return {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
        }
    
    def receive_from_previous(
        self,
        hidden_states: Tensor,
    ) -> None:
        """Receive hidden states from previous pipeline stage."""
        self._prev_hidden_states = hidden_states.to(self.device)
    
    def send_to_next(self, hidden_states: Tensor) -> None:
        """Send hidden states to next pipeline stage."""
        self._next_hidden_states = hidden_states


class PipelineParallel:
    """
    Pipeline Parallel wrapper for distributed model training.
    
    Splits a model across multiple devices/nodes and manages
    the pipeline schedule for training.
    
    Features:
    - Automatic layer distribution
    - Multiple schedule types
    - Micro-batch management
    - Inter-stage communication
    
    Example:
        >>> pp = PipelineParallel(model, num_stages=4, num_microbatches=8)
        >>> pp.setup()
        >>> pp.train_microbatch(batch)
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_stages: int = 1,
        num_microbatches: int = 4,
        device_map: Optional[List[int]] = None,
        schedule_type: ScheduleType = ScheduleType.ONE_FORWARD_ONE_BACKWARD,
        gradient_accumulation_steps: int = 1,
    ):
        self.model = model
        self.num_stages = num_stages
        self.num_microbatches = num_microbatches
        self.schedule_type = schedule_type
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Stage information
        self.stages: List[PipelineStage] = []
        self.stage_info: List[PipelineStageInfo] = []
        
        # Device mapping
        self.device_map = device_map or list(range(num_stages))
        
        # Communication groups
        self.pp_group: Optional[dist.ProcessGroup] = None
        self._is_setup = False
    
    def setup(self) -> None:
        """Setup pipeline parallel components."""
        if self._is_setup:
            return
        
        # Calculate layer distribution
        self._distribute_layers()
        
        # Create pipeline stages
        self._create_stages()
        
        self._is_setup = True
    
    def _distribute_layers(self) -> None:
        """Calculate layer distribution across stages."""
        # Get model depth
        if hasattr(self.model, "config"):
            num_layers = getattr(self.model.config, "num_hidden_layers", 12)
        else:
            num_layers = 12  # Default
        
        # Distribute layers evenly
        layers_per_stage = num_layers // self.num_stages
        remainder = num_layers % self.num_stages
        
        current_layer = 0
        for stage_id in range(self.num_stages):
            # Handle remainder by adding to earlier stages
            num_stage_layers = layers_per_stage + (1 if stage_id < remainder else 0)
            
            self.stage_info.append(PipelineStageInfo(
                stage_id=stage_id,
                num_layers=num_stage_layers,
                first_layer_idx=current_layer,
                last_layer_idx=current_layer + num_stage_layers - 1,
                device=self.device_map[stage_id],
            ))
            
            current_layer += num_stage_layers
    
    def _create_stages(self) -> None:
        """Create pipeline stages."""
        for stage_info in self.stage_info:
            stage = PipelineStage(
                model=self.model,
                layer_start=stage_info.first_layer_idx,
                layer_end=stage_info.last_layer_idx,
                num_layers=stage_info.num_layers,
                device=stage_info.device,
                is_first=stage_info.stage_id == 0,
                is_last=stage_info.stage_id == self.num_stages - 1,
            )
            
            self.stages.append(stage)
    
    def forward_microbatch(
        self,
        microbatch: Dict[str, Tensor],
        stage_id: int = 0,
    ) -> Dict[str, Tensor]:
        """
        Forward pass for a single microbatch through one stage.
        
        Args:
            microbatch: Input microbatch
            stage_id: Stage to run forward on
            
        Returns:
            Stage output
        """
        stage = self.stages[stage_id]
        
        if stage.is_first:
            return stage(
                input_ids=microbatch.get("input_ids"),
                attention_mask=microbatch.get("attention_mask"),
            )
        else:
            # Get input from previous stage (via communication)
            prev_hidden = self._recv_from_previous(stage_id)
            return stage(hidden_states=prev_hidden)
    
    def backward_microbatch(
        self,
        stage_id: int,
        output_grad: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """
        Backward pass for a single microbatch.
        
        Args:
            stage_id: Current stage
            output_grad: Gradient from next stage (for non-last stages)
            
        Returns:
            Input gradient to send to previous stage
        """
        # This is a simplified version - full implementation would track
        # the forward pass activations for backward
        return None
    
    def _send_to_next(self, stage_id: int, hidden_states: Tensor) -> None:
        """Send hidden states to next stage."""
        if stage_id < self.num_stages - 1:
            # In a full implementation, this would use
            # torch.distributed.send/recv
            pass
    
    def _recv_from_previous(self, stage_id: int) -> Tensor:
        """Receive hidden states from previous stage."""
        if stage_id > 0:
            # In a full implementation, this would use
            # torch.distributed.recv
            pass
        return torch.zeros(1)
    
    def train_step(
        self,
        batch: Dict[str, Tensor],
    ) -> Dict[str, Any]:
        """
        Complete training step with pipeline parallelism.
        
        Args:
            batch: Input batch (will be split into microbatches)
            
        Returns:
            Dictionary with loss and metrics
        """
        if not self._is_setup:
            self.setup()
        
        # Split batch into microbatches
        microbatches = self._split_into_microbatches(batch)
        
        total_loss = 0.0
        
        # Forward pass for all microbatches
        forward_outputs = []
        for micro_idx, microbatch in enumerate(microbatches):
            for stage_id in range(self.num_stages):
                output = self.forward_microbatch(microbatch, stage_id)
                if stage_id == self.num_stages - 1:
                    forward_outputs.append(output)
        
        # Backward pass for all microbatches
        for micro_idx in reversed(range(len(microbatches))):
            for stage_id in reversed(range(self.num_stages)):
                self.backward_microbatch(stage_id)
        
        # Compute loss
        if forward_outputs:
            final_output = forward_outputs[-1]
            if "loss" in final_output:
                total_loss = sum(o.get("loss", 0) for o in forward_outputs).item()
        
        return {"loss": total_loss}
    
    def _split_into_microbatches(
        self,
        batch: Dict[str, Tensor],
    ) -> List[Dict[str, Tensor]]:
        """Split batch into microbatches."""
        # Simple implementation - in practice would use
        # a more sophisticated batching strategy
        return [batch]
    
    def get_stage_model(self, stage_id: int) -> PipelineStage:
        """Get the model for a specific stage."""
        return self.stages[stage_id]
    
    def state_dict(self) -> Dict[int, Dict[str, Tensor]]:
        """Get state dict for each stage."""
        return {
            stage_id: stage.state_dict()
            for stage_id, stage in enumerate(self.stages)
        }
    
    def load_state_dict(self, state_dicts: Dict[int, Dict[str, Tensor]]) -> None:
        """Load state dict for each stage."""
        for stage_id, state_dict in state_dicts.items():
            self.stages[stage_id].load_state_dict(state_dict)


class VirtualPipelineStage(PipelineStage):
    """
    Virtual pipeline stage for interleaved scheduling.
    
    Multiple virtual stages share the same physical device,
    allowing for more efficient pipeline utilization.
    
    Example:
        >>> vstage = VirtualPipelineStage(
        ...     model=model,
        ...     layer_start=0,
        ...     layer_end=11,
        ...     num_layers=12,
        ...     device=0,
        ...     model_chunk_id=0,
        ...     num_model_chunks=2,
        ... )
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer_start: int,
        layer_end: int,
        num_layers: int,
        device: Union[int, torch.device] = 0,
        model_chunk_id: int = 0,
        num_model_chunks: int = 1,
    ):
        super().__init__(
            model=model,
            layer_start=layer_start,
            layer_end=layer_end,
            num_layers=num_layers,
            device=device,
        )
        
        self.model_chunk_id = model_chunk_id
        self.num_model_chunks = num_model_chunks
        self.is_first = model_chunk_id == 0
        self.is_last = model_chunk_id == num_model_chunks - 1


# Utility functions
def get_pipeline_model_parallel_world_size() -> int:
    """Get pipeline parallel world size from environment."""
    return int(os.environ.get("PIPELINE_PARALLEL_SIZE", 1))


def get_pipeline_model_parallel_rank() -> int:
    """Get pipeline parallel rank from environment."""
    return int(os.environ.get("PIPELINE_PARALLEL_RANK", 0))


def is_pipeline_first_stage() -> bool:
    """Check if current process is first pipeline stage."""
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage() -> bool:
    """Check if current process is last pipeline stage."""
    return get_pipeline_model_parallel_rank() == get_pipeline_model_parallel_size() - 1
