"""
Model Wrapper with Mixed Precision and Flash Attention

Provides unified interface for loading and training LLMs with:
- Automatic mixed precision (FP16/BF16)
- Flash Attention integration
- Gradient checkpointing
- ZeRO optimization support
"""

from __future__ import annotations

import gc
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig
from transformers.modeling_utils import no_init_weights

from noctua.core.config import ModelConfig, NoctuaConfig, PrecisionType


# Try to import Flash Attention
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class MixedPrecisionWrapper:
    """
    Context manager for mixed precision training.
    
    Supports FP16, BF16, and FP32 precision modes with automatic
    loss scaling for FP16 stability.
    
    Example:
        >>> wrapper = MixedPrecisionWrapper(precision=PrecisionType.FP16)
        >>> with wrapper:  # doctest: +SKIP
        ...     output = model(inputs)
    """
    
    def __init__(
        self,
        precision: PrecisionType = PrecisionType.FP16,
        loss_scale: Optional[float] = None,
        initial_scale: float = 2**16,
        enabled: bool = True,
    ):
        self.precision = precision
        self.loss_scale = loss_scale
        self.initial_scale = initial_scale
        self.enabled = enabled and torch.cuda.is_available()
        
        self._scaler: Optional[GradScaler] = None
        self._ctx_manager = None
        
        if self.enabled and precision == PrecisionType.FP16:
            self._scaler = GradScaler(
                init_scale=initial_scale,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=100,
            )
    
    @property
    def use_fp16(self) -> bool:
        """Check if using FP16."""
        return self.precision == PrecisionType.FP16
    
    @property
    def use_bf16(self) -> bool:
        """Check if using BF16."""
        return self.precision == PrecisionType.BF16
    
    @property
    def use_amp(self) -> bool:
        """Check if using any AMP."""
        return self.use_fp16 or self.use_bf16
    
    def autocast(self, device_type: str = "cuda"):
        """Get autocast context manager."""
        if not self.enabled:
            return contextmanager(lambda: (yield))()
        
        dtype = torch.float16 if self.use_fp16 else torch.bfloat16
        return autocast(device_type=device_type, dtype=dtype, enabled=True)
    
    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient computation."""
        if self._scaler is not None:
            return self._scaler.scale(loss)
        return loss
    
    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients for gradient clipping."""
        if self._scaler is not None:
            self._scaler.unscale_(optimizer)
    
    def step(
        self,
        optimizer: torch.optim.Optimizer,
        *args,
        **kwargs
    ) -> Optional[float]:
        """Step the scaler after optimizer step."""
        if self._scaler is not None:
            self._scaler.step(optimizer, *args, **kwargs)
            self._scaler.update()
            return None
        optimizer.step(*args, **kwargs)
        return None
    
    def get_scale(self) -> float:
        """Get current loss scale."""
        if self._scaler is not None:
            return self._scaler.get_scale()
        return 1.0
    
    def get_growth_tracker(self) -> int:
        """Get growth iterations."""
        if self._scaler is not None:
            return self._scaler.get_growth_tracker()
        return 0
    
    def __enter__(self):
        """Enter context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        return False


class FlashAttentionWrapper:
    """
    Flash Attention integration wrapper.
    
    Provides efficient attention computation with:
    - Flash Attention 2.x support
    - Variable length sequence packing
    - Automatic backend selection (CUDA/sliding window)
    
    Example:
        >>> wrapper = FlashAttentionWrapper()
        >>> q, k, v = prepare_qkv(hidden_states)
        >>> output = wrapper.forward(q, k, v, attn_mask)
    """
    
    def __init__(
        self,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Optional[Tuple[int, int]] = None,
        use_fused_kernel: bool = True,
    ):
        self.dropout_p = dropout_p
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.window_size = window_size
        self.use_fused_kernel = use_fused_kernel and FLASH_ATTN_AVAILABLE
        
        if use_fused_kernel and not FLASH_ATTN_AVAILABLE:
            import warnings
            warnings.warn(
                "Flash Attention not available. Falling back to standard attention. "
                "Install with: pip install flash-attn --no-build-isolation"
            )
    
    @property
    def is_available(self) -> bool:
        """Check if Flash Attention is available."""
        return FLASH_ATTN_AVAILABLE
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention with Flash Attention.
        
        Args:
            q: Query tensor [B, H, S, D] or [B, S, H, D]
            k: Key tensor
            v: Value tensor
            attention_mask: Optional attention mask
            dropout_p: Dropout probability
            key_padding_mask: Optional key padding mask
            
        Returns:
            Attention output
        """
        if not self.use_fused_kernel:
            return self._standard_attention(q, k, v, attention_mask)
        
        # Ensure tensors are contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        # Flash Attention expects [B, S, H, D] format
        original_shape = q.shape
        if q.dim() == 4 and q.shape[1] == original_shape[1] if len(original_shape) > 2 else False:
            # Assume [B, H, S, D] format, convert to [B, S, H, D]
            if q.shape[1] < q.shape[2]:  # Heuristic for head dim
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
        
        # Compute attention
        output = flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )
        
        return output
    
    def forward_packed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute attention on packed sequences.
        
        Args:
            q: Packed query tensor [total_seq, H, D]
            k: Packed key tensor
            v: Packed value tensor
            cu_seqlens: Cumulative sequence lengths [B+1]
            max_seqlen: Maximum sequence length in batch
            dropout_p: Dropout probability
            
        Returns:
            Packed attention output
        """
        if not self.use_fused_kernel:
            raise RuntimeError("Packed attention requires Flash Attention")
        
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            max_seqlen,
            dropout_p=dropout_p,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
            return_attn_weights=False,
        )
        
        return output
    
    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard attention fallback."""
        scale = self.softmax_scale or (q.shape[-1] ** -0.5)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if mask is not None:
            scores = scores + mask
        
        attn_weights = F.softmax(scores, dim=-1)
        
        if self.dropout_p > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p)
        
        return torch.matmul(attn_weights, v)


class ModelWrapper(nn.Module):
    """
    Unified model wrapper for distributed LLM training.
    
    Features:
    - Automatic mixed precision (FP16/BF16)
    - Flash Attention integration
    - Gradient checkpointing
    - Unified interface for training/evaluation
    
    Example:
        >>> wrapper = ModelWrapper.from_pretrained("gpt2")
        >>> wrapper.setup_training(optimizer_config, training_config)
        >>> loss = wrapper.training_step(batch)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ModelConfig,
        precision: PrecisionType = PrecisionType.FP16,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        
        self.model = model
        self.config = config
        self.precision = precision
        self.use_flash_attention = use_flash_attention and FLASH_ATTN_AVAILABLE
        
        # Mixed precision wrapper
        self.mixed_precision = MixedPrecisionWrapper(precision=precision)
        
        # Flash attention wrapper
        self.flash_attention = FlashAttentionWrapper(
            causal=True,
            dropout_p=config.attention_dropout,
        ) if self.use_flash_attention else None
        
        # Training state
        self._is_setup = False
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        config: Optional[ModelConfig] = None,
        precision: PrecisionType = PrecisionType.FP16,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_flash_attention: bool = True,
    ) -> ModelWrapper:
        """
        Load pretrained model with automatic configuration.
        
        Args:
            model_name: HuggingFace model name or path
            config: Optional custom model config
            precision: Mixed precision mode
            device_map: Device placement strategy
            load_in_8bit: Load in 8-bit quantization
            load_in_4bit: Load in 4-bit quantization
            use_flash_attention: Use Flash Attention
            
        Returns:
            ModelWrapper instance
        """
        # Get model config
        if config is None:
            config = ModelConfig.from_pretrained(model_name)
        
        # Load model
        dtype = {
            PrecisionType.FP32: torch.float32,
            PrecisionType.FP16: torch.float16,
            PrecisionType.BF16: torch.bfloat16,
        }[precision]
        
        if load_in_8bit or load_in_4bit:
            # Use bitsandbytes for quantization
            try:
                import bitsandbytes as bnb
                
                if load_in_4bit:
                    quantization_config = BnBQuantizationConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=dtype,
                    )
                else:
                    quantization_config = BnBQuantizationConfig(
                        load_in_8bit=True,
                    )
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map=device_map,
                    torch_dtype=dtype,
                )
            except ImportError:
                raise ImportError("bitsandbytes required for 8-bit/4-bit loading")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device_map,
            )
        
        return cls(
            model=model,
            config=config,
            precision=precision,
            use_flash_attention=use_flash_attention,
        )
    
    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        precision: PrecisionType = PrecisionType.FP16,
        use_flash_attention: bool = True,
    ) -> ModelWrapper:
        """Create model from configuration."""
        hf_config = AutoConfig.from_pretrained(config.model_name)
        
        model = AutoModelForCausalLM.from_config(hf_config)
        
        return cls(
            model=model,
            config=config,
            precision=precision,
            use_flash_attention=use_flash_attention,
        )
    
    def setup_training(
        self,
        optimizer_config: Any,
        training_config: Any,
    ) -> None:
        """Setup training components."""
        self._is_setup = True
        
        # Enable gradient checkpointing if configured
        if training_config.gradient_checkpointing:
            self.enable_gradient_checkpointing()
    
    def enable_gradient_checkpointing(
        self,
        checkpoint_fn: Callable = torch.utils.checkpoint.checkpoint,
    ) -> None:
        """Enable gradient checkpointing to save memory."""
        if hasattr(self.model, "enable_gradient_checkpointing"):
            self.model.enable_gradient_checkpointing(checkpoint_fn)
        else:
            for module in self.model.modules():
                if hasattr(module, "gradient_checkpointing_enable"):
                    module.gradient_checkpointing_enable(checkpoint_fn)
    
    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        if hasattr(self.model, "disable_gradient_checkpointing"):
            self.model.disable_gradient_checkpointing()
        else:
            for module in self.model.modules():
                if hasattr(module, "gradient_checkpointing_disable"):
                    module.gradient_checkpointing_disable()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with mixed precision.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Training labels
            
        Returns:
            Dictionary with loss and outputs
        """
        with self.mixed_precision.autocast():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        
        return {
            "loss": outputs.loss if hasattr(outputs, "loss") else None,
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            "attentions": outputs.attentions if hasattr(outputs, "attentions") else None,
        }
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> torch.Tensor:
        """
        Single training step.
        
        Args:
            batch: Input batch with input_ids, attention_mask, labels
            optimizer: Optional optimizer for gradient zeroing
            
        Returns:
            Training loss
        """
        # Move batch to correct device
        if isinstance(batch, dict):
            batch = {k: v.cuda() if torch.cuda.is_available() and isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
        
        # Forward pass
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
        )
        
        loss = outputs["loss"]
        
        if loss is None:
            raise ValueError("Loss is None, check if labels are provided")
        
        return loss
    
    def eval_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Single evaluation step."""
        with torch.no_grad():
            outputs = self(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
            )
        
        return {
            "loss": outputs["loss"].item() if outputs["loss"] is not None else 0.0,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        **kwargs,
    ) -> torch.Tensor:
        """Generate text with the model."""
        self.model.eval()
        
        with torch.no_grad():
            with self.mixed_precision.autocast():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    **kwargs,
                )
        
        return outputs
    
    def get_num_params(self, trainable_only: bool = False) -> int:
        """Get number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_trainable_params(self) -> None:
        """Print summary of trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = self.get_num_trainable_params()
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Trainable percentage: {100 * trainable / total:.2f}%")
    
    def freeze_parameters(self, freeze_embeddings: bool = True) -> None:
        """Freeze model parameters."""
        for name, param in self.model.named_parameters():
            if freeze_embeddings and "embeddings" in name:
                param.requires_grad = False
            elif not param.requires_grad:
                param.requires_grad = False
    
    def unfreeze_parameters(self, pattern: Optional[str] = None) -> None:
        """Unfreeze model parameters."""
        for name, param in self.model.named_parameters():
            if pattern is None or pattern in name:
                param.requires_grad = True
    
    def get_state_dict(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Get model state dict."""
        return self.model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], *args, **kwargs) -> None:
        """Load model state dict."""
        self.model.load_state_dict(state_dict, *args, **kwargs)
    
    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.model.parameters()).device
    
    @property
    def dtype(self) -> torch.dtype:
        """Get model dtype."""
        return next(self.model.parameters()).dtype
    
    def to(self, *args, **kwargs) -> ModelWrapper:
        """Move model to device."""
        self.model = self.model.to(*args, **kwargs)
        return self
    
    def cuda(self, device: Optional[int] = None) -> ModelWrapper:
        """Move model to CUDA."""
        self.model = self.model.cuda(device)
        return self
    
    def cpu(self) -> ModelWrapper:
        """Move model to CPU."""
        self.model = self.model.cpu()
        return self
    
    def half(self) -> ModelWrapper:
        """Convert model to half precision."""
        self.model = self.model.half()
        return self
    
    def bfloat16(self) -> ModelWrapper:
        """Convert model to bfloat16."""
        self.model = self.model.to(dtype=torch.bfloat16)
        return self
    
    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to underlying model."""
        if name.startswith("_"):
            return super().__getattribute__(name)
        return getattr(self.model, name)
    
    def __repr__(self) -> str:
        return f"ModelWrapper(\n{self.model.__repr__()}\n)"
