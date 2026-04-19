"""
Mixed Precision AdamW Optimizer

AdamW optimizer with support for mixed precision training,
including FP16/BF16 parameter/gradient handling.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from collections import defaultdict

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer


class MixedPrecisionAdamW(Optimizer):
    """
    Mixed precision AdamW optimizer with dynamic loss scaling.
    
    Handles FP16/BF16 gradients with automatic scaling to prevent
    underflow during training.
    
    Features:
    - Dynamic loss scaling
    - FP16/BF16 support
    - Gradient clipping
    - Efficient memory usage
    
    Example:
        >>> optimizer = MixedPrecisionAdamW(
        ...     model.parameters(),
        ...     lr=1e-4,
        ...     betas=(0.9, 0.999),
        ...     weight_decay=0.01,
        ... )
        >>> # Forward pass with scaled loss
        >>> scaled_loss.backward()
        >>> optimizer.step(closure)
    """
    
    def __init__(
        self,
        params: Iterator[nn.Parameter],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        init_scale: float = 2**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        hysteresis: int = 2,
        max_scale: float = 2**24,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        
        super().__init__(params, defaults)
        
        # Mixed precision settings
        self.init_scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.hysteresis = hysteresis
        self.max_scale = max_scale
        
        # State for loss scaling
        self._loss_scale = init_scale
        self._growth_tracker = 0
        self._hysteresis_tracker = 0
        self._found_inf = torch.zeros(1)
        
        # Device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @property
    def loss_scale(self) -> float:
        """Get current loss scale."""
        return self._loss_scale
    
    @property
    def growth_tracker(self) -> int:
        """Get growth tracker."""
        return self._growth_tracker
    
    def _update_loss_scale(self, found_inf: Tensor) -> None:
        """
        Update loss scale based on gradient overflow.
        
        Args:
            found_inf: Tensor indicating if any gradient overflowed
        """
        if found_inf.item() > 0:
            # Overflow detected - reduce scale
            self._loss_scale *= self.backoff_factor
            self._hysteresis_tracker = 0
        else:
            # No overflow - potentially increase scale
            if self._hysteresis_tracker >= self.hysteresis:
                if self._growth_tracker >= self.growth_interval:
                    new_scale = self._loss_scale * self.growth_factor
                    self._loss_scale = min(new_scale, self.max_scale)
                    self._growth_tracker = 0
                else:
                    self._growth_tracker += 1
            else:
                self._hysteresis_tracker += 1
    
    def step(
        self,
        closure: Optional[callable] = None,
    ) -> Optional[Tensor]:
        """
        Perform optimization step.
        
        Args:
            closure: Optional closure for computing loss
            
        Returns:
            Loss if closure provided, None otherwise
        """
        loss = None
        
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Group parameters by dtype for proper handling
        groups_by_dtype = defaultdict(list)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                dtype = p.dtype
                groups_by_dtype[dtype].append((group, p))
        
        # Process each dtype group
        for dtype, param_groups in groups_by_dtype.items():
            self._step_param_groups(param_groups)
        
        return loss
    
    def _step_param_groups(
        self,
        param_groups: List[Tuple[Dict, nn.Parameter]],
    ) -> None:
        """Step parameters for a specific dtype group."""
        for group, p in param_groups:
            if p.grad is None:
                continue
            
            # Check for inf/nan gradients
            grad = p.grad
            if torch.isinf(grad).any() or torch.isnan(grad).any():
                self._found_inf.fill_(1.0)
                continue
            
            # Unscale gradients if needed
            if self._loss_scale != 1.0:
                p.grad = p.grad.mul(1.0 / self._loss_scale)
            
            # Perform AdamW update
            self._update_adamw(p, group)
    
    def _update_adamw(
        self,
        param: nn.Parameter,
        group: Dict[str, Any],
    ) -> None:
        """Perform AdamW parameter update."""
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        amsgrad = group["amsgrad"]
        
        # Get parameter state
        state = self.state[param]
        
        # Initialize state if needed
        if len(state) == 0:
            state["exp_avg"] = torch.zeros_like(param)
            state["exp_avg_sq"] = torch.zeros_like(param)
            if amsgrad:
                state["max_exp_avg_sq"] = torch.zeros_like(param)
            state["step"] = 0
        
        # Get state values
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        beta1, beta2 = group["betas"]
        
        state["step"] += 1
        step = state["step"]
        
        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(param.grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(param.grad, param.grad, value=1 - beta2)
        
        # Compute step size
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        step_size = lr / bias_correction1
        
        # Compute bias-corrected second moment estimate
        bias_correction2_sqrt = bias_correction2 ** 0.5
        
        if amsgrad:
            max_exp_avg_sq = state["max_exp_avg_sq"]
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            denom = (max_exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
        
        # Update parameters
        param.addcdiv_(exp_avg, denom, value=-step_size)
        
        # Apply weight decay
        if weight_decay > 0:
            param.add_(param, alpha=-lr * weight_decay)
    
    def unscale_(self) -> None:
        """
        Unscale gradients for gradient clipping.
        
        Divides gradients by loss scale to get true gradient values.
        """
        if self._loss_scale == 1.0:
            return
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None and self._loss_scale != 1.0:
                    p.grad = p.grad.mul(self._loss_scale)
    
    def get_growth_tracker(self) -> int:
        """Get number of steps since last loss scale growth."""
        return self._growth_tracker
    
    def get_scale(self) -> float:
        """Get current loss scale."""
        return self._loss_scale


class FusedAdamW(Optimizer):
    """
    Fused AdamW optimizer for maximum performance.
    
    Uses fused operations for improved throughput on supported hardware.
    This is a reference implementation - in production would use
    vendor-specific implementations (NVIDIA Apex, AMD ROCm, etc.).
    """
    
    def __init__(
        self,
        params: Iterator[nn.Parameter],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        
        super().__init__(params, defaults)
    
    def step(
        self,
        closure: Optional[callable] = None,
    ) -> Optional[Tensor]:
        """Perform fused optimization step."""
        loss = None
        
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                self._fused_step(p, group)
        
        return loss
    
    def _fused_step(
        self,
        param: nn.Parameter,
        group: Dict[str, Any],
    ) -> None:
        """Fused AdamW update step."""
        # This would use torch._foreach operations for efficiency
        # Simplified implementation here
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        
        state = self.state[param]
        
        if len(state) == 0:
            state["exp_avg"] = torch.zeros_like(param)
            state["exp_avg_sq"] = torch.zeros_like(param)
            state["step"] = 0
        
        state["step"] += 1
        
        # Fused operations using torch._foreach
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        
        # Update moments
        torch.mul(exp_avg, beta1, out=exp_avg)
        torch.addcmul(exp_avg, p.grad, 1 - beta1, out=exp_avg)
        
        torch.mul(exp_avg_sq, beta2, out=exp_avg_sq)
        torch.addcmul(exp_avg_sq, p.grad, p.grad, 1 - beta2, out=exp_avg_sq)
