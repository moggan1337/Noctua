"""
Metrics Tracking Utilities

Provides tools for computing and tracking training metrics
including perplexity, throughput, and memory usage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict, deque

import torch
import numpy as np


@dataclass
class MetricsTracker:
    """
    Tracks training metrics over time.
    
    Computes running averages, handles distributed aggregation,
    and provides metric history.
    
    Example:
        >>> tracker = MetricsTracker()
        >>> tracker.update({"loss": 0.5, "accuracy": 0.8})
        >>> tracker.update({"loss": 0.4, "accuracy": 0.85})
        >>> print(tracker.get_average())  # {'loss': 0.45, 'accuracy': 0.825}
    """
    
    metrics: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    steps: List[int] = field(default_factory=list)
    current_step: int = 0
    
    # Running statistics
    _running_sums: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _running_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _window_size: int = 100
    
    # History windows
    _recent_values: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=100)))
    
    def update(
        self,
        metrics: Dict[str, Union[float, int]],
        step: Optional[int] = None,
    ) -> None:
        """
        Update with new metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Current training step
        """
        if step is not None:
            self.current_step = step
            self.steps.append(step)
        
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                # Store in history
                self.metrics[name].append(value)
                
                # Update running statistics
                self._running_sums[name] += value
                self._running_counts[name] += 1
                
                # Update window
                self._recent_values[name].append(value)
    
    def get_average(self, name: Optional[str] = None) -> Union[float, Dict[str, float]]:
        """
        Get average of a metric or all metrics.
        
        Args:
            name: Metric name, or None for all metrics
            
        Returns:
            Average value or dictionary of averages
        """
        if name is not None:
            if name in self._running_counts and self._running_counts[name] > 0:
                return self._running_sums[name] / self._running_counts[name]
            return 0.0
        
        return {
            name: self._running_sums[name] / self._running_counts[name]
            if self._running_counts[name] > 0 else 0.0
            for name in self._running_sums
        }
    
    def get_recent_average(self, name: str, window: Optional[int] = None) -> float:
        """Get average over recent values."""
        window = window or self._window_size
        values = list(self._recent_values[name])
        
        if not values:
            return 0.0
        
        return sum(values[-window:]) / len(values[-window:])
    
    def get_last(self, name: str) -> Optional[float]:
        """Get most recent value of a metric."""
        if name in self._recent_values and len(self._recent_values[name]) > 0:
            return list(self._recent_values[name])[-1]
        return None
    
    def get_min(self, name: str) -> Optional[float]:
        """Get minimum value of a metric."""
        if name in self.metrics and len(self.metrics[name]) > 0:
            return min(self.metrics[name])
        return None
    
    def get_max(self, name: str) -> Optional[float]:
        """Get maximum value of a metric."""
        if name in self.metrics and len(self.metrics[name]) > 0:
            return max(self.metrics[name])
        return None
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.steps.clear()
        self._running_sums.clear()
        self._running_counts.clear()
        self._recent_values.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metrics": dict(self.metrics),
            "steps": self.steps,
            "current_step": self.current_step,
            "averages": self.get_average(),
        }


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss
        
    Returns:
        Perplexity
    """
    return np.exp(loss)


def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """
    Compute token-level accuracy.
    
    Args:
        predictions: Predicted token IDs
        targets: Target token IDs
        ignore_index: Token to ignore in computation
        
    Returns:
        Accuracy as float
    """
    mask = targets != ignore_index
    correct = (predictions == targets) & mask
    total = mask.sum().item()
    
    if total == 0:
        return 0.0
    
    return correct.sum().item() / total


def compute_token_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> Dict[str, float]:
    """
    Compute accuracy from logits and labels.
    
    Args:
        logits: Model logits [B, S, V]
        labels: Target labels [B, S]
        ignore_index: Token to ignore
        
    Returns:
        Dictionary with accuracy metrics
    """
    # Get predictions
    predictions = logits.argmax(dim=-1)
    
    # Mask valid tokens
    mask = labels != ignore_index
    
    # Compute metrics
    correct = (predictions == labels) & mask
    total = mask.sum().item()
    
    if total == 0:
        return {"accuracy": 0.0, "token_count": 0}
    
    accuracy = correct.sum().item() / total
    
    return {
        "accuracy": accuracy,
        "token_count": total,
    }


class ThroughputTracker:
    """
    Tracks training throughput (samples/sec, tokens/sec).
    
    Example:
        >>> tracker = ThroughputTracker()
        >>> tracker.start_step()
        >>> # ... training step ...
        >>> tracker.end_step(batch_size=32, seq_length=512)
    """
    
    def __init__(
        self,
        window_size: int = 100,
        device: Optional[torch.device] = None,
    ):
        self.window_size = window_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._step_times = deque(maxlen=window_size)
        self._start_time: Optional[float] = None
        self._step_count = 0
    
    def start_step(self) -> None:
        """Mark the start of a training step."""
        self._start_time = time.time()
    
    def end_step(
        self,
        batch_size: int,
        seq_length: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Mark the end of a training step and compute throughput.
        
        Args:
            batch_size: Samples processed
            seq_length: Sequence length (for tokens/sec)
            
        Returns:
            Dictionary with throughput metrics
        """
        if self._start_time is None:
            raise RuntimeError("Must call start_step() before end_step()")
        
        elapsed = time.time() - self._start_time
        self._step_times.append(elapsed)
        
        self._step_count += 1
        
        # Compute throughput
        samples_per_sec = batch_size / elapsed if elapsed > 0 else 0
        
        metrics = {
            "samples_per_sec": samples_per_sec,
            "step_time_ms": elapsed * 1000,
            "steps_per_sec": 1 / elapsed if elapsed > 0 else 0,
        }
        
        if seq_length is not None:
            tokens_per_sec = (batch_size * seq_length) / elapsed if elapsed > 0 else 0
            metrics["tokens_per_sec"] = tokens_per_sec
        
        return metrics
    
    def get_average_throughput(self) -> Dict[str, float]:
        """Get average throughput over window."""
        if not self._step_times:
            return {"samples_per_sec": 0.0, "tokens_per_sec": 0.0}
        
        avg_time = sum(self._step_times) / len(self._step_times)
        
        return {
            "avg_samples_per_sec": 1 / avg_time if avg_time > 0 else 0,
            "avg_step_time_ms": avg_time * 1000,
        }


class MemoryTracker:
    """
    Tracks GPU/CPU memory usage during training.
    
    Example:
        >>> tracker = MemoryTracker()
        >>> tracker.start_step()
        >>> # ... training ...
        >>> tracker.end_step()
        >>> print(tracker.get_stats())
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._peak_memory: Dict[str, int] = {}
        self._start_memory: Dict[str, int] = {}
    
    def start_step(self) -> None:
        """Mark start and record memory."""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
        
        self._start_memory = self.get_current_usage()
    
    def end_step(self) -> None:
        """Mark end and update peak memory."""
        if self.device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(self.device)
            self._peak_memory["allocated"] = peak
    
    def get_current_usage(self) -> Dict[str, int]:
        """Get current memory usage in bytes."""
        if self.device.type == "cuda":
            return {
                "allocated": torch.cuda.memory_allocated(self.device),
                "reserved": torch.cuda.memory_reserved(self.device),
            }
        return {}
    
    def get_peak_usage(self) -> Dict[str, int]:
        """Get peak memory usage in bytes."""
        if self.device.type == "cuda":
            return {
                "allocated": torch.cuda.max_memory_allocated(self.device),
                "reserved": torch.cuda.max_memory_reserved(self.device),
            }
        return self._peak_memory
    
    def get_stats(self) -> Dict[str, float]:
        """Get formatted memory stats in GB."""
        current = self.get_current_usage()
        peak = self.get_peak_usage()
        
        def bytes_to_gb(b: int) -> float:
            return b / (1024 ** 3)
        
        return {
            "current_allocated_gb": bytes_to_gb(current.get("allocated", 0)),
            "peak_allocated_gb": bytes_to_gb(peak.get("allocated", 0)),
            "current_reserved_gb": bytes_to_gb(current.get("reserved", 0)),
            "peak_reserved_gb": bytes_to_gb(peak.get("reserved", 0)),
        }
    
    def reset(self) -> None:
        """Reset memory tracking."""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
        
        self._peak_memory.clear()
        self._start_memory.clear()
