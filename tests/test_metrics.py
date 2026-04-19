"""
Tests for metrics utilities.
"""

import pytest
import torch

from noctua.utils.metrics import (
    MetricsTracker,
    compute_perplexity,
    compute_accuracy,
    ThroughputTracker,
    MemoryTracker,
)


class TestMetricsTracker:
    """Test MetricsTracker class."""
    
    def test_creation(self):
        """Test tracker creation."""
        tracker = MetricsTracker()
        assert tracker is not None
        assert len(tracker.metrics) == 0
    
    def test_update(self):
        """Test metric updates."""
        tracker = MetricsTracker()
        
        tracker.update({"loss": 0.5, "accuracy": 0.8})
        tracker.update({"loss": 0.4, "accuracy": 0.85})
        
        assert "loss" in tracker.metrics
        assert "accuracy" in tracker.metrics
        assert len(tracker.metrics["loss"]) == 2
    
    def test_get_average(self):
        """Test average computation."""
        tracker = MetricsTracker()
        
        tracker.update({"loss": 0.5})
        tracker.update({"loss": 0.3})
        
        avg = tracker.get_average("loss")
        assert avg == 0.4
    
    def test_get_recent_average(self):
        """Test recent average computation."""
        tracker = MetricsTracker()
        tracker._window_size = 3
        
        for i in range(5):
            tracker.update({"loss": 0.1 * (i + 1)})
        
        recent_avg = tracker.get_recent_average("loss")
        # Recent 3 values: 0.3, 0.4, 0.5 -> avg = 0.4
        assert recent_avg == 0.4
    
    def test_get_last(self):
        """Test getting last value."""
        tracker = MetricsTracker()
        
        tracker.update({"loss": 0.5})
        tracker.update({"loss": 0.3})
        
        last = tracker.get_last("loss")
        assert last == 0.3
    
    def test_reset(self):
        """Test resetting metrics."""
        tracker = MetricsTracker()
        
        tracker.update({"loss": 0.5})
        tracker.reset()
        
        assert len(tracker.metrics) == 0


class TestComputePerplexity:
    """Test perplexity computation."""
    
    def test_perplexity(self):
        """Test perplexity from loss."""
        loss = 2.0
        ppl = compute_perplexity(loss)
        
        assert abs(ppl - 7.389) < 0.01  # e^2 ≈ 7.389
    
    def test_zero_loss(self):
        """Test perplexity with zero loss."""
        ppl = compute_perplexity(0.0)
        assert ppl == 1.0
    
    def test_negative_loss(self):
        """Test perplexity with negative loss (should not happen in practice)."""
        ppl = compute_perplexity(-1.0)
        assert ppl < 1.0


class TestComputeAccuracy:
    """Test accuracy computation."""
    
    def test_accuracy(self):
        """Test basic accuracy."""
        predictions = torch.tensor([1, 2, 3, 4, 5])
        targets = torch.tensor([1, 2, 3, 4, 5])
        
        acc = compute_accuracy(predictions, targets)
        assert acc == 1.0
    
    def test_partial_accuracy(self):
        """Test partial accuracy."""
        predictions = torch.tensor([1, 2, 3, 4, 5])
        targets = torch.tensor([1, 2, 6, 4, 5])
        
        acc = compute_accuracy(predictions, targets)
        assert acc == 0.8  # 4/5 correct
    
    def test_accuracy_with_ignore(self):
        """Test accuracy with ignore index."""
        predictions = torch.tensor([1, 2, 3, 4, 5])
        targets = torch.tensor([1, -100, 3, -100, 5])
        
        acc = compute_accuracy(predictions, targets, ignore_index=-100)
        assert acc == 1.0  # Only 3 valid, all correct


class TestThroughputTracker:
    """Test ThroughputTracker class."""
    
    def test_creation(self):
        """Test tracker creation."""
        tracker = ThroughputTracker()
        assert tracker is not None
    
    def test_step_throughput(self):
        """Test step throughput calculation."""
        tracker = ThroughputTracker()
        
        tracker.start_step()
        # Simulate some work
        import time
        time.sleep(0.01)
        metrics = tracker.end_step(batch_size=32, seq_length=512)
        
        assert "samples_per_sec" in metrics
        assert "step_time_ms" in metrics
        assert "tokens_per_sec" in metrics
    
    def test_average_throughput(self):
        """Test average throughput calculation."""
        tracker = ThroughputTracker()
        
        for _ in range(5):
            tracker.start_step()
            import time
            time.sleep(0.001)
            tracker.end_step(batch_size=8)
        
        avg = tracker.get_average_throughput()
        assert "avg_samples_per_sec" in avg


class TestMemoryTracker:
    """Test MemoryTracker class."""
    
    def test_creation(self):
        """Test tracker creation."""
        tracker = MemoryTracker()
        assert tracker is not None
    
    def test_get_current_usage(self):
        """Test getting current memory usage."""
        tracker = MemoryTracker()
        usage = tracker.get_current_usage()
        
        # Should have allocated key
        assert "allocated" in usage
    
    def test_stats(self):
        """Test getting formatted stats."""
        tracker = MemoryTracker()
        stats = tracker.get_stats()
        
        assert "current_allocated_gb" in stats
        assert "peak_allocated_gb" in stats
