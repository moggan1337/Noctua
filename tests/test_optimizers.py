"""
Tests for optimizer components.
"""

import pytest
import torch
import torch.nn as nn

from noctua.optimizers import (
    ZeroDistributedOptimizer,
    MixedPrecisionAdamW,
    PartitionedOptimizer,
)


class SimpleModel(nn.Module):
    """Simple test model."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)


class TestMixedPrecisionAdamW:
    """Test MixedPrecisionAdamW optimizer."""
    
    @pytest.fixture
    def model(self):
        """Create test model."""
        return SimpleModel()
    
    def test_optimizer_creation(self, model):
        """Test optimizer can be created."""
        optimizer = MixedPrecisionAdamW(
            model.parameters(),
            lr=1e-4,
        )
        
        assert optimizer is not None
        assert optimizer.loss_scale == 2**16
    
    def test_step(self, model):
        """Test optimizer step."""
        optimizer = MixedPrecisionAdamW(
            model.parameters(),
            lr=1e-3,
        )
        
        # Forward pass
        x = torch.randn(2, 10)
        y = model(x)
        loss = y.sum()
        
        # Backward
        loss.backward()
        
        # Step
        optimizer.step()
        optimizer.zero_grad()
        
        assert True  # No exception means success
    
    def test_loss_scaling(self, model):
        """Test loss scaling functionality."""
        optimizer = MixedPrecisionAdamW(
            model.parameters(),
            lr=1e-4,
            init_scale=2**10,
        )
        
        assert optimizer.loss_scale == 2**10
    
    def test_growth_tracking(self, model):
        """Test loss scale growth tracking."""
        optimizer = MixedPrecisionAdamW(
            model.parameters(),
            lr=1e-4,
            growth_interval=10,
        )
        
        assert optimizer.growth_tracker == 0


class TestPartitionedOptimizer:
    """Test PartitionedOptimizer."""
    
    @pytest.fixture
    def model(self):
        """Create test model."""
        return SimpleModel()
    
    def test_optimizer_creation(self, model):
        """Test partitioned optimizer can be created."""
        optimizer = PartitionedOptimizer(
            model.parameters(),
            partition_size=1,
            lr=1e-4,
        )
        
        assert optimizer is not None
        assert optimizer.partition_size == 1
    
    def test_step(self, model):
        """Test optimizer step."""
        optimizer = PartitionedOptimizer(
            model.parameters(),
            partition_size=1,
            lr=1e-3,
        )
        
        # Forward pass
        x = torch.randn(2, 10)
        y = model(x)
        loss = y.sum()
        
        # Backward
        loss.backward()
        
        # Step
        optimizer.step()
        optimizer.zero_grad()
        
        assert True


class TestZeroDistributedOptimizer:
    """Test ZeroDistributedOptimizer."""
    
    @pytest.fixture
    def model(self):
        """Create test model."""
        return SimpleModel()
    
    def test_optimizer_creation(self, model):
        """Test ZeRO optimizer can be created."""
        base_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
        )
        
        optimizer = ZeroDistributedOptimizer(
            optimizer=base_optimizer,
            named_parameters=model.named_parameters(),
            level=1,
        )
        
        assert optimizer is not None
        assert optimizer.level == 1
