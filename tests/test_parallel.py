"""
Tests for parallel training components.
"""

import pytest
import torch
import torch.nn as nn

from noctua.parallel import DataParallelTrainer, PipelineParallel
from noctua.core.config import ParallelConfig, ParallelStrategy


class SimpleModel(nn.Module):
    """Simple test model."""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return {"logits": x, "loss": nn.functional.cross_entropy(x, torch.randint(0, 5, (x.shape[0],)))}


class TestDataParallel:
    """Test DataParallel training."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ParallelConfig(
            strategy=ParallelStrategy.DATA_PARALLEL,
            world_size=1,
            rank=0,
            local_rank=0,
            batch_size=4,
        )
    
    @pytest.fixture
    def model(self):
        """Create test model."""
        return SimpleModel()
    
    def test_model_creation(self, model):
        """Test model can be created."""
        assert model is not None
        assert isinstance(model, nn.Module)
    
    def test_forward_pass(self, model):
        """Test forward pass."""
        x = torch.randn(2, 10)
        output = model(x)
        
        assert "logits" in output
        assert output["logits"].shape == (2, 5)


class TestPipelineParallel:
    """Test PipelineParallel training."""
    
    @pytest.fixture
    def model(self):
        """Create test model."""
        return SimpleModel(input_size=10, hidden_size=20, output_size=5)
    
    def test_pipeline_creation(self, model):
        """Test pipeline can be created."""
        pp = PipelineParallel(
            model=model,
            num_stages=2,
            num_microbatches=4,
        )
        
        assert pp.num_stages == 2
        assert pp.num_microbatches == 4
    
    def test_stage_distribution(self, model):
        """Test layer distribution across stages."""
        pp = PipelineParallel(
            model=model,
            num_stages=2,
        )
        pp.setup()
        
        # Check stages are created
        assert len(pp.stages) == 2
        
        # Check stage info
        assert len(pp.stage_info) == 2
        for info in pp.stage_info:
            assert info.num_layers > 0
