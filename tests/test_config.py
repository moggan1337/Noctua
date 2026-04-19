"""
Tests for Noctua configuration system.
"""

import pytest
import tempfile
from pathlib import Path

from noctua import NoctuaConfig
from noctua.core.config import (
    ModelConfig,
    DataConfig,
    TrainingConfig,
    OptimizerConfig,
    ParallelConfig,
    PrecisionType,
    ParallelStrategy,
    BackendType,
)


class TestNoctuaConfig:
    """Test NoctuaConfig class."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = NoctuaConfig()
        
        assert config.model.model_name == "gpt2"
        assert config.data.batch_size == 8
        assert config.training.max_steps == 100000
        assert config.parallel.world_size == 1
    
    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "model": {"model_name": "gpt2-medium"},
            "data": {"batch_size": 16},
            "training": {"max_steps": 5000},
            "parallel": {"world_size": 4},
        }
        
        config = NoctuaConfig.from_dict(config_dict)
        
        assert config.model.model_name == "gpt2-medium"
        assert config.data.batch_size == 16
        assert config.training.max_steps == 5000
        assert config.parallel.world_size == 4
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = NoctuaConfig()
        
        # Valid configuration
        config.parallel.world_size = 8
        assert config.parallel.world_size == 8
        
        # Invalid world_size should raise
        with pytest.raises(ValueError):
            config.parallel.world_size = 0
            NoctuaConfig._validate(config)
    
    def test_save_and_load_yaml(self):
        """Test saving and loading YAML configuration."""
        config = NoctuaConfig()
        config.model.model_name = "gpt2"
        config.training.max_steps = 1000
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            config.save(path)
            
            # Load it back
            loaded = NoctuaConfig.from_yaml(path)
            
            assert loaded.model.model_name == "gpt2"
            assert loaded.training.max_steps == 1000
    
    def test_save_and_load_json(self):
        """Test saving and loading JSON configuration."""
        config = NoctuaConfig()
        config.model.model_name = "gpt2-large"
        config.parallel.world_size = 8
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            config.save(path, format="json")
            
            # Load it back
            loaded = NoctuaConfig.from_json(path)
            
            assert loaded.model.model_name == "gpt2-large"
            assert loaded.parallel.world_size == 8
    
    def test_config_copy(self):
        """Test configuration copying."""
        config = NoctuaConfig()
        config.model.model_name = "gpt2"
        config.data.batch_size = 32
        
        copied = config.copy()
        
        assert copied.model.model_name == "gpt2"
        assert copied.data.batch_size == 32
        
        # Modify original, check copy is independent
        config.data.batch_size = 64
        assert copied.data.batch_size == 32
    
    def test_config_merge(self):
        """Test configuration merging."""
        config1 = NoctuaConfig()
        config1.model.model_name = "gpt2"
        config1.data.batch_size = 8
        
        config2 = NoctuaConfig()
        config2.model.model_name = "gpt2-medium"
        config2.parallel.world_size = 8
        
        merged = config1.merge(config2)
        
        assert merged.model.model_name == "gpt2-medium"
        assert merged.data.batch_size == 8
        assert merged.parallel.world_size == 8
    
    def test_get_effective_batch_size(self):
        """Test effective batch size calculation."""
        config = NoctuaConfig()
        config.data.batch_size = 32
        config.parallel.world_size = 8
        config.parallel.gradient_accumulation_steps = 4
        
        effective = config.get_effective_batch_size()
        
        assert effective == 32 * 8 * 4  # 1024
    
    def test_is_main_process(self):
        """Test main process detection."""
        config = NoctuaConfig()
        
        config.parallel.rank = 0
        assert config.is_main_process() is True
        
        config.parallel.rank = 1
        assert config.is_main_process() is False


class TestModelConfig:
    """Test ModelConfig class."""
    
    def test_default_config(self):
        """Test default model configuration."""
        config = ModelConfig()
        
        assert config.model_name == "gpt2"
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12


class TestParallelConfig:
    """Test ParallelConfig class."""
    
    def test_default_config(self):
        """Test default parallel configuration."""
        config = ParallelConfig()
        
        assert config.strategy == ParallelStrategy.DATA_PARALLEL
        assert config.world_size == 1
        assert config.zero_stage == 1
    
    def test_zero_stage_values(self):
        """Test ZeRO stage configuration."""
        config = ParallelConfig()
        
        config.zero_stage = 1
        assert config.zero_stage == 1
        
        config.zero_stage = 2
        assert config.zero_stage == 2
        
        config.zero_stage = 3
        assert config.zero_stage == 3


class TestOptimizerConfig:
    """Test OptimizerConfig class."""
    
    def test_default_config(self):
        """Test default optimizer configuration."""
        config = OptimizerConfig()
        
        assert config.optimizer_type == "adamw"
        assert config.learning_rate == 1e-4
        assert config.beta1 == 0.9
        assert config.beta2 == 0.999
    
    def test_to_optimizer_kwargs(self):
        """Test optimizer kwargs conversion."""
        config = OptimizerConfig()
        config.learning_rate = 0.001
        config.beta1 = 0.95
        config.beta2 = 0.999
        config.epsilon = 1e-7
        config.weight_decay = 0.01
        
        kwargs = config.to_optimizer_kwargs()
        
        assert kwargs["lr"] == 0.001
        assert kwargs["betas"] == (0.95, 0.999)
        assert kwargs["eps"] == 1e-7
        assert kwargs["weight_decay"] == 0.01


class TestDataConfig:
    """Test DataConfig class."""
    
    def test_default_config(self):
        """Test default data configuration."""
        config = DataConfig()
        
        assert config.batch_size == 8
        assert config.max_seq_length == 512
        assert config.num_workers == 4
    
    def test_batch_size_per_device(self):
        """Test per-device batch size calculation."""
        config = DataConfig()
        config.batch_size = 32
        
        per_device = config.get_batch_size_per_device(world_size=8)
        
        assert per_device == 4  # 32 / 8
