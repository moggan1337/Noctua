"""
Noctua Configuration System

Provides comprehensive configuration management for distributed LLM training,
supporting YAML/JSON config files and programmatic configuration.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class PrecisionType(Enum):
    """Mixed precision training types."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class ParallelStrategy(Enum):
    """Distributed parallelization strategies."""
    DATA_PARALLEL = "dp"
    PIPELINE_PARALLEL = "pp"
    TENSOR_PARALLEL = "tp"
    ZERO1 = "zero1"
    ZERO2 = "zero2"
    ZERO3 = "zero3"
    HYBRID = "hybrid"


class BackendType(Enum):
    """Distributed backend types."""
    NCCL = "nccl"
    GLOO = "gloo"
    MPI = "mpi"


@dataclass
class ParallelConfig:
    """Configuration for parallel distributed training."""
    
    strategy: ParallelStrategy = ParallelStrategy.DATA_PARALLEL
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: BackendType = BackendType.NCCL
    
    # Pipeline Parallelism settings
    pipeline_parallel_size: int = 1
    num_pipeline_stages: int = 1
    pipeline_chunk_size: int = 1
    
    # Tensor Parallelism settings
    tensor_parallel_size: int = 1
    
    # ZeRO settings
    zero_stage: int = 1
    zero_cpu_offload: bool = False
    zero_initialize_on_model: bool = True
    
    # Gradient settings
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float = 1.0
    
    # Communication
    broadcast_buffers: bool = True
    find_unused_parameters: bool = False
    

@dataclass
class ModelConfig:
    """Configuration for the LLM model."""
    
    model_name: str = "gpt2"
    model_type: str = "causal-lm"
    
    # Architecture overrides
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    vocab_size: int = 50257
    max_position_embeddings: int = 1024
    
    # Quantization (for loading)
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    # Flash Attention
    use_flash_attention: bool = True
    attention_dropout: float = 0.0
    
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> ModelConfig:
        """Create config from pretrained model name."""
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(model_name)
        
        return cls(
            model_name=model_name,
            hidden_size=hf_config.hidden_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            intermediate_size=getattr(hf_config, "intermediate_size", 4 * hf_config.hidden_size),
            vocab_size=hf_config.vocab_size,
            max_position_embeddings=getattr(hf_config, "max_position_embeddings", 1024),
            **kwargs
        )


@dataclass
class OptimizerConfig:
    """Configuration for the optimizer."""
    
    optimizer_type: str = "adamw"
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.01
    
    # Learning rate scheduling
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100
    warmup_ratio: float = 0.0
    min_lr: float = 0.0
    
    # Advanced optimizer settings
    max_grad_norm: float = 1.0
    loss_scale: float = 1.0
    initial_loss_scale: float = 2**16
    loss_scale_factor: float = 2.0
    loss_scale_window: int = 1000
    
    # Distributed optimizer (for ZeRO)
    oslo_config: Optional[Dict[str, Any]] = None
    
    def to_optimizer_kwargs(self) -> Dict[str, Any]:
        """Convert to optimizer keyword arguments."""
        return {
            "lr": self.learning_rate,
            "betas": (self.beta1, self.beta2),
            "eps": self.epsilon,
            "weight_decay": self.weight_decay,
        }


@dataclass
class DataConfig:
    """Configuration for data loading."""
    
    dataset_path: str = ""
    dataset_name: Optional[str] = None
    dataset_split: str = "train"
    
    # Data processing
    max_seq_length: int = 512
    batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # Caching
    cache_dir: Optional[str] = None
    overwrite_cache: bool = False
    
    # Filtering
    filter_short_sequences: bool = True
    min_seq_length: int = 16
    
    # Augmentation / Processing
    preprocessing_num_workers: int = 8
    multiline_training: bool = False
    
    # Packing (for efficient sequence packing)
    use_packing: bool = False
    packing_ratio: float = 1.0
    
    def get_batch_size_per_device(self, world_size: int) -> int:
        """Calculate per-device batch size."""
        return max(1, self.batch_size // world_size)


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""
    
    output_dir: str = "./output"
    run_name: str = "noctua-run"
    
    # Training duration
    max_steps: int = 100000
    max_epochs: Optional[int] = None
    eval_steps: int = 1000
    save_steps: int = 5000
    logging_steps: int = 100
    warmup_steps: int = 100
    
    # Checkpointing
    save_total_limit: int = 3
    save_strategy: str = "steps"
    load_checkpoint: Optional[str] = None
    
    # Logging
    logging_dir: Optional[str] = None
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    log_gradients: bool = False
    log_weights: bool = False
    
    # Evaluation
    do_eval: bool = True
    eval_first: bool = False
    
    # Optimization
    precision: PrecisionType = PrecisionType.FP16
    seed: int = 42
    deterministic: bool = False
    
    # Training behavior
    dataloader_pin_memory: bool = True
    resume_from_checkpoint: bool = False
    gradient_checkpointing: bool = False
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.0
    
    # Profiling
    profile: bool = False
    profile_steps: List[int] = field(default_factory=lambda: [1, 5])
    
    # Misc
    dry_run: bool = False
    fast_dev_run: bool = False


@dataclass 
class NoctuaConfig:
    """
    Main configuration class for Noctua distributed training.
    
    Aggregates all configuration components and provides methods
    for loading/saving configurations from files.
    
    Example:
        >>> config = NoctuaConfig.from_yaml("config.yaml")
        >>> config = NoctuaConfig.from_dict({
        ...     "parallel": {"strategy": "zero2", "world_size": 8},
        ...     "training": {"max_steps": 10000, "batch_size": 32}
        ... })
        >>> config.save("my_config.yaml")
    """
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    
    # System settings
    use_cuda: bool = True
    use_flash_attention: bool = True
    custom_kernel_backends: List[str] = field(default_factory=list)
    
    # Environment
    master_addr: str = "localhost"
    master_port: int = 29500
    init_method: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate configuration values."""
        if self.parallel.world_size < 1:
            raise ValueError("world_size must be >= 1")
        
        if self.data.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        
        if self.training.max_steps < 1 and self.training.max_epochs is None:
            raise ValueError("Must specify either max_steps or max_epochs")
        
        # Validate parallel strategy compatibility
        if self.parallel.strategy == ParallelStrategy.PIPELINE_PARALLEL:
            if self.parallel.pipeline_parallel_size > self.parallel.world_size:
                raise ValueError("pipeline_parallel_size cannot exceed world_size")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> NoctuaConfig:
        """Create config from dictionary."""
        config = cls()
        
        for key in ["model", "data", "training", "optimizer", "parallel"]:
            if key in config_dict:
                config_dict_key = config_dict[key]
                dataclass_type = {
                    "model": ModelConfig,
                    "data": DataConfig,
                    "training": TrainingConfig,
                    "optimizer": OptimizerConfig,
                    "parallel": ParallelConfig,
                }[key]
                if isinstance(config_dict_key, dict):
                    setattr(config, key, dataclass_type(**config_dict_key))
                else:
                    setattr(config, key, config_dict_key)
        
        # Handle top-level settings
        for key in ["use_cuda", "use_flash_attention", "master_addr", "master_port"]:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        return config
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> NoctuaConfig:
        """Load configuration from YAML file."""
        import yaml
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> NoctuaConfig:
        """Load configuration from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r") as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_env(cls) -> NoctuaConfig:
        """Create config from environment variables."""
        config = cls()
        
        # Parallel settings
        if "WORLD_SIZE" in os.environ:
            config.parallel.world_size = int(os.environ["WORLD_SIZE"])
        if "RANK" in os.environ:
            config.parallel.rank = int(os.environ["RANK"])
        if "LOCAL_RANK" in os.environ:
            config.parallel.local_rank = int(os.environ["LOCAL_RANK"])
        if "MASTER_ADDR" in os.environ:
            config.master_addr = os.environ["MASTER_ADDR"]
        if "MASTER_PORT" in os.environ:
            config.master_port = int(os.environ["MASTER_PORT"])
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "optimizer": self.optimizer.__dict__,
            "parallel": self.parallel.__dict__,
            "use_cuda": self.use_cuda,
            "use_flash_attention": self.use_flash_attention,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
        }
    
    def save(self, path: Union[str, Path], format: str = "yaml") -> None:
        """Save configuration to file."""
        import yaml
        import json
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        if format == "yaml":
            with open(path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        elif format == "json":
            with open(path, "w") as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def copy(self) -> NoctuaConfig:
        """Create a deep copy of the config."""
        return copy.deepcopy(self)
    
    def merge(self, other: NoctuaConfig) -> NoctuaConfig:
        """Merge another config into this one (other takes precedence)."""
        result = self.copy()
        
        for key in ["model", "data", "training", "optimizer", "parallel"]:
            other_section = getattr(other, key)
            current_section = getattr(result, key)
            
            if isinstance(other_section, dict):
                other_section = type(current_section)(**other_section)
            
            for field_name in other_section.__dataclass_fields__.keys():
                other_value = getattr(other_section, field_name)
                if other_value is not None and other_value != getattr(type(other_section), field_name, None).default:
                    setattr(current_section, field_name, other_value)
        
        return result
    
    def get_device(self) -> str:
        """Get the primary device for this config."""
        if self.use_cuda:
            return f"cuda:{self.parallel.local_rank}" if self.parallel.local_rank > 0 else "cuda"
        return "cpu"
    
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.parallel.rank == 0
    
    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size with gradient accumulation."""
        base_batch = self.data.batch_size * self.parallel.world_size
        return base_batch * self.parallel.gradient_accumulation_steps
    
    def __repr__(self) -> str:
        return (
            f"NoctuaConfig(\n"
            f"  model={self.model.model_name},\n"
            f"  parallel_strategy={self.parallel.strategy.value},\n"
            f"  world_size={self.parallel.world_size},\n"
            f"  precision={self.training.precision.value},\n"
            f"  batch_size={self.data.batch_size}\n"
            f")"
        )


# Preset configurations for common setups
class PresetConfigs:
    """Pre-configured settings for common training scenarios."""
    
    @staticmethod
    def llama_7b_single_node() -> NoctuaConfig:
        """Config for LLaMA 7B on a single 8-GPU node with ZeRO-2."""
        config = NoctuaConfig()
        config.model = ModelConfig(
            model_name="meta-llama/Llama-2-7b-hf",
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            intermediate_size=11008,
            vocab_size=32000,
        )
        config.parallel = ParallelConfig(
            strategy=ParallelStrategy.ZERO2,
            world_size=8,
            backend=BackendType.NCCL,
            zero_stage=2,
            gradient_accumulation_steps=8,
        )
        config.data.batch_size = 4
        config.training.max_steps = 100000
        return config
    
    @staticmethod
    def gpt3_175b_multi_node() -> NoctuaConfig:
        """Config for GPT-3 175B across multiple nodes with pipeline parallelism."""
        config = NoctuaConfig()
        config.model = ModelConfig(
            model_name="gpt3-175b",
            hidden_size=12288,
            num_hidden_layers=96,
            num_attention_heads=96,
            intermediate_size=49152,
            vocab_size=50257,
        )
        config.parallel = ParallelConfig(
            strategy=ParallelStrategy.PIPELINE_PARALLEL,
            world_size=64,
            pipeline_parallel_size=8,
            num_pipeline_stages=8,
            gradient_accumulation_steps=16,
        )
        config.data.batch_size = 16
        config.training.max_steps = 50000
        return config
    
    @staticmethod
    def small_research_run() -> NoctuaConfig:
        """Minimal config for research experiments on single GPU."""
        config = NoctuaConfig()
        config.parallel = ParallelConfig(
            strategy=ParallelStrategy.DATA_PARALLEL,
            world_size=1,
        )
        config.training.max_steps = 1000
        config.data.batch_size = 8
        return config
