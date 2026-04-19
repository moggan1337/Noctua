"""
Data Utilities

Tools for dataset creation, tokenization, and efficient dataloading
with support for packed sequences and distributed training.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset, Dataset as HFDataset

from noctua.core.config import DataConfig


class TokenizedDataset(Dataset):
    """
    Tokenized dataset for language model training.
    
    Handles tokenization, padding, and sequence packing.
    
    Example:
        >>> dataset = TokenizedDataset(
        ...     text_data=["Hello world", "Another text"],
        ...     tokenizer=tokenizer,
        ...     max_length=512,
        ... )
        >>> dataloader = DataLoader(dataset, batch_size=8)
    """
    
    def __init__(
        self,
        texts: Optional[List[str]] = None,
        dataset: Optional[HFDataset] = None,
        text_column: str = "text",
        tokenizer: Any = None,
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt",
        add_special_tokens: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
        self.add_special_tokens = add_special_tokens
        
        # Load data
        if texts is not None:
            self.texts = texts
        elif dataset is not None:
            self.texts = dataset[text_column] if text_column in dataset.column_names else dataset["text"]
        else:
            self.texts = []
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single tokenized example."""
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=self.return_tensors,
            add_special_tokens=self.add_special_tokens,
        )
        
        # Extract tensors
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        
        # Add labels (same as input_ids for causal LM)
        item["labels"] = item["input_ids"].clone()
        
        return item


class PackedDataset(IterableDataset):
    """
    Dataset with packed sequences for efficient training.
    
    Packs multiple short sequences into longer sequences
    to maximize GPU utilization.
    
    Example:
        >>> dataset = PackedDataset(
        ...     texts=texts,
        ...     tokenizer=tokenizer,
        ...     max_length=4096,
        ...     packing_ratio=1.5,
        ... )
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: Any,
        max_length: int = 4096,
        packing_ratio: float = 1.0,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.packing_ratio = packing_ratio
        self.shuffle = shuffle
        self.seed = seed
        
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over packed sequences."""
        if self.shuffle:
            indices = torch.randperm(len(self.texts), generator=self.rng)
        else:
            indices = torch.arange(len(self.texts))
        
        current_tokens = []
        current_masks = []
        
        for idx in indices:
            text = self.texts[idx]
            
            # Tokenize
            encoding = self.tokenizer(
                text,
                max_length=self.max_length // 4,  # Target shorter sequences
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            
            current_tokens.append(input_ids)
            current_masks.append(attention_mask)
            
            # Check if pack is full
            total_length = sum(t.size(0) for t in current_tokens)
            
            if total_length >= self.max_length * self.packing_ratio:
                # Pack and yield
                packed = self._pack_sequences(current_tokens, current_masks)
                yield packed
                
                # Reset
                current_tokens = []
                current_masks = []
        
        # Yield remaining
        if current_tokens:
            packed = self._pack_sequences(current_tokens, current_masks)
            yield packed
    
    def _pack_sequences(
        self,
        tokens: List[torch.Tensor],
        masks: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Pack multiple sequences into one."""
        # Concatenate
        packed_input_ids = torch.cat(tokens)[:self.max_length]
        packed_masks = torch.cat(masks)[:self.max_length]
        
        # Create labels (shift for causal LM)
        labels = packed_input_ids.clone()
        labels[:-1] = packed_input_ids[1:]
        labels[-1] = -100  # Ignore last token
        
        return {
            "input_ids": packed_input_ids,
            "attention_mask": packed_masks,
            "labels": labels,
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    collate_fn: Optional[Callable] = None,
    sampler: Optional[DistributedSampler] = None,
) -> DataLoader:
    """
    Create a dataloader with optimized settings.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size per step
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop incomplete final batch
        prefetch_factor: Number of batches to prefetch
        persistent_workers: Keep workers alive between epochs
        collate_fn: Custom collate function
        sampler: Distributed sampler (if using distributed training)
        
    Returns:
        Configured DataLoader
    """
    # Determine number of workers
    effective_workers = min(num_workers, batch_size) if num_workers > 0 else 0
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=effective_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        prefetch_factor=prefetch_factor if effective_workers > 0 else None,
        persistent_workers=persistent_workers if effective_workers > 0 else False,
        collate_fn=collate_fn,
        sampler=sampler,
        multiprocessing_context=None,  # Use default
    )
    
    return dataloader


def load_text_dataset(
    path: Union[str, Path],
    split: str = "train",
    text_column: str = "text",
    streaming: bool = False,
) -> HFDataset:
    """
    Load text dataset from file or HuggingFace hub.
    
    Args:
        path: Path to local file or HuggingFace dataset name
        split: Dataset split to load
        text_column: Name of text column
        streaming: Whether to use streaming mode
        
    Returns:
        Loaded dataset
    """
    path = str(path)
    
    # Check if local file or HF dataset
    if Path(path).exists():
        # Local file
        from datasets import load_dataset
        
        ext = Path(path).suffix.lower()
        if ext in [".txt", ".text"]:
            dataset = load_dataset("text", data_files=path, split=split)
        elif ext in [".json", ".jsonl"]:
            dataset = load_dataset("json", data_files=path, split=split)
        elif ext in [".csv"]:
            dataset = load_dataset("csv", data_files=path, split=split)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    else:
        # HuggingFace dataset
        dataset = load_dataset(path, split=split, streaming=streaming)
    
    return dataset


def prepare_training_batch(
    batch: Dict[str, Any],
    device: Union[str, torch.device] = "cuda",
    ignore_index: int = -100,
) -> Dict[str, torch.Tensor]:
    """
    Prepare a training batch for the model.
    
    Args:
        batch: Raw batch from dataloader
        device: Target device
        ignore_index: Index to ignore in loss computation
        
    Returns:
        Prepared batch ready for training
    """
    prepared = {}
    
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            prepared[key] = value.to(device)
        elif isinstance(value, (list, tuple)):
            prepared[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
    
    # Ensure labels exist (for causal LM)
    if "labels" not in prepared and "input_ids" in prepared:
        prepared["labels"] = prepared["input_ids"].clone()
    
    return prepared


def create_collate_fn(
    tokenizer: Any,
    max_length: int = 512,
    padding: str = "max_length",
    label_ignore_index: int = -100,
) -> Callable:
    """Create a collate function for dynamic padding."""
    
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract all input_ids and attention_masks
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]
        labels = [item.get("labels", item["input_ids"]) for item in batch]
        
        # Tokenizer pad
        padded = tokenizer.pad(
            {
                "input_ids": input_ids,
                "attention_mask": attention_masks,
            },
            padding=padding,
            max_length=max_length,
            return_tensors="pt",
        )
        
        # Process labels
        labels_padded = tokenizer.pad(
            {"input_ids": labels},
            padding=padding,
            max_length=max_length,
            return_tensors="pt",
        )["input_ids"]
        
        # Replace padding with ignore index
        labels_padded[labels_padded == tokenizer.pad_token_id] = label_ignore_index
        
        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "labels": labels_padded,
        }
    
    return collate_fn
