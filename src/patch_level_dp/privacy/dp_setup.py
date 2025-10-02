"""Opacus integration utilities for DP model training."""

import math

import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from opacus import GradSampleModule
from opacus.optimizers import DPOptimizer
from opacus.utils.batch_memory_manager import BatchMemoryManager, BatchSplittingSampler

from .dataloader import UniformWithoutReplacementSampler


class ExtendedBatchSplittingSampler(BatchSplittingSampler):
    """Extended BatchSplittingSampler that properly handles UniformWithoutReplacementSampler."""
    
    def __len__(self):
        if isinstance(self.sampler, UniformWithoutReplacementSampler):
            expected_batch_size = self.sampler.batch_size
            return math.ceil(len(self.sampler) * (expected_batch_size / self.max_batch_size))
        return super().__len__()


class ExtendedBatchMemoryManager(BatchMemoryManager):
    """Extended BatchMemoryManager that properly handles UniformWithoutReplacementSampler."""
    
    def __enter__(self):
        """Create a custom wrapped data loader using our extended sampler."""
        return self._wrap_data_loader(
            data_loader=self.data_loader,
            max_batch_size=self.max_physical_batch_size,
            optimizer=self.optimizer
        )
    
    def _wrap_data_loader(self, *, data_loader, max_batch_size, optimizer):
        """Custom version of wrap_data_loader that uses ExtendedBatchSplittingSampler."""
        return DataLoader(
            dataset=data_loader.dataset,
            batch_sampler=ExtendedBatchSplittingSampler(
                sampler=data_loader.batch_sampler,
                max_batch_size=max_batch_size,
                optimizer=optimizer,
            ),
            num_workers=data_loader.num_workers,
            collate_fn=data_loader.collate_fn,
            pin_memory=data_loader.pin_memory,
            timeout=data_loader.timeout,
            worker_init_fn=data_loader.worker_init_fn,
        )


def setup_dp_model(model: nn.Module) -> nn.Module:
    """Apply GradSampleModule wrapping for DP-SGD.
    
    Args:
        model: PyTorch model to wrap
        
    Returns:
        Wrapped model with gradient sampling capabilities
    """
    return GradSampleModule(model)


def setup_dp_optimizer(
    optimizer: Optimizer,
    noise_multiplier: float,
    max_grad_norm: float,
    expected_batch_size: int
) -> DPOptimizer:
    """Create a DP optimizer from a regular optimizer.
    
    Args:
        optimizer: Regular PyTorch optimizer
        noise_multiplier: Standard deviation for DP noise 
        max_grad_norm: Maximum gradient norm for clipping
        expected_batch_size: Expected batch size for normalization
        
    Returns:
        DPOptimizer instance
    """
    return DPOptimizer(
        optimizer,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        expected_batch_size=expected_batch_size,
    )


def setup_batch_memory_manager(
    data_loader: DataLoader,
    max_physical_batch_size: int,
    optimizer: DPOptimizer
) -> ExtendedBatchMemoryManager:
    """Create a batch memory manager for efficient memory usage.
    
    Args:
        data_loader: DataLoader to manage
        max_physical_batch_size: Maximum physical batch size
        optimizer: DP optimizer to use
        
    Returns:
        ExtendedBatchMemoryManager instance
    """
    return ExtendedBatchMemoryManager(
        data_loader=data_loader,
        max_physical_batch_size=max_physical_batch_size,
        optimizer=optimizer
    )