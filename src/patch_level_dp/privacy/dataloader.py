# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from functools import partial
from typing import Any, List, Optional, Sequence, Tuple, Type, Union

import torch
from opacus.utils.uniform_sampler import (
    DistributedUniformWithReplacementSampler,
    UniformWithReplacementSampler,
)
from torch.utils.data import BatchSampler, DataLoader, Dataset, IterableDataset, Sampler
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataloader import _collate_fn_t
import random


logger = logging.getLogger(__name__)


class UniformWithoutReplacementSampler(Sampler):
    """Samples a fixed-size batch without replacement within each batch,
    but with replacement across batches.
    """

    def __init__(self, *, num_samples: int, batch_size: int, sample_rate: float, generator=None, steps=None):
        """
        Args:
            num_samples: Total number of available samples in the dataset.
            batch_size: Number of samples in each batch.
            sample_rate: Probability of selecting a sample in each batch.
            generator: Optional PyTorch Generator for randomness.
            steps: Number of batches per epoch (defaults to 1/sample_rate).
        """
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.generator = generator

        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be a positive integer, got {self.num_samples}")

        if self.batch_size > self.num_samples:
            raise ValueError(f"batch_size ({self.batch_size}) cannot be larger than num_samples ({self.num_samples})")

        self.steps = steps if steps is not None else int(1 / self.sample_rate)

    def __len__(self):
        return self.steps

    def __iter__(self):
        for _ in range(self.steps):
            batch_indices = random.sample(range(self.num_samples), self.batch_size)
            yield batch_indices


def collate(
    batch: List[torch.Tensor],
    *,
    collate_fn: Optional[_collate_fn_t],
    sample_empty_shapes: Sequence[Tuple],
    dtypes: Sequence[Union[torch.dtype, Type]],
):
    """Wraps `collate_fn` to handle empty batches.

    Default `collate_fn` implementations typically can't handle batches of length zero.
    Since this is a possible case for poisson sampling, we need to wrap the collate
    method, producing tensors with the correct shape and size (albeit the batch
    dimension being zero-size).

    Args:
        batch: List of tensors to be passed to collate_fn implementation
        collate_fn: Collate method to be wrapped
        sample_empty_shapes: Sample tensors with the expected shape
        dtypes: Expected dtypes

    Returns:
        Batch tensor(s)
    """
    if len(batch) > 0:
        return collate_fn(batch)
    else:
        return [
            torch.zeros(shape, dtype=_convert_dtype_to_torch_dtype(dtype))
            for shape, dtype in zip(sample_empty_shapes, dtypes)
        ]


def _convert_dtype_to_torch_dtype(dtype):
    """Convert various dtype representations to PyTorch dtype.
    
    Args:
        dtype: A dtype which could be torch.dtype, numpy.dtype, or a Python type
        
    Returns:
        torch.dtype: The corresponding PyTorch dtype
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    
    if hasattr(dtype, "kind"):
        import numpy as np
        if dtype == np.float32:
            return torch.float32
        elif dtype == np.float64:
            return torch.float64
        elif dtype == np.float16:
            return torch.float16
        elif dtype == np.int64:
            return torch.int64
        elif dtype == np.int32:
            return torch.int32
        elif dtype == np.int16:
            return torch.int16
        elif dtype == np.int8:
            return torch.int8
        elif dtype == np.uint8:
            return torch.uint8
        elif dtype == np.bool_:
            return torch.bool
        return torch.float32
    
    if dtype == int:
        return torch.int64
    elif dtype == float:
        return torch.float32
    elif dtype == bool:
        return torch.bool
    
    return torch.float32


def wrap_collate_with_empty(
    *,
    collate_fn: Optional[_collate_fn_t],
    sample_empty_shapes: Sequence[Tuple],
    dtypes: Sequence[Union[torch.dtype, Type]],
):
    """Wraps given collate function to handle empty batches.

    Args:
        collate_fn: collate function to wrap
        sample_empty_shapes: expected shape for a batch of size 0. Input is a sequence -
            one for each tensor in the dataset

    Returns:
        New collate function, which is equivalent to input ``collate_fn`` for non-empty
        batches and outputs empty tensors with shapes from ``sample_empty_shapes`` if
        the input batch is of size 0
    """
    return partial(
        collate,
        collate_fn=collate_fn,
        sample_empty_shapes=sample_empty_shapes,
        dtypes=dtypes,
    )


def shape_safe(x: Any) -> Tuple:
    """Exception-safe getter for ``shape`` attribute.

    Args:
        x: any object

    Returns:
        ``x.shape`` if attribute exists, empty tuple otherwise
    """
    return getattr(x, "shape", ())


def dtype_safe(x: Any) -> Union[torch.dtype, Type]:
    """Exception-safe getter for ``dtype`` attribute.

    Args:
        x: any object

    Returns:
        ``x.dtype`` if attribute exists, type of x otherwise
    """
    return getattr(x, "dtype", type(x))


class DPDataLoader(DataLoader):
    """DataLoader subclass that always does Poisson sampling and supports empty batches.

    Typically instantiated via ``DPDataLoader.from_data_loader()`` method based
    on another DataLoader. DPDataLoader would preserve the behaviour of the original
    data loader, except for the sampling mechanism and empty batch handling.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        sample_rate: float,
        collate_fn: Optional[_collate_fn_t] = None,
        drop_last: bool = False,
        generator=None,
        distributed: bool = False,
        replacement: bool = False,
        batch_size: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            dataset: See :class:`torch.utils.data.DataLoader`
            sample_rate: probability with which each element of the dataset is included
                in the next batch.
            collate_fn: See :class:`torch.utils.data.DataLoader`
            drop_last: See :class:`torch.utils.data.DataLoader`
            generator: Random number generator used to sample elements
            distributed: set ``True`` if you'll be using DPDataLoader in a DDP environment
            replacement: set ``False`` if you want to use without replacement sampling
            batch_size: needed for without replacement sampling
        """
        self.sample_rate = sample_rate
        self.distributed = distributed

        if replacement:
            if distributed:
                batch_sampler = DistributedUniformWithReplacementSampler(
                    total_size=len(dataset),  # type: ignore[assignment, arg-type]
                    sample_rate=sample_rate,
                    generator=generator,
                )
            else:
                batch_sampler = UniformWithReplacementSampler(
                    num_samples=len(dataset),  # type: ignore[assignment, arg-type]
                    sample_rate=sample_rate,
                    generator=generator,
                )
        else:
            if distributed:
                raise NotImplementedError('Currently only works for not distributed')
            else:
                batch_sampler = UniformWithoutReplacementSampler(
                    num_samples=len(dataset),  # type: ignore[assignment, arg-type]
                    batch_size=batch_size,
                    sample_rate=sample_rate,
                    generator=generator
                )

        sample_empty_shapes = [(0, *shape_safe(x)) for x in dataset[0]]
        dtypes = [dtype_safe(x) for x in dataset[0]]
        if collate_fn is None:
            collate_fn = default_collate

        if drop_last:
            logger.warning(
                "Ignoring drop_last as it is not compatible with DPDataLoader."
            )

        super().__init__(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=wrap_collate_with_empty(
                collate_fn=collate_fn,
                sample_empty_shapes=sample_empty_shapes,
                dtypes=dtypes,
            ),
            generator=generator,
            **kwargs,
        )

    @classmethod
    def from_data_loader(
        cls, data_loader: DataLoader, *, batch_size=None, distributed: bool = False, replacement: bool = False, generator=None
    ):
        """Creates new ``DPDataLoader`` based on passed ``data_loader`` argument.

        Args:
            data_loader: Any DataLoader instance. Must not be over an ``IterableDataset``
            distributed: set ``True`` if you'll be using DPDataLoader in a DDP environment
            generator: Random number generator used to sample elements. Defaults to
                generator from the original data loader.

        Returns:
            New DPDataLoader instance, with all attributes and parameters inherited
            from the original data loader, except for sampling mechanism.

        Examples:
            >>> x, y = torch.randn(64, 5), torch.randint(0, 2, (64,))
            >>> dataset = TensorDataset(x,y)
            >>> data_loader = DataLoader(dataset, batch_size=4)
            >>> dp_data_loader = DPDataLoader.from_data_loader(data_loader)
        """
        if isinstance(data_loader.dataset, IterableDataset):
            raise ValueError("Uniform sampling is not supported for IterableDataset")

        return cls(
            dataset=data_loader.dataset,
            sample_rate=1 / len(data_loader),
            num_workers=data_loader.num_workers,
            collate_fn=data_loader.collate_fn,
            pin_memory=data_loader.pin_memory,
            drop_last=data_loader.drop_last,
            timeout=data_loader.timeout,
            worker_init_fn=data_loader.worker_init_fn,
            multiprocessing_context=data_loader.multiprocessing_context,
            generator=generator if generator else data_loader.generator,
            prefetch_factor=data_loader.prefetch_factor,
            persistent_workers=data_loader.persistent_workers,
            distributed=distributed,
            replacement=replacement,
            batch_size=batch_size
        )


def _is_supported_batch_sampler(sampler: Sampler):
    return (
        isinstance(sampler, BatchSampler)
        or isinstance(sampler, UniformWithReplacementSampler)
        or isinstance(sampler, DistributedUniformWithReplacementSampler)
        or isinstance(sampler, UniformWithoutReplacementSampler)
    )


def switch_generator(*, data_loader: DataLoader, generator):
    """Creates new instance of a ``DataLoader``, with the exact same behaviour of the
    provided data loader, except for the source of randomness.

    Typically used to enhance a user-provided data loader object with cryptographically
    secure random number generator.

    Args:
        data_loader: Any ``DataLoader`` object
        generator:  Random number generator object

    Returns:
        New ``DataLoader`` object with the exact same behaviour as the input data loader,
        except for the source of randomness.
    """
    batch_sampler = data_loader.batch_sampler

    if batch_sampler is None or not _is_supported_batch_sampler(batch_sampler):
        raise ValueError(
            "Non-batch processing is not supported: Opacus always assumes one of the input dimensions to be batch dimension."
        )

    if isinstance(batch_sampler, BatchSampler):
        if not hasattr(batch_sampler.sampler, "generator"):
            raise ValueError(
                "Target sampler doesn't have generator attribute: nothing to switch"
            )

        batch_sampler.sampler.generator = generator
    else:
        batch_sampler.generator = generator

    return DataLoader(
        dataset=data_loader.dataset,
        batch_sampler=batch_sampler,
        num_workers=data_loader.num_workers,
        collate_fn=data_loader.collate_fn,
        pin_memory=data_loader.pin_memory,
        drop_last=data_loader.drop_last,
        timeout=data_loader.timeout,
        worker_init_fn=data_loader.worker_init_fn,
        multiprocessing_context=data_loader.multiprocessing_context,
        generator=generator,
        prefetch_factor=data_loader.prefetch_factor,
        persistent_workers=data_loader.persistent_workers,
    )