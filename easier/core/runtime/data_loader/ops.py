# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from types import EllipsisType
from typing import Iterator, List, Literal, Optional, Sequence, Tuple, TypeAlias, Union, cast
import h5py
import functools
import copy

import numpy as np
import sympy
import torch

from easier.core.runtime.data_loader.base import \
    DataLoaderBase, SimpleIndex, Num
from easier.core.runtime.data_loader.factories import \
    ArangeTensorLoader, FulledTensorLoader
from easier.core.runtime.data_loader.utils import \
    get_strides

from easier.core.runtime.dist_env import \
    get_default_dist_env, get_runtime_dist_env
from easier.core.runtime.utils import check_collective_equality
from easier.core.utils import EasierJitException






class StridedDataLoader(DataLoaderBase):
    def __init__(self, inner: DataLoaderBase, index: Sequence[SimpleIndex]):
        super().__init__()

        self.inner = inner

        # The index must be converted to valid and in-range values.
        self.index = index

        strided = inner.get_placeholder()[index]
        self.shape = tuple(strided.shape)
    
    def collective_init(self) -> None:
        self.coll_check_dtype_shape_devicetype()
    
    def _rev_compose_index_range(self, region: SimpleIndex) -> SimpleIndex:
        raise NotImplementedError()
    def _rev_compose_index_tensor(self, index: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    def minmax(self, region: SimpleIndex) -> Tuple[Num, Num]:
        region = self._rev_compose_index_range(region)
        return self.inner.minmax(region)
    
    def count_unique(self, region: SimpleIndex) -> int:
        region = self._rev_compose_index_range(region)
        return self.inner.count_unique(region)
    
    def partially_load_by_range(self, region: SimpleIndex) -> torch.Tensor:
        region = self._rev_compose_index_range(region)
        return self.inner.partially_load_by_range(region)
    
    def partially_load_by_index(self, index: torch.Tensor, **kwargs) -> torch.Tensor:
        index = self._rev_compose_index_tensor(index)
        return self.inner.partially_load_by_index(index, **kwargs)
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(tensor={self.tensor})'


class CartesianProductDataLoader(DataLoaderBase):
    """
    Result is 1-d.
    """
    def __init__(
        self,
        components: Sequence[DataLoaderBase],
        # TODO to support item datatype other than scalar, we may need 'form':
        # form: Literal['flatten', 'stack'] = 'flatten'
    ):
        super().__init__()

        self.components = list(components)
    
    def collective_init(self) -> None:
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(tensor={self.tensor})'


    


class ConcatDataLoader(DataLoaderBase):
    def __init__(
        self,
        components: Sequence[DataLoaderBase],
        # TODO allow to concat along dim>0
    ):
        super().__init__()

        if 1 != len(
            set((d.shape[1:], d.dtype, d.device) for d in components)
        ):
            raise ValueError(
                "Inputs to concat must have the same shape[1:]/dtype/device"
            )

        self.components = list(components)
        self._lengths = list(d.shape[0] for d in components)

        self.shape = (sum(self._lengths),) + components[0].shape[1:]
        self.dtype = components[0].dtype
        self.device = components[0].device

    def collective_init(self) -> None:
        self.coll_check_dtype_shape_devicetype()
        check_collective_equality(
            'components batch sizes', self._lengths
        )
        # TODO check the equality of the whole hierarchy?
    
    def _foreach_in_region(self, region, fn):
        # fn: (DataLoaderBase, _SimpleIndex) -> None
        # NOTE if region[0] is slice-with-negative-step, foreach in reversed.

        if not isinstance(region, tuple):
            region = (region,)

        batch_idx = region[0]
        other_idxes = region[1:]

        _components = self.components
        if isinstance(batch_idx, slice):
            if batch_idx.step < 0:
                _components = reversed(self.components)

        _offset = 0
        for comp in _components:
            comp_start = _offset
            comp_end = _offset + comp.shape[0]

            if isinstance(batch_idx, int):
                if comp_start <= batch_idx and batch_idx < comp_end:
                    comp_bs_idx = batch_idx - comp_start
                    fn(comp, (comp_bs_idx,) + other_idxes)
                    return

            if isinstance(batch_idx, slice):
                overlap = _get_overlapped_range(
                    slice(comp_start, comp_end), batch_idx
                )
                if overlap.start != overlap.stop:
                    comp_batch = slice(
                        overlap.start - comp_start,
                        overlap.stop - comp_end,
                        overlap.step
                    )
                    fn(comp, (comp_batch,) + other_idxes)

            elif batch_idx is Ellipsis:
                fn(comp, (Ellipsis,) + other_idxes)

            else:
                assert False, f'unexpected dim-0 index {batch_idx}'

            _offset = comp_end

    
    def minmax(self, region: SimpleIndex) -> Tuple[Num, Num]:
        aminmax = [math.inf, -math.inf]
        def _minmax(comp: DataLoaderBase, comp_region: SimpleIndex):
            comp_minmax = comp.minmax(comp_region)
            aminmax[0] = min(comp_minmax[0], aminmax[0])
            aminmax[1] = max(comp_minmax[1], aminmax[1])
        self._foreach_in_region(region, _minmax)
        return tuple(aminmax)  # type: ignore

    
    def count_unique(self, region: SimpleIndex) -> int:
        _, c = self.partially_load_by_range(region).unique(return_counts=True)
        return c
    
    def partially_load_by_range(self, region: SimpleIndex) -> torch.Tensor:
        parts = []
        def _load_comp(comp: DataLoaderBase, comp_region: SimpleIndex):
            parts.append(comp.partially_load_by_range(comp_region))
        self._foreach_in_region(region, _load_comp)
        # If region[0].step < 0, parts will be in reversed order
        return torch.concat(parts, dim=0)
    
    def partially_load_by_index(self, index: torch.Tensor, **kwargs) -> torch.Tensor:
        ret = torch.empty(
            (index.shape[0],) + self.shape[1:], dtype=self.dtype, device='cpu'
        )

        _offset = 0
        for i, comp in enumerate(self.components):
            comp_start = _offset
            comp_end = _offset + comp.shape[0]

            mask = torch.logical_and(comp_start <= index, index < comp_end)
            comp_index = index[mask] - comp_start
            comp_slice = comp.partially_load_by_index(comp_index)

            ret[mask] = comp_slice

            _offset = comp_end
        
        return ret

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}' \
            f'(components={repr(self.components)})'