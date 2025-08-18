# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
from typing import List, Sequence, Tuple, Union

import torch

from easier.core.runtime.data_loader.base import \
    DataLoaderBase, NormalizedSlice, Num
from easier.core.runtime.data_loader.utils import \
    compose_slice, get_overlapping_slice, get_strides, range_unpack

from easier.core.runtime.dist_env import \
    get_default_dist_env
from easier.core.runtime.utils import check_collective_equality



class StridedDataLoader(DataLoaderBase):
    def __init__(
        self, inner: DataLoaderBase, index: Sequence[Union[NormalizedSlice, int]]
    ):
        super().__init__()

        if len(index) > len(inner.shape):
            raise IndexError(
                f"Too many indices for input with ndim=={len(inner.shape)}"
            )

        for i, idx in enumerate(index):
            if isinstance(idx, int):
                if not (0 <= idx < inner.shape[i]):
                    raise IndexError(
                        f"Index {idx} is out-of-range for dimension {i}"
                        f" with length {inner.shape[i]}"
                    )

            elif isinstance(idx, NormalizedSlice):
                1

            else:
                raise IndexError(f"Unexpected index {idx}")

        # The index must be converted to valid and in-range values.
        self.index = index
        self.inner = inner

        shape = []
        for idx in index:
            if isinstance(idx, NormalizedSlice):
                shape.append(len(idx))
        shape += list(self.inner.shape[len(index):])

        self.shape = tuple(shape)
        self.dtype = inner.dtype
        self.device = inner.device

        # Given `self.index` may contain ints, the dimensions related to those
        # ints are discarded, i.e.:
        assert len(self.shape) <= len(self.inner.shape)
    
    def _compose_region(
        self, region: Sequence[NormalizedSlice]
    ) -> Sequence[NormalizedSlice]:
        # If `self.index` contains ints, those dimensions are excluded during
        # region composition.
        region_i = 0

        composed_region = []
        for idx in self.index:
            if isinstance(idx, int):
                composed_region.append(slice(idx, idx + 1))

            elif isinstance(idx, slice):
                if region_i < len(region):
                    region_slice = region[region_i]
                    composed_slice = compose_slice(idx, region_slice)
                    composed_region.append(composed_slice)

                    region_i += 1
                else:
                    composed_region.append(idx)

            else:
                assert False, 'unreachable'

        return composed_region
    
    def minmax(self, index: NormalizedSlice) -> Tuple[Num, Num]:
        composed_indices = self._compose_region(index)
        return self.inner.minmax(composed_indices)

    def count_unique(self, index: NormalizedSlice) -> int:
        composed_region = self._compose_region(index)
        return self.inner.count_unique(composed_region)

    def partially_load_by_range(self, index: NormalizedSlice) -> torch.Tensor:
        bs_idx = self.index[0]
        if isinstance(bs_idx, int):
            idxed_inner = self.inner.partially_load_by_range(
                slice(bs_idx, bs_idx + 1)
            )
            return idxed_inner[0, *self.index[1:]][index]
        else:
            composed_idx = compose_slice(bs_idx, index)
            return self.inner.partially_load_by_range(composed_idx)

    def partially_load_by_index(self, index: torch.Tensor) -> torch.Tensor:
        bs_idx = self.index[0]
        if isinstance(bs_idx, int):
            idxed_inner = self.inner.partially_load_by_range(
                slice(bs_idx, bs_idx + 1)
            )
            return idxed_inner[0, *self.index[1:]][index]
        else:
            composed_idx = bs_idx.start + bs_idx.step * index
            return self.inner.partially_load_by_index(composed_idx)

    def fully_load(
        self, device: torch.device, replicated: bool
    ) -> torch.Tensor:
        return self.inner.fully_load(device, replicated)[*self.index]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}' \
            f'(inner={self.inner}, index={self.index})'


class CartesianProductDataLoader(DataLoaderBase):
    """
    Takes N input DataLoader component, the i-th component must have its
    shape in form of `(L_i,)`, and all components must have the same dtype.

    The result shape is `(L_0*...*L_{N-1}, N)`.
    """
    def __init__(
        self,
        components: Sequence[DataLoaderBase],
        # TODO to support item datatype other than scalar, we may need 'form':
        # form: Literal['flatten', 'stack'] = 'flatten'
    ):
        super().__init__()

        nd_sizes = []

        dtypes = []
        devices = []
        for i, dl in enumerate(components):
            if len(dl.shape) != 1:
                raise ValueError(f"{i}-th input ndim != 1")
            nd_sizes.append(dl.shape[0])

            dtypes.append(dl.dtype)
            devices.append(dl.device)
        
        if len(set(dtypes)) != 1:
            raise ValueError("Input dtypes must be the same")
        if len(set(devices)) != 1:
            raise ValueError("Input devices must be the same")
        
        self.shape = (math.prod(nd_sizes), len(components))
        self.dtype = dtypes[0]
        self.device = devices[0]

        self._nd_sizes = nd_sizes
        self.components = list(components)

        self._chunk_size = 128 * 1024 * 1024
    
    def partially_load_by_range(self, index: NormalizedSlice) -> torch.Tensor:
        index_tensor = torch.arange(*range_unpack(index), dtype=torch.int64)
        return self.partially_load_by_index(index_tensor)

    def partially_load_by_index(self, index: torch.Tensor) -> torch.Tensor:
        nd_strides: List[int] = get_strides(self._nd_sizes).tolist()

        ret = torch.empty(
            [index.shape[0], len(self.components)],
            dtype=self.dtype
        )

        # Load by chunks for each component DataLoader, in extreme cases
        # their shapes may be like (1e9,) and (2,) and need partitioning.
        nd_sizes = torch.tensor(self._nd_sizes, dtype=torch.int64)

        chunk_size = self._chunk_size
        nd_nchunks = nd_sizes // chunk_size
        have_remainder = (nd_sizes % chunk_size) > 0
        nd_nchunks = (nd_nchunks + have_remainder.to(torch.int64)).tolist()

        chunk_ids_list: List[torch.Tensor] = []
        for idl, dl in enumerate(self.components):
            chunk_ids_list.append(torch.arange(nd_nchunks[idl]))
        # (prod(nchunks), N) -- Combinations of chunk ids
        chunk_ids_combs = torch.cartesian_prod(*chunk_ids_list)

        for icomb in range(chunk_ids_combs.shape[0]):
            chunk_ids = chunk_ids_combs[icomb]

            for idl, dl in enumerate(self.components):
                if nd_nchunks[idl] == 0:
                    continue

                chunk_id = int(chunk_ids[idl])
                comp_size = self._nd_sizes[idl]

                chunk_start = chunk_id * chunk_size
                chunk_end = min(chunk_start + chunk_size, comp_size)
                chunk = dl.partially_load_by_range(
                    slice(chunk_start, chunk_end)
                )

                stride_i = nd_strides[idl]

                comp_idx = (index / stride_i) % comp_size
                if nd_nchunks[idl] == 1:
                    # avoid calculating the mask.
                    ret[:, idl] = chunk[comp_idx]
                else:
                    idx_mask = torch.logical_and(
                        chunk_start <= comp_idx, comp_idx < chunk_end
                    )
                    ret[idx_mask, idl] = chunk[comp_idx[idx_mask]]
        
        return ret

    def fully_load(
        self, device: torch.device, replicated: bool
    ) -> torch.Tensor:
        dist_env = get_default_dist_env()
        rank = dist_env.rank
        if replicated or rank == 0:
            vectors = []
            for comp in self.components:
                vector = comp.fully_load(device, replicated)
                vectors.append(vector)
            return torch.cartesian_prod(*vectors)
        else:
            return self.get_placeholder(device)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(components={repr(self.components)})'

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

        self._lengths = list(d.shape[0] for d in components)
        self.components = list(components)

        self.shape = (sum(self._lengths),) + components[0].shape[1:]
        self.dtype = components[0].dtype
        self.device = components[0].device


    def _foreach_in_region(self, region: NormalizedSlice, fn):
        # fn: (DataLoaderBase, _SimpleIndex) -> None
        # NOTE if region[0] is slice-with-negative-step, foreach in reversed.

        _components = self.components
        if region.step < 0:
            _components = reversed(self.components)

        _offset = 0
        for comp in _components:
            comp_start = _offset
            comp_end = _offset + comp.shape[0]

            overlap = get_overlapping_slice(
                slice(comp_start, comp_end), region
            )
            if overlap.start != overlap.stop:
                comp_batch = slice(
                    overlap.start - comp_start,
                    overlap.stop - comp_end,
                    overlap.step
                )
                fn(comp, comp_batch)

            _offset = comp_end


    def minmax(self, index: NormalizedSlice) -> Tuple[Num, Num]:
        aminmax = [math.inf, -math.inf]
        def _minmax(comp: DataLoaderBase, comp_region: NormalizedSlice):
            comp_minmax = comp.minmax([comp_region] + list(index[1:]))
            aminmax[0] = min(comp_minmax[0], aminmax[0])
            aminmax[1] = max(comp_minmax[1], aminmax[1])
        self._foreach_in_region(index[0], _minmax)
        return tuple(aminmax)  # type: ignore

    
    def partially_load_by_range(self, index: NormalizedSlice) -> torch.Tensor:
        parts = []
        def _load_comp(comp: DataLoaderBase, comp_region: NormalizedSlice):
            parts.append(comp.partially_load_by_range(comp_region))
        self._foreach_in_region(index, _load_comp)
        # If region[0].step < 0, parts will be in reversed order
        return torch.concat(parts, dim=0)
    
    def partially_load_by_index(self, index: torch.Tensor) -> torch.Tensor:
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
        return f'{self.__class__.__name__}(components={repr(self.components)})'