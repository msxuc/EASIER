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
    DataLoaderBase, RegionIndex, Num
from easier.core.runtime.data_loader.utils import \
    compose_slice, get_region_shape, range_unpack, get_strides

from easier.core.runtime.dist_env import \
    get_default_dist_env, get_runtime_dist_env
from easier.core.runtime.utils import check_collective_equality
from easier.core.utils import EasierJitException

class InMemoryTensorLoader(DataLoaderBase):
    """
    Expected to have the same data on all ranks.

    Remarks:
    To initialize with empty data, do not use `torch.empty()`, use
    `torch.zeros()` instead.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        super().__init__()

        self.dtype = tensor.dtype
        self.shape = tensor.shape
        self.device = tensor.device

        # The data is always stored as CPU tensor
        self.tensor = tensor.cpu()

    def collective_init(self) -> None:
        self.coll_check_dtype_shape_devicetype()

        def _eq_tensor(v, v0):
            # torch.allclose support broadcasting, so we need to check shapes.
            return v.shape == v0.shape and torch.allclose(v, v0)
        check_collective_equality(
            f"The input tensor of {self.easier_hint_name}",
            self.tensor,
            eq=_eq_tensor
        )

    @functools.cache
    def minmax(self, index: RegionIndex) -> Tuple[Num, Num]:
        amin, amax = self.tensor[index].aminmax()
        return amin.item(), amax.item()

    @functools.cache
    def count_unique(self, index: RegionIndex) -> int:
        return self.tensor[index].unique().shape[0]

    def partially_load_by_range(self, index: RegionIndex) -> torch.Tensor:
        return self.tensor[index].clone()

    def partially_load_by_index(self, index: torch.Tensor) -> torch.Tensor:
        return self.tensor[index]

    def fully_load(
        self, device: torch.device, replicated: bool
    ) -> torch.Tensor:
        dist_env = get_default_dist_env()
        rank = dist_env.rank
        if replicated or rank == 0:
            return self.tensor.to(device, copy=True)
        else:
            return self.get_placeholder(device)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(tensor={self.tensor})'


def numpy_dtype_to_torch_dtype(np_dtype: np.dtype):
    # torch.from_numpy accepts only:
    # float64, float32, float16, complex64, complex128,
    # int64, int32, int16, int8, uint8, bool
    # Otherwise it raises TypeError.
    return torch.from_numpy(np.ndarray(shape=[0], dtype=np_dtype)).dtype


def torch_dtype_to_numpy_dtype(torch_dtype: torch.dtype):
    return torch.empty([0], dtype=torch_dtype).numpy().dtype


class H5DataLoader(DataLoaderBase):
    """
    Read the specified dataset from rank-0,
    broadcast or distribute to other ranks.
    """

    def __init__(self,
                 h5_file_path: str, h5_dataset_path: str,
                 *,
                 device: Union[torch.device, str],
                 # Optional reading configs for users to load the dataset.
                 dtype: Optional[torch.dtype],
                 **h5_file_kwargs) -> None:
        """
        The constructor only is collective.
        """
        super().__init__()

        self._file_path = os.path.expanduser(h5_file_path)
        self._dataset_path = h5_dataset_path
        self._file_kwargs = h5_file_kwargs

        self.device = torch.device(device)

        self.chunk_size = 1024 * 1024 * 128  # roughly 128M elements

        dist_env = get_default_dist_env()  # runtime dist env not decided yet
        if dist_env.rank == 0:
            with h5py.File(self._file_path, 'r', **self._file_kwargs) as f:
                d = f[self._dataset_path]
                if not isinstance(d, h5py.Dataset):
                    raise TypeError()

                raw_np_dtype = cast(np.dtype, d.dtype)
                self.shape = tuple(d.shape)

            if dtype is not None:
                self._target_np_dtype = torch_dtype_to_numpy_dtype(dtype)
                self.dtype = dtype
            else:
                self._target_np_dtype = None
                self.dtype = numpy_dtype_to_torch_dtype(raw_np_dtype)

            dist_env.broadcast_object_list(
                0, [self._target_np_dtype, self.dtype, self.shape]
            )

        else:
            [self._target_np_dtype, self.dtype, self.shape] = \
                dist_env.broadcast_object_list(0)

    def collective_init(self) -> None:
        # Simply to additionally check device
        self.coll_check_dtype_shape_devicetype()

        # Since H5 paths are only required on rank-0, let's not check them
        # collectively.

    @contextmanager
    def _dataset_as_dtype(self):
        """
        Temporarily open the H5 File and cast the Dataset to the target dtype.
        After reading, the dataset should be closed in time to free memeory.

        Only callable on rank-0.
        """
        # NOTE we cannot really check rank == 0 here as in some situations
        # we haven't yet initialized the DistEnv.
        # assert rank == 0

        with h5py.File(self._file_path, 'r', **self._file_kwargs) as f:
            d = f[self._dataset_path]
            assert isinstance(d, h5py.Dataset)
            if self._target_np_dtype is not None:
                # NOTE the result type of `astype` has no attr `.shape/dtype`.
                d = d.astype(self._target_np_dtype)  # type: ignore

            yield d
    

    def _load_by_chunk(
        self, region: Sequence[RegionIndex]
    ) -> Iterator[torch.Tensor]:
        """
        Only callable on rank-0.
        """
        batch_idx = region[0]
        if isinstance(batch_idx, int):
            with self._dataset_as_dtype() as d:
                chunk_np: np.ndarray = d[batch_idx, *region[1:]]
                chunk: torch.Tensor = torch.from_numpy(chunk_np)
                yield chunk

        elif isinstance(batch_idx, slice):
            batch_range = range(*range_unpack(batch_idx))
            view_len = len(batch_range)

            # Put tailing elements in an individual chunk whose size is smaller.
            nchunk, remainder = divmod(view_len, self.chunk_size)
            if remainder > 0:
                nchunk += 1

            with self._dataset_as_dtype() as d:
                start = batch_idx.start

                # TODO slice util: split_slice(): Iterator[slice]
                for i in range(nchunk):
                    end = start + batch_idx.step * self.chunk_size * i

                    # TODO unit test this H5 view
                    # end may go beyond `self.shape[0]` if step > 0
                    # or beyond 0 if step < 0,
                    # but H5 would ignore the out-of-range part.
                    view_chunk_slice = slice(start, end, batch_idx.step)

                    chunk_np: np.ndarray = d[view_chunk_slice, *region[1:]]
                    chunk: torch.Tensor = torch.from_numpy(chunk_np)
                    yield chunk

                    start = end

        else:
            assert False, 'unreachable'


    @functools.cache
    def minmax(self, index: RegionIndex) -> Tuple[Num, Num]:
        if self.dtype.is_floating_point:
            raise NotImplementedError("Not supporting floats yet")

        dist_env = get_runtime_dist_env()
        if dist_env.rank == 0:
            # TODO basically this is only used for idx, which are ints,
            # but if we want this to be a universal component, we need to
            # ensure float.NaN etc. work as expected.
            amin, amax = None, None

            def _opt_cmp(a: Optional[torch.Tensor], c: torch.Tensor, op):
                return c if a is None else op(a, c)

            for chunk in self._load_by_chunk(index):
                chunk_min, chunk_max = torch.aminmax(chunk)
                amin = _opt_cmp(amin, chunk_min, min)
                amax = _opt_cmp(amax, chunk_max, max)

            amin, amax = amin.item(), amax.item()  # type: ignore

            dist_env.broadcast_object_list(0, [amin, amax])

        else:
            [amin, amax] = dist_env.broadcast_object_list(0)

        return amin, amax

    @functools.cache
    def count_unique(self, index: RegionIndex) -> int:
        if self.dtype.is_floating_point:
            raise NotImplementedError("Not supporting floats yet")

        amin, amax = self.minmax(index)
        if not (amin >= 0):
            raise NotImplementedError("simplify for Reducer.fullness cases")
        assert isinstance(amax, int)

        dist_env = get_runtime_dist_env()
        if dist_env.rank == 0:
            nunique = 0

            # Each time we count elements that fall in the pack,
            # in case the pack gets too big;
            # For each such pack, traverse all .idx data and "set the bit" and
            # count "bits".

            bitpack_maxlen = 1024 * 1024 * 128  # 128MB with bools

            # TODO for Reducer.fullness cases, amax upperbound is number of
            # vertices, so this bitpack won't be too big. But generally the
            # amax is not bounded, causing the bitpack super sparse.
            bitpack_n, remainder = divmod(amax, bitpack_maxlen)
            if remainder > 0:
                bitpack_n += 1

            # TODO use real bitmap and popcount instead of *bool*pack.
            for bitpack_i in range(bitpack_n):
                bitpack_min = bitpack_i * bitpack_maxlen
                bitpack_max = min((bitpack_i + 1) * bitpack_maxlen, amax)

                bitpack = torch.zeros(
                    [bitpack_max - bitpack_min], dtype=torch.bool
                )

                for chunk in self._load_by_chunk(index):
                    in_bitpack = torch.logical_and(
                        chunk >= bitpack_min, chunk < bitpack_max)
                    bitpack[chunk[in_bitpack] - bitpack_min] = 1

                bitpack_nnz = int(torch.count_nonzero(bitpack))
                nunique += bitpack_nnz

            dist_env.broadcast_object_list(0, [nunique])

        else:
            [nunique] = dist_env.broadcast_object_list(0)

        return nunique


    def partially_load_by_range(self, index: RegionIndex) -> torch.Tensor:
        dist_env = get_runtime_dist_env()
        rank = dist_env.rank

        idx_world = dist_env.gather_object_list(0, index)  # type: ignore

        # To avoid OOM, we cannot load the whole dataset on rank-0 then
        # simply call dist.scatter.
        # Instead, we load the part for each rank once, and do P2P.
        if rank == 0:
            idx_world: List[RegionIndex]

            with self._dataset_as_dtype() as d:
                for w in range(1, dist_env.world_size):
                    part_np: np.ndarray = d[idx_world[w]]
                    part: torch.Tensor = \
                        torch.from_numpy(part_np).to(dist_env.comm_device)
                    isend = dist_env.def_isend(part, dst=w, tag=w)
                    for req in dist_env.batch_isend_irecv([isend]):
                        req.wait()

                    # TODO each rank-0-rank-w comm may take a while,
                    # subsequennt recvs should not timeout.

                part0_np: np.ndarray = d[index]
                part0 = torch.from_numpy(part0_np)
                return part0

        else:
            shape = get_region_shape(self.shape, [index])
            buffer = torch.empty(
                shape, dtype=self.dtype, device=dist_env.comm_device
            )
            irecv = dist_env.def_irecv(buffer, src=0, tag=rank)
            for req in dist_env.batch_isend_irecv([irecv]):
                req.wait()

            return buffer.cpu()


    def partially_load_by_index(self, index: torch.Tensor) -> torch.Tensor:
        """
        Each time, rank-0 broadcasts a chunk [chunk_size*i, chunk_size*(i+1))
        to all ranks, and each rank picks the part it needs by
        intersecting with `index`.

        Args:
        - index: element index in the global index space, may be not ordered.
        """
        sorted_index, sort_pos = torch.sort(index, stable=True)

        dist_env = get_runtime_dist_env()

        orig_len = self.shape[0]
        sub_shape = self.shape[1:]

        # Put tailing elements in an individual chunk whose size is smaller.
        nchunk, remainder = divmod(orig_len, self.chunk_size)
        if remainder > 0:
            nchunk += 1

        local_parts = []
        rev_poses = []

        def _run(d):
            for i in range(nchunk):
                start = self.chunk_size * i
                end = min(orig_len, self.chunk_size * (i + 1))

                if dist_env.rank == 0:
                    chunk_np: np.ndarray = d[start:end]
                    chunk: torch.Tensor = \
                        torch.from_numpy(chunk_np).to(dist_env.comm_device)
                    chunk = dist_env.broadcast(src=0, tensor=chunk)

                else:
                    # similar to halo calculation in dist_pass,
                    # but the chunk is defined by a pair (start, end)
                    # TODO therefore for sparse cases
                    # we can use P2P instead of broadcasting.
                    chunk = dist_env.broadcast(
                        src=0, shape=(end - start,) + sub_shape,
                        dtype=self.dtype
                    )

                chunk = chunk.cpu()

                region = torch.logical_and(
                    sorted_index >= start, sorted_index < end)
                local_idx = sorted_index[region] - start

                local_part = chunk[local_idx]
                local_parts.append(local_part)

                rev_pos = sort_pos[region]
                rev_poses.append(rev_pos)

            if len(local_parts) == 0:
                return torch.empty((0,) + self.shape[1:], dtype=self.dtype)
            else:
                data = torch.concat(local_parts)
                pos = torch.concat(rev_poses)

                res = torch.empty_like(data)
                res[pos, ...] = data
                return res

        if dist_env.rank == 0:
            with self._dataset_as_dtype() as d:
                return _run(d)
        else:
            return _run(None)

    def fully_load(
        self, device: torch.device, replicated: bool
    ) -> torch.Tensor:
        """
        Called by backend=='none' case, only default_dist_env is available.
        """
        dist_env = get_default_dist_env()
        rank = dist_env.rank

        if replicated:
            if rank == 0:
                with self._dataset_as_dtype() as d:
                    t = torch.from_numpy(d[...]).to(dist_env.comm_device)
                    dist_env.broadcast(0, t)
            else:
                t = dist_env.broadcast(0, shape=self.shape, dtype=self.dtype)
            return t.to(device)

        else:
            if rank == 0:
                with self._dataset_as_dtype() as d:
                    t = torch.from_numpy(d[...])
                return t.to(device)

            else:
                # We cannot call .to(device) on the placeholder
                # (therefore we have to calls .to(device) many times above)
                ph = self.get_placeholder(device)
                return ph

    def __repr__(self) -> str:
        return ''.join([
            f'{self.__class__.__name__}(',
            # TODO we didn't escape the path strings properly
            f'h5_file_path={self._file_path}, ',
            f'h5_dataset_path={self._dataset_path}, ',
            f'dtype={self.dtype}',
            ')'
        ])


class FulledTensorLoader(DataLoaderBase):
    def __init__(self, value: Union[int, float], shape, dtype, device) -> None:
        super().__init__()

        self.value = value
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = torch.device(device)

        # Views are not effective for FullTensorLoader.

    def collective_init(self) -> None:
        self.coll_check_dtype_shape_devicetype()
        check_collective_equality(
            f"fill value of {self.easier_hint_name}", self.value
        )

    def minmax(self, index) -> Tuple[Num, Num]:
        return self.value, self.value

    def count_unique(self, index) -> int:
        return 1

    def _full(self, shape, device: Union[torch.device, str]):
        return torch.full(
            shape, self.value, dtype=self.dtype, device=device)  # type: ignore

    def partially_load_by_range(self, index: RegionIndex) -> torch.Tensor:
        shape = get_region_shape(self.shape, [index])
        return self._full(shape, 'cpu')

    def partially_load_by_index(self, index: torch.Tensor) -> torch.Tensor:
        shape = (index.shape[0],) + self.shape[1:]
        return self._full(shape, 'cpu')

    def fully_load(
        self, device: torch.device, replicated: bool
    ) -> torch.Tensor:
        dist_env = get_default_dist_env()
        rank = dist_env.rank
        if replicated or rank == 0:
            return self._full(None, device)
        else:
            return self.get_placeholder(device)

    def __repr__(self) -> str:
        return ''.join([
            f'{self.__class__.__name__}(',
            f'value={self.value}, ',
            f'shape={self.shape}, ',
            f'dtype={self.dtype}',
            ')'
        ])


class ArangeTensorLoader(DataLoaderBase):
    def __init__(
        self,
        start: Num,
        end: Num,
        step: Num,
        dtype: torch.dtype,
        device: Union[str, torch.device]
    ):
        super().__init__()

        if step == 0:
            raise ValueError("step must not be 0")
        if math.isinf(start) or math.isinf(end):
            raise ValueError(f"range cannot be {start} to {end}")

        # # TODO torch.arange rejects (0, 10, -1) -- sign must be consistent
        # # -- but numpy allows and returns empty list.
        # if not (
        #     (step > 0 and end >= start) or (step < 0 and end <= start)
        # ):
        #     raise ValueError("inconsistent step sign")

        if dtype.is_complex:
            raise NotImplementedError("range cannot be complex")

        self._start = start
        self._end = end
        self._step = step

        # TODO both torch and numpy have very careful calculation for length,
        # we need to reinforce this.
        length = math.ceil((end - start) / step)
        length = max(0, length)
        
        self.shape = (length,)
        self.dtype = dtype
        self.device = torch.device(device)

    def collective_init(self) -> None:
        self.coll_check_dtype_shape_devicetype()
        check_collective_equality(
            f"arange of {self.easier_hint_name}",
            [self._start, self._end, self._step]
        )

    def minmax(self, index: RegionIndex) -> Tuple[Num, Num]:
        s = compose_slice(slice(self._start, self._end, self._step), index)
        l = len(range(*range_unpack(s)))
        if s.step > 0:
            return s.start, s.start + l * s.step
        else:
            return s.start + l * s.step, s.start

    def count_unique(self, index: RegionIndex) -> int:
        s = compose_slice(slice(self._start, self._end, self._step), index)
        l = len(range(*range_unpack(s)))
        return l


    def partially_load_by_rank_REMOVE_THIS(self) -> Tuple[torch.Tensor, int, int]:
        dist_env = get_runtime_dist_env()
        rank = dist_env.rank
        orig_len = self.shape[0]
        offset_start, offset_end = get_offset_exactly_nparts(
            orig_len, dist_env.world_size, rank)

        range_start = self._start + offset_start * self._step
        range_end = self._start + offset_end * self._step
        if self._step > 0:
            range_end = min(self._end, range_end)
        else:
            range_end = max(self._end, range_end)

        chunk = torch.arange(range_start, range_end, self._step,
                             dtype=self.dtype, device='cpu')
        return chunk, offset_start, offset_end

    def partially_load_by_index(self, index: torch.Tensor) -> torch.Tensor:
        return (index * self._step + self._start).to(dtype=self.dtype)

    def fully_load(
        self, device: torch.device, replicated: bool
    ) -> torch.Tensor:
        dist_env = get_default_dist_env()
        rank = dist_env.rank
        if replicated or rank == 0:
            return torch.arange(
                self._start, self._end, self._step,
                dtype=self.dtype, device=device
            )
        else:
            return self.get_placeholder(device)

    def __repr__(self) -> str:
        return ''.join([
            f'{self.__class__.__name__}(',
            f'start={self._start}, ',
            f'end={self._end}, ',
            f'step={self._step}, ',
            f'dtype={self.dtype}',
            ')'
        ])