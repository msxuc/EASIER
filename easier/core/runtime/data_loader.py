# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from contextlib import contextmanager
from typing import Iterator, Optional, Tuple, Union, cast
import h5py

import numpy as np
import torch

from easier.core.runtime.dist_env import get_cpu_dist_env


ATTRIBUTE_PLACEHOLDER = "easier_placeholder"


def _get_offset_exactly_nparts(orig_len: int, nparts: int, part: int
                               ) -> Tuple[int, int]:
    """
    A part will have roughly `orig_len // nparts` elements.
    The remaining elements will be put in the last part.

    Please note how the remaining elements are handled. When the remaining
    elements are treated as an individial part, this auxiliary method cannot
    be used.
    """
    per_worker_len = orig_len // nparts

    start = per_worker_len * part

    if part + 1 == nparts:
        end = orig_len
    else:
        end = per_worker_len * (part + 1)

    return start, end


class DataLoaderBase:
    """
    The data loader for one specified data source, e.g. a HDF5 dataset.

    Calls to every method should be collective.

    # TODO ensure whether the result tensor is always a new clone to avoid
    accidently modify the views, if any.
    """

    def __init__(self) -> None:
        pass

    def partially_load_by_chunk(self, chunk_size: int
                                ) -> Iterator[torch.Tensor]:
        """
        Only callable at rank-0.

        Chuck size is only about the first dimension and item tensors in
        the resultant sequence:
        - always on CPU;
        - may not have batch size that exactly equals chunk_size.
        """
        raise NotImplementedError()

    def partially_load_by_rank(self) -> Tuple[torch.Tensor, int, int]:
        """
        Collectively load an evenly distributed part of the target dataset
        for each rank.

        Returns:
        - torch.Tensor: the loaded part, always on CPU
        - int: the beginning offset of the part (inclusive)
        - int: the end offset of the part (exclusive)
        """
        raise NotImplementedError()

    def partially_load_by_index(self, index: torch.Tensor, **kwargs
                                ) -> torch.Tensor:
        """
        Collectively load a part of the target dataset with the
        specified index tensor.
        The index is defined in the global index space.

        Args:
        - index: should always be on CPU
        - kwargs: subtype-specific config

        Returns:
        - torch.Tensor: the loaded part, always on CPU
        """
        raise NotImplementedError()

    def fully_load(self, device: Union[torch.device, str, None]
                   ) -> torch.Tensor:
        """
        When fully loading the dataset.

        Args:
        - device:
            If None, the initial device specified on the data loader is used;
            otherwise, load the data to the input device.

        Returns:
        - torch.Tensor: the full tensor, on the resolved device.
        """
        raise NotImplementedError()

    @property
    def dtype(self) -> torch.dtype:
        raise NotImplementedError()

    @property
    def shape(self) -> Tuple[int, ...]:
        raise NotImplementedError()

    @property
    def device(self) -> torch.device:
        """
        The device to where the data will be initially loaded.
        This device configuration only take effect with "torch" JIT backend.
        """
        raise NotImplementedError()

    def get_placeholder(self) -> torch.Tensor:
        """
        Allocate a new placeholder torch.Tensor of the same dtype/shape/device
        but generally consuming no memory to be compatible with cases where
        torch.Tensor object and information is needed.

        Any subclass implementation should add the attribute
        "easier_placeholder" to indicate the result is a placeholder, too.

        The placeholder is needed mainly to:
        1.  ease the inspection of tensor properties like dtype/shape/device,
            especially for the metadata pass.
        2.  fulfil `esr.Tensor.__new__(cls, data)` where the underlying data
            tensor should be set.
        """
        # torch.Tensor.expand can expand shape-(1,) to e.g. shape-(0,0,0),
        # but not to ndim=0 shape ().
        if len(self.shape) == 0:
            ph = torch.zeros((), dtype=self.dtype, device=self.device)
        else:
            ph = torch.zeros(
                (1,), dtype=self.dtype, device=self.device
            ).expand(self.shape)  # can even expand to (0,0,0)

        setattr(ph, ATTRIBUTE_PLACEHOLDER, True)
        return ph


class InMemoryTensorLoader(DataLoaderBase):
    # Generally all ops are no-ops.
    def __init__(self, tensor: torch.Tensor) -> None:
        super().__init__()

        # This tensor may be on any device.
        self.tensor = tensor

    def partially_load_by_chunk(self, chunk_size: int
                                ) -> Iterator[torch.Tensor]:
        cpu_dist_env = get_cpu_dist_env()
        assert cpu_dist_env.rank == 0, \
            "Loading-by-chunk is only available on rank-0"

        orig_len = self.tensor.shape[0]

        # Put tailing elements in an individual chunk whose size is smaller.
        nchunk, remainder = divmod(orig_len, chunk_size)
        if remainder > 0:
            nchunk += 1

        # After we have decided a valid nchunk (>=1), it won't matter even if
        # `_get_offset_exactly_nparts` to get offsets.
        # But we still follow the chunk partition above.
        for i in range(nchunk):
            start = chunk_size * i
            end = min(orig_len, chunk_size * (i + 1))
            chunk = self.tensor[start:end].to('cpu', copy=True)
            yield chunk

    def partially_load_by_rank(self) -> Tuple[torch.Tensor, int, int]:
        cpu_dist_env = get_cpu_dist_env()
        world_size = cpu_dist_env.world_size
        rank = cpu_dist_env.rank
        orig_len = self.tensor.shape[0]

        # Put tailing elements in the part for the last rank, making the size
        # of that part bigger than chunk_size.
        start, end = _get_offset_exactly_nparts(orig_len, world_size, rank)

        return self.tensor[start:end].to('cpu', copy=True), start, end

    def partially_load_by_index(self, index: torch.Tensor,
                                **kwargs) -> torch.Tensor:
        assert index.device == torch.device('cpu')
        return self.tensor.to('cpu')[index]

    def fully_load(self, device: Union[torch.device, str, None]
                   ) -> torch.Tensor:
        return self.tensor.to(device or self.device, copy=True)

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.tensor.shape

    @property
    def device(self) -> torch.device:
        return self.tensor.device


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
                 # Optional reading configs for users to load the dataset.
                 dtype: Optional[torch.dtype] = None,
                 device: Union[torch.device, str] = 'cpu',
                 **h5_file_kwargs) -> None:
        super().__init__()

        self._file_path = h5_file_path
        self._file_kwargs = h5_file_kwargs
        self._dataset_path = h5_dataset_path

        self._device = torch.device(device)

        self._target_np_dtype = None

        cpu_dist_env = get_cpu_dist_env()
        if cpu_dist_env.rank == 0:
            with self._dataset() as d:
                assert isinstance(d, h5py.Dataset)
                raw_np_dtype = cast(np.dtype, d.dtype)
                raw_dtype = numpy_dtype_to_torch_dtype(raw_np_dtype)

                self._shape = tuple(d.shape)

            if dtype is not None:
                if not isinstance(dtype, torch.dtype):
                    raise TypeError()
                self._dtype = dtype
            else:
                self._dtype = raw_dtype

            cpu_dist_env.broadcast_object_list(0, [self._dtype, self._shape])

        else:
            self._dtype, self._shape = \
                cpu_dist_env.broadcast_object_list(0)

        self._target_np_dtype = torch_dtype_to_numpy_dtype(self.dtype)

    @contextmanager
    def _dataset(self):
        """
        Temporarily open the H5 File and Dataset.
        After reading, the dataset should be closed in time to free memeory.
        """
        cpu_dist_env = get_cpu_dist_env()
        assert cpu_dist_env.rank == 0

        with h5py.File(self._file_path, 'r', **self._file_kwargs) as f:
            d = f[self._dataset_path]
            if not isinstance(d, h5py.Dataset):
                raise TypeError()

            if self._target_np_dtype is not None:
                # NOTE the result type of `astype` has no attr `.shape/dtype`.
                d = d.astype(self._target_np_dtype)

            yield d

    def partially_load_by_chunk(self, chunk_size: int
                                ) -> Iterator[torch.Tensor]:
        cpu_dist_env = get_cpu_dist_env()
        assert cpu_dist_env.rank == 0, \
            "Loading-by-chunk is only available on rank-0"

        orig_len = self.shape[0]

        # Put tailing elements in an individual chunk whose size is smaller.
        nchunk, remainder = divmod(orig_len, chunk_size)
        if remainder > 0:
            nchunk += 1

        with self._dataset() as d:
            for i in range(nchunk):
                start = chunk_size * i
                end = min(orig_len, chunk_size * (i + 1))

                chunk_np: np.ndarray = d[start:end]
                chunk: torch.Tensor = torch.from_numpy(chunk_np)
                yield chunk

    def partially_load_by_rank(self) -> Tuple[torch.Tensor, int, int]:
        cpu_dist_env = get_cpu_dist_env()
        rank = cpu_dist_env.rank

        orig_len = self.shape[0]
        sub_shape = self.shape[1:]

        # To avoid OOM, we cannot load the whole dataset on rank-0 then
        # simply call dist.scatter.
        # Instead, we load the part for each rank once, and do P2P.
        if rank == 0:
            with self._dataset() as d:
                for w in range(1, cpu_dist_env.world_size):
                    start, end = _get_offset_exactly_nparts(
                        orig_len, nparts=cpu_dist_env.world_size, part=w)

                    part_np: np.ndarray = d[start:end]
                    part: torch.Tensor = torch.from_numpy(part_np)
                    isend = cpu_dist_env.def_isend(part, dst=w, tag=w)
                    for req in cpu_dist_env.batch_isend_irecv([isend]):
                        req.wait()

                    # TODO each rank-0-rank-w comm may take a while,
                    # subsequennt recvs should not timeout.

                s0, e0 = _get_offset_exactly_nparts(
                    orig_len, nparts=cpu_dist_env.world_size, part=0)
                part0_np: np.ndarray = d[s0:e0]
                part0 = torch.from_numpy(part0_np)
                return part0, s0, e0

        else:
            start, end = _get_offset_exactly_nparts(
                orig_len, cpu_dist_env.world_size, rank)
            buffer = torch.empty((end - start,) + sub_shape, dtype=self.dtype)
            irecv = cpu_dist_env.def_irecv(buffer, src=0, tag=rank)
            for req in cpu_dist_env.batch_isend_irecv([irecv]):
                req.wait()

            return buffer, start, end

    def partially_load_by_index(
        self, index: torch.Tensor, *,
        chunk_size=1024 * 1024 * 128  # roughly 128M elements
    ) -> torch.Tensor:
        """
        Each time, rank-0 broadcasts a chunk [chunk_size*i, chunk_size*(i+1))
        to all ranks, and each rank picks the part it needs by
        intersecting with `index`.

        Args:
        - index: element index in the global index space, must be ordered
            and on CPU.
        """
        assert index.device == torch.device('cpu')

        cpu_dist_env = get_cpu_dist_env()

        orig_len = self.shape[0]
        sub_shape = self.shape[1:]

        # Put tailing elements in an individual chunk whose size is smaller.
        nchunk, remainder = divmod(orig_len, chunk_size)
        if remainder > 0:
            nchunk += 1

        local_parts = []

        def _run(d):
            for i in range(nchunk):
                start = chunk_size * i
                end = min(orig_len, chunk_size * (i + 1))

                if cpu_dist_env.rank == 0:
                    chunk_np: np.ndarray = d[start:end]
                    chunk: torch.Tensor = torch.from_numpy(chunk_np)
                    chunk = cpu_dist_env.broadcast(src=0, tensor=chunk)

                else:
                    # similar to halo calculation in dist_pass,
                    # but the chunk is defined by a pair (start, end)
                    # TODO therefore for sparse cases
                    # we can use P2P instead of broadcasting.
                    chunk = cpu_dist_env.broadcast(
                        src=0, shape=(end - start,) + sub_shape,
                        dtype=self.dtype)

                local_idx = index[
                    torch.logical_and(index >= start, index < end)
                ] - start

                local_part = chunk[local_idx]
                local_parts.append(local_part)

            if len(local_parts) == 0:
                return torch.empty((0,) + self.shape[1:], dtype=self.dtype)
            else:
                return torch.concat(local_parts)

        if cpu_dist_env.rank == 0:
            with self._dataset() as d:
                return _run(d)
        else:
            return _run(None)

    def fully_load(self, device: Union[torch.device, str, None]
                   ) -> torch.Tensor:
        cpu_dist_env = get_cpu_dist_env()

        if cpu_dist_env.rank == 0:
            with self._dataset() as d:
                t = torch.from_numpy(d[...])
                cpu_dist_env.broadcast(0, t)
        else:
            t = cpu_dist_env.broadcast(0, shape=self.shape, dtype=self.dtype)

        return t.to(device=device or self.device)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def device(self) -> torch.device:
        return self._device


class FulledTensorLoader(DataLoaderBase):
    def __init__(self, value: Union[int, float], shape, dtype, device) -> None:
        super().__init__()

        self.value = value
        self._shape = tuple(shape)
        self._dtype = dtype
        self._device = device

    def _full(self, batch_dim_len: Optional[int] = None,
              device: Union[torch.device, str, None] = None):
        if batch_dim_len is None:
            batch_dim_len = self._shape[0]

        shape = (batch_dim_len,) + self._shape[1:]
        return torch.full(
            shape, self.value, dtype=self.dtype, device=device)  # type: ignore

    def partially_load_by_chunk(self, chunk_size: int
                                ) -> Iterator[torch.Tensor]:
        cpu_dist_env = get_cpu_dist_env()
        assert cpu_dist_env.rank == 0, \
            "Loading-by-chunk is only available on rank-0"

        orig_len = self.shape[0]

        # Put tailing elements in an individual chunk whose size is smaller.
        nchunk, remainder = divmod(orig_len, chunk_size)
        if remainder > 0:
            nchunk += 1

        for i in range(nchunk):
            start = chunk_size * i
            end = min(orig_len, chunk_size * (i + 1))

            chunk = self._full(end - start, 'cpu')
            yield chunk

    def partially_load_by_rank(self) -> Tuple[torch.Tensor, int, int]:
        cpu_dist_env = get_cpu_dist_env()
        rank = cpu_dist_env.rank
        orig_len = self._shape[0]
        start, end = _get_offset_exactly_nparts(
            orig_len, cpu_dist_env.world_size, rank)
        return self._full(end - start, 'cpu'), start, end

    def partially_load_by_index(self, index: torch.Tensor,
                                **kwargs) -> torch.Tensor:
        return self._full(index.shape[0], 'cpu')

    def fully_load(self, device: Union[torch.device, str, None]
                   ) -> torch.Tensor:
        return self._full(None, device=device or self.device)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def device(self) -> torch.device:
        return self._device
