# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from types import EllipsisType
from typing import Iterator, List, Optional, Tuple, TypeAlias, Union, cast
import h5py
import functools
import copy

import numpy as np
import torch

from easier.core.runtime.dist_env import \
    get_default_dist_env, get_runtime_dist_env
from easier.core.runtime.utils import check_collective_equality
from easier.core.utils import EasierJitException


ATTRIBUTE_PLACEHOLDER = "easier_placeholder"

Num: TypeAlias = Union[int, float, bool]


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


def _wrap_function(pre_hook, post_hook, func):
    # NOTE Python captures capsule by stackframe, a dedicated function like
    # this is required, in case this is called wihtin a loop.
    @functools.wraps(func)
    def wrapper(this, *args, **kwargs):
        if pre_hook is not None:
            pre_hook(this, *args, **kwargs)
        res = func(this, *args, **kwargs)
        if post_hook is not None:
            res = post_hook(this, res, *args, **kwargs)
        return res
    return wrapper


# Python __getitem__ protocol only support one index parameter, and require
# caller to wrap multiple arugments in a tuple (only tuple, not list).
# Given a value of such a Union type, we can directly pass it into
# Tensor.__getitem__ etc. without extra unpacking like `*args`.
_SimpleIndex: TypeAlias = Union[
    int, slice, EllipsisType, None,
    Tuple['_SimpleIndex', ...]
]
_GeneralIndex: TypeAlias = Union[
    int, slice, EllipsisType, None, torch.Tensor,
    Tuple['_GeneralIndex', ...]
]



"""
TODO

Passes like tensor_partitioning and sparse_encoding (and potentially codegen)
leverages properties of idx tensors like being arange-d, being bounded or
being ordered to boost index calculation. The idx tensors are all originally
loaded by DataLoaders.
However, DataLoader internal methods currently return Tensors only,
therefore we didn't recognize such properties in the 1st place.
And the passes themselves maintain then discard the record objects that
indicate such properties, without making a global effort to leverage it.

Given the data-oriented nature of EASIER AOT, we may make every subsystem in
AOT return symbolic representation of data.
It's not the `pass.py` itself, but the interpreter in another layer to evaluate
the symbolic representation, deciding how to do the transformation on data.
To the extreme extent, (the symbolic representation of) simple DataLoaders
like esr.arange might get kept, and in codegen each thread can calculate data
from thread id rather than loading arange-d data from memory.
Nonetheless, CUDA acceleration, GC in AOT can also be handled by it, authors
of passes can focus on compilation logic.

That said, because of the need to balance computation costs for general cases
and the requirement of managing synchronization points of collective
communication APIs,
we may fallback to tensor calculation once when the "generalness" occur again
during AOT.

P.S. DataLoaders themselves and the AOT passes that use torch vectorized
operators (which opened a space for CUDA acceleration) are already similar
approaches, but using different symbol sets from easier.runtime.data_loader
or of torch operators.
"""


class DataLoaderBase:
    """
    The data loader for one specified data source, e.g. a HDF5 dataset.

    Calls to every method should be collective.

    NOTE for subclasses:
    Subclasses should implement each method in a way that suits the use case,
    for example, to load a range we should avoid loading-all-then-slicing.
    
    But this is not always possible, an extreme example may be:
    ```
    class RandomSamplerLoader(DataLoaderBase):
        def __init__(self, inner: DataLoaderBase):
            self.inner = inner
        def partially_load_by_range(self, region):
            random_idx = randint([for dimlen in region])
            return self.inner.partially_load_by_index(random_idx)
    ```
    where we cannot keep the simplicity of range and must fallback to load
    by index tensor.
    """

    def __init__(self) -> None:
        """
        The constructor should do simple member data storage and local tasks.

        When collective operations are needed, implementations should use
        `get_default_dist_env` because the constructors are called by users
        before `esr.compile()`.

        NOTE subclasses should not put communication, especially not call
        `coll_check_dtype_shape_devicetype` etc. in constructors.
        As DataLoaders are not only used by end users (where collective check
        makes sense) but also used by internal passes (where it's not
        collective at all).
        """
        self.shape: Tuple[int, ...]
        self.dtype: torch.dtype

        # The device on which the data loader is intially defined.
        # This device configuration only take effect with "torch" JIT backend.
        self.device: torch.device

        # e.g. "(Module).(a.b.c:Selector).idx"
        # Decided during `esr.compile()`
        self.easier_hint_name: str

    def __init_subclass__(cls) -> None:
        """
        Before a data loader API provided by a subclass runs,
        any _prefilter_ and _postfilter_ defined in this DataLoaderBase will
        run to check the environment and ensure critical requirements are
        satisfied.
        """
        for member_name, member in list(cls.__dict__.items()):
            # cls.__dict__ doesn't contain inherited methods from DistEnv
            if callable(member):
                # both `member` and `pre/post_hook` function objects are not
                # bound to some DataLoader instance yet, the `self` argument
                # will be included at the head in `args` in the wrapper.
                pre_hook = getattr(
                    DataLoaderBase, '_pre_' + member_name, None
                )
                post_hook = getattr(
                    DataLoaderBase, '_post_' + member_name, None
                )
                setattr(
                    cls, member_name,
                    _wrap_function(pre_hook, post_hook, member)
                )

    def coll_check_dtype_shape_devicetype(self):
        check_collective_equality(
            f"Tensor properties of {self.easier_hint_name}",
            [self.dtype, self.shape, self.device.type]
        )

    def collective_init(self) -> None:
        """
        Validate if the the data of this data loader is collectively correct.

        Mainly to validate DataLoaders defiend by users.
        DataLoaders created by EASIER internally generally do not need this.

        Require callers (i.e. collective_initialization pass)
        to first ensure the data loders among workers are
        actually referring to the same data set i.e. of the same type.
        """
        raise NotImplementedError()

    def minmax(self, region: _SimpleIndex) -> Tuple[Num, Num]:
        """
        Get minimum and maximum value within the `region` range of data source.

        TODO minmax() and count_unique() both accept *simple* index as range,
        i.e. tensor-typed index is not allowed.
        This indicates we may need to reimplement minmax()/count_unique() on a
        very detailed, tensor-indexed region if we provides such DataLoaders.
        """
        raise NotImplementedError()

    def count_unique(self, region: _SimpleIndex) -> int:
        """
        Count unique elements in the exact `region` range of data source.

        Used by Reducer.set_fullness()
        """
        raise NotImplementedError()

    def to(
        self,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str]] = None
    ) -> 'DataLoaderBase':
        """
        Return a cloned DataLoader with the specified properties changed.
        """
        clone = copy.deepcopy(self)
        if dtype is not None:
            clone.dtype = dtype
        if device is not None:
            clone.device = torch.device(device)
        return clone

    def __getitem__(self, index: _SimpleIndex) -> 'DataLoaderBase':
        # TODO certain DataLoader stack can be reduced and simplified
        return StridedDataLoader(self, index)

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

    def _pre_partially_load_by_chunk(self, chunk_size):
        dist_env = get_runtime_dist_env()
        assert dist_env.rank == 0, \
            "Loading-by-chunk is only available on rank-0"

    def _post_partially_load_by_chunk(
        self, res: Iterator[torch.Tensor], chunk_size
    ) -> Iterator[torch.Tensor]:
        def _check_item(chunk):
            assert chunk.device.type == 'cpu'
            return chunk

        # `map()` make a new iterator, the `_check_item` is run when the
        # iterator get actually iterated.
        return map(_check_item, res)

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
    
    def _post_partially_load_by_rank(self, res):
        (tensor, begin, end) = res
        assert tensor.device.type == 'cpu'
        return res

    def partially_load_by_range(self, region: _SimpleIndex) -> torch.Tensor:
        raise NotImplementedError()


    def _pre_partially_load_by_range(self, index):
        assert isinstance(index, (int, slice, tuple)) \
            or index in [Ellipsis, None]

    def _post_partially_load_by_range(self, res):
        assert res.device.type == 'cpu'
        return res


    def partially_load_by_index(self, index: torch.Tensor, **kwargs
                                ) -> torch.Tensor:
        """
        Collectively load a part of the target dataset with the
        specified index tensor.
        The index is defined in the global index space.

        Args:
        - index: should always be on CPU
        - kwargs: subtype-specific config

            Common configs:
            -   chunk_size: int

            TODO as DataLoader subsystem gets complicated, other loading API
            may need kwargs config too.
            TODO or any DataLoader needs configs may have configs being
            attributes.

        Returns:
        - torch.Tensor: the loaded part, always on CPU
        """
        raise NotImplementedError()

    def _pre_partially_load_by_index(self, index, **kwargs):
        assert index.device.type == 'cpu'

    def _post_partially_load_by_index(self, res, index, **kwargs):
        assert res.device.type == 'cpu'
        return res

    def fully_load(
        self, device: Union[torch.device, str], replicated=False
    ) -> torch.Tensor:
        """
        Collectively and fully load the dataset on all ranks,
        typically for compile backend=='none' case.

        Args:
        - device: the device to load data to. (self.device will not be used.)

            NOTE
            Unlike other load methods where data is always loaded to
            CPU for JIT-internal use, after full load the data is expected
            to be ready on user-specified device, immediately.
            So instead of going via CPU, fully_load() accepts the argument
            for the fianl device.

        - replicated: if True, load the same, full data to all ranks,
            otherwise only load to rank-0, other ranks will have the
            placeholder tensor and should not do calculation with it.

            NOTE
            With placeholder tensors, other ranks can still get proper
            shapes/dtypes.

        Returns:
        - torch.Tensor: the full tensor, on the specified device.
        """
        raise NotImplementedError()

    def get_placeholder(self, device=None) -> torch.Tensor:
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
        3.  when backend=='none', distributed data is only loaded on rank-0,
            create placeholders on other ranks to ease retrival shape/dtype
            especially to create receiving buffers in communication.
        """
        if device is None:
            device = self.device

        # torch.Tensor.expand can expand shape-(1,) to e.g. shape-(0,0,0),
        # but not to ndim=0 shape ().
        if len(self.shape) == 0:
            ph = torch.zeros((), dtype=self.dtype, device=device)
        else:
            # TODO using `.expand()` is compatible (i.e. works with
            # `Tensor.data=ph` and propagates shape/dtype) and simple.
            #
            # However, `torch.nn.Parameter.__new__` and
            # `torch.nn._ParameterMeta(torch._C._TensorMeta)`
            # may have indicated the protocol to make a very customized object
            # to be compatible.
            #
            # That would be good because we can get totally ride of OOM,
            # not only by `.to()`, and also prevent `esr.Tensor` from e.g.
            # `torch.ones_like()`.
            ph = torch.zeros(
                (1,), dtype=self.dtype, device=device
            ).expand(self.shape)  # can even expand to (0,0,0)

        ########################################
        #              WARNING
        ########################################
        # .to() method on placeholder tensors must be strictly disabled,
        # otherwise it might materialize the memory and cause OOM!
        def _to_forbidden(self, *args, **kwargs):
            raise EasierJitException("Cannot can .to() on placeholders!")
        ph.to = _to_forbidden.__get__(ph)

        setattr(ph, ATTRIBUTE_PLACEHOLDER, True)
        return ph

    def __repr__(self) -> str:
        """
        When possible, return a string which could be treated as valid Python
        code to construct this data loader, except for `.device`, e.g.
        ```
        ArangeTensorLoader(start=0, end=1, step=1, dtype=torch.float64)
        ```

        NOTE this repr str will be used to validate compilation cache for
        mesh consistency across EASIER sessions.
        """
        raise NotImplementedError()


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
    def minmax(self, region: _SimpleIndex) -> Tuple[Num, Num]:
        amin, amax = self.tensor[region].aminmax()
        return amin.item(), amax.item()

    @functools.cache
    def count_unique(self, region: _SimpleIndex) -> int:
        return self.tensor[region].unique().shape[0]

    def partially_load_by_chunk(
        self, chunk_size: int
    ) -> Iterator[torch.Tensor]:
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
            chunk = self.tensor[start:end].clone()
            yield chunk

    def partially_load_by_rank(self) -> Tuple[torch.Tensor, int, int]:
        dist_env = get_runtime_dist_env()
        world_size = dist_env.world_size
        rank = dist_env.rank
        orig_len = self.tensor.shape[0]

        # Put tailing elements in the part for the last rank, making the size
        # of that part bigger than chunk_size.
        start, end = _get_offset_exactly_nparts(orig_len, world_size, rank)

        return self.tensor[start:end].clone(), start, end

    def partially_load_by_index(
        self, index: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return self.tensor[index]

    def fully_load(
        self, device: Union[torch.device, str], replicated=False
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

    @functools.cache
    def minmax(self, region: _SimpleIndex) -> Tuple[Num, Num]:
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

            for chunk in self.partially_load_by_chunk(1024 * 1024 * 128):
                chunk_min, chunk_max = torch.aminmax(chunk)
                amin = _opt_cmp(amin, chunk_min, min)
                amax = _opt_cmp(amax, chunk_max, max)

            amin, amax = amin.item(), amax.item()  # type: ignore

            dist_env.broadcast_object_list(0, [amin, amax])

        else:
            [amin, amax] = dist_env.broadcast_object_list(0)

        return amin, amax

    @functools.cache
    def count_unique(self, ) -> int:
        if self.dtype.is_floating_point:
            raise NotImplementedError("Not supporting floats yet")

        amin, amax = self.minmax()
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

                for chunk in self.partially_load_by_chunk(1024 * 1024 * 128):
                    in_bitpack = torch.logical_and(
                        chunk >= bitpack_min, chunk < bitpack_max)
                    bitpack[chunk[in_bitpack] - bitpack_min] = 1

                bitpack_nnz = int(torch.count_nonzero(bitpack))
                nunique += bitpack_nnz

            dist_env.broadcast_object_list(0, [nunique])

        else:
            [nunique] = dist_env.broadcast_object_list(0)

        return nunique

    def partially_load_by_chunk(
        self, chunk_size: int
    ) -> Iterator[torch.Tensor]:
        orig_len = self.shape[0]

        # Put tailing elements in an individual chunk whose size is smaller.
        nchunk, remainder = divmod(orig_len, chunk_size)
        if remainder > 0:
            nchunk += 1

        with self._dataset_as_dtype() as d:
            for i in range(nchunk):
                start = chunk_size * i
                end = min(orig_len, chunk_size * (i + 1))

                chunk_np: np.ndarray = d[start:end]
                chunk: torch.Tensor = torch.from_numpy(chunk_np)
                yield chunk

    def partially_load_by_rank(self) -> Tuple[torch.Tensor, int, int]:
        dist_env = get_runtime_dist_env()
        rank = dist_env.rank

        orig_len = self.shape[0]
        sub_shape = self.shape[1:]

        # To avoid OOM, we cannot load the whole dataset on rank-0 then
        # simply call dist.scatter.
        # Instead, we load the part for each rank once, and do P2P.
        if rank == 0:
            with self._dataset_as_dtype() as d:
                for w in range(1, dist_env.world_size):
                    start, end = _get_offset_exactly_nparts(
                        orig_len, nparts=dist_env.world_size, part=w)

                    part_np: np.ndarray = d[start:end]
                    part: torch.Tensor = \
                        torch.from_numpy(part_np).to(dist_env.comm_device)
                    isend = dist_env.def_isend(part, dst=w, tag=w)
                    for req in dist_env.batch_isend_irecv([isend]):
                        req.wait()

                    # TODO each rank-0-rank-w comm may take a while,
                    # subsequennt recvs should not timeout.

                s0, e0 = _get_offset_exactly_nparts(
                    orig_len, nparts=dist_env.world_size, part=0)
                part0_np: np.ndarray = d[s0:e0]
                part0 = torch.from_numpy(part0_np)
                return part0, s0, e0

        else:
            start, end = _get_offset_exactly_nparts(
                orig_len, dist_env.world_size, rank)
            buffer = torch.empty(
                (end - start,) + sub_shape,
                dtype=self.dtype, device=dist_env.comm_device
            )
            irecv = dist_env.def_irecv(buffer, src=0, tag=rank)
            for req in dist_env.batch_isend_irecv([irecv]):
                req.wait()

            return buffer.cpu(), start, end

    def partially_load_by_index(
        self, index: torch.Tensor, *,
        chunk_size=1024 * 1024 * 128,  # roughly 128M elements
        **kwargs
    ) -> torch.Tensor:
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
        nchunk, remainder = divmod(orig_len, chunk_size)
        if remainder > 0:
            nchunk += 1

        local_parts = []
        rev_poses = []

        def _run(d):
            for i in range(nchunk):
                start = chunk_size * i
                end = min(orig_len, chunk_size * (i + 1))

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
        self, device: Union[torch.device, str], replicated=False
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

    def minmax(self) -> Tuple[Num, Num]:
        return self.value, self.value

    def count_unique(self) -> int:
        return 1

    def _full(
        self, batch_dim_len: Optional[int], device: Union[torch.device, str]
    ):
        if batch_dim_len is None:
            batch_dim_len = self.shape[0]

        shape = (batch_dim_len,) + self.shape[1:]
        return torch.full(
            shape, self.value, dtype=self.dtype, device=device)  # type: ignore

    def partially_load_by_chunk(self, chunk_size: int
                                ) -> Iterator[torch.Tensor]:
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
        dist_env = get_runtime_dist_env()
        rank = dist_env.rank
        orig_len = self.shape[0]
        start, end = _get_offset_exactly_nparts(
            orig_len, dist_env.world_size, rank)
        return self._full(end - start, 'cpu'), start, end

    def partially_load_by_index(self, index: torch.Tensor,
                                **kwargs) -> torch.Tensor:
        return self._full(index.shape[0], 'cpu')

    def fully_load(
        self, device: Union[torch.device, str], replicated=False
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

    def minmax(self, region: _SimpleIndex) -> Tuple[Num, Num]:
        r = range(self._start, self._end, self._step)
        if self._step > 0:
            return r[0], r[-1]
        else:
            return r[-1], r[0]

    def count_unique(self, region: _SimpleIndex) -> int:
        return self.shape[0]

    def partially_load_by_chunk(
        self, chunk_size: int
    ) -> Iterator[torch.Tensor]:
        orig_len = self.shape[0]

        # Put tailing elements in an individual chunk whose size is smaller.
        nchunk, remainder = divmod(orig_len, chunk_size)
        if remainder > 0:
            nchunk += 1

        for i in range(nchunk):
            range_start = self._start + chunk_size * i * self._step
            range_end = range_start + chunk_size * self._step
            if self._step > 0:
                range_end = min(self._end, range_end)
            else:
                range_end = max(self._end, range_end)

            chunk = torch.arange(range_start, range_end, self._step,
                                 dtype=self.dtype, device='cpu')
            yield chunk

    def partially_load_by_rank(self) -> Tuple[torch.Tensor, int, int]:
        dist_env = get_runtime_dist_env()
        rank = dist_env.rank
        orig_len = self.shape[0]
        offset_start, offset_end = _get_offset_exactly_nparts(
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

    def partially_load_by_index(self, index: torch.Tensor,
                                **kwargs) -> torch.Tensor:
        return (index * self._step + self._start).to(dtype=self.dtype)

    def fully_load(
        self, device: Union[torch.device, str], replicated=False
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


class StridedDataLoader(DataLoaderBase):
    def __init__(self, inner: DataLoaderBase, index: _SimpleIndex):
        super().__init__()

        self.inner = inner
        self.index = index

        strided = inner.get_placeholder()[index]
        self.shape = tuple(strided.shape)
    
    def collective_init(self) -> None:
        self.coll_check_dtype_shape_devicetype()
    
    def _rev_compose_index_range(self, region: _SimpleIndex) -> _SimpleIndex:
        raise NotImplementedError()
    def _rev_compose_index_tensor(self, index: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    def minmax(self, region: _SimpleIndex) -> Tuple[Num, Num]:
        region = self._rev_compose_index_range(region)
        return self.inner.minmax(region)
    
    def count_unique(self, region: _SimpleIndex) -> int:
        region = self._rev_compose_index_range(region)
        return self.inner.count_unique(region)
    
    def partially_load_by_range(self, region: _SimpleIndex) -> torch.Tensor:
        region = self._rev_compose_index_range(region)
        return self.inner.partially_load_by_range(region)
    
    def partially_load_by_index(self, index: torch.Tensor, **kwargs) -> torch.Tensor:
        index = self._rev_compose_index_tensor(index)
        return self.inner.partially_load_by_index(index, **kwargs)


class CartesianProductDataLoader(DataLoaderBase):
    def __init__(self, components: List[DataLoaderBase]):
        super().__init__()

        self.components = components
    
    def collective_init(self) -> None:
        pass


class FlattenDataLoader(DataLoaderBase):
    """
    Flatten n-dim DataLoader to 1-d, in an innermost-major manner.
    """
    def __init__(self, inner: DataLoaderBase):
        super().__init__()

        self.inner = inner


# Make Mesh a nn.Module so that EASIER can look through to get DataLoaders
# that are Mesh's attributes.
class Mesh(torch.nn.Module):
    """
    Define a regular N-d mesh whose vertices are the cartesian product of
    N input arrays, the i-th input array has the shape of `(L_i,) + DIMS_i`
    (If original `DIM_i` is empty then it's treated as `(1,)`).

    The vertices of the mesh are organized in a flattened manner, the result
    shape will be 2-d and be
    `( L_0*...*L_{N-1} , prod(DIMS_1)+...+prod(DIMS_{N-1}) )`.

    Remarks:
    -   In case of each input array carries different vertex properties,
        the subtensors for vertices are always flattened firstly
        i.e. `prod(DIMS_i)`.
    """
    def __init__(
        self,
        *dimensional_vertices: DataLoaderBase,
        device: Union[torch.device, str, None] = None
    ):
        if device is None:
            # TODO like torch.set_default_device()
            device = 'cpu'

        ndim = len(dimensional_vertices)
        if ndim == 0:
            raise ValueError("Must have at least one input data")

        # number of vertices per dim
        nvs: List[int] = []
        # number of hypercubes per dim
        ncs: List[int] = []
        for dt in dimensional_vertices:
            if isinstance(dt, (ArangeTensorLoader, FulledTensorLoader)):
                # TODO support general DataLoader like H5 and InMemTensor.
                raise NotImplementedError(
                    "only support easier.arange/linspace/full/ones/zeros"
                )

            if not len(dt.shape) >= 1:
                raise ValueError("Input data must be at least 1-d")
            dim_nv = dt.shape[0]
            if dim_nv == 0:
                raise ValueError("Input data must not be empty")

            nvs.append(dim_nv)
            ncs.append(dim_nv - 1)
        
        self.nv: int = math.prod(nvs)

        # each hypercube has 2*ND (ND-1)-d faces.
        nfaces = math.prod(ncs) * 2 * ndim
        nbfaces = 1  # 2 * \sum_i { \prod_{i!=j} ncs[j]  }
        self.ne: int = nfaces - nbfaces

        # TODO
        # We may assume hypercubes/cells are innermost flattened,
        # to apply src/dst Selectors, we must ensure from vertices+get_index
        # user could construct cell data in the exactly innermost order.

        # shape=(ne, ND)
        self.src = MeshInteriorFaceIdxDataLoader(ncs, True, device)
        self.dst = MeshInteriorFaceIdxDataLoader(ncs, False, device)

        # flattened cartesian product of all arg dataloaders.
        # shape=(nv, ND)
        self.vertices = CartesianProductDataLoader(list(dimensional_vertices))
    
    def get_index(self, *idx: Union[int, slice, EllipsisType]) -> DataLoaderBase:
        """
        Since EASIER requires vertices in a mesh to be organized as 1-d list,
        user can call `mesh.get_index(*IDX)` to get an index data for
        EASIER program to reconstruct indexing in traditional N-d array for the
        mesh/vertices, i.e.:
        ```
        mesh = esr.Mesh(d0, ..., d{N-1})
        idx = mesh.get_index(idx_0, ..., idx_{N-1})
        vdata = mesh.vertices[idx]
        # equals to
        ndarray = torch.cartesian_prod(
                d0, ..., d{N-1}
            ).reshape(L0, ..., L{N-1}, -1)
        vdata = ndarray[idx_0, ..., idx_{N-1}]
        ```

        For example, to calculate the distances between vertices along dim-2:
        ```
        def __init__(self):
            self.mesh = esr.Mesh(d0, d1, d2)

            end_vertices_idx = self.mesh.get_index(0, 0, 1:)  # for dim-2
            start_vertices_idx = self.mesh.get_index(0, 0, :-1)

            self.end_vertices_selector = Selector(end_vertices_idx)
            self.start_vertices_selector = Selector(start_vertices_idx)

        def forward(self):
            dim2_distances = \
                self.end_vertices_selector(self.mesh.vertices) \
                - self.start_vertices_selector(self.mesh.vertices)
        ```

        TODO Looks a bit rigid that users must define so many idx/selector
        fields. How about allowing directly indexing batch dim using
        DataLoaders? If detected we can insert Selector for it (and can share
        Selector instances).
        TODO arguably Reducer won't be symmetrically benefited from syntactic
        sugar like this, as Reducer is more configureable and there seems no
        torch ops for Reducer as concise as getitem for Selector.
        (torch.index_reduce_ seems to exactly match Reducer)

        """
        pass

        # TODO can be composed as
        # flatten( cartesian_product([ ... esr.arange(L_i) ... ])[*idx] )


class MeshInteriorFaceIdxDataLoader(DataLoaderBase):
    R"""
    Given a N-d hypercube volume at (i_1, i_2, ..., i_N) in the regular mesh
    whose i-d edge has L_i hypercubes (L_i + 1 vertices):
    -   it has 2N faces, each face is a (N-1)-d hypercube.
        e.g. 2d rect has 4 edges, 3d cube has 6 faces.
    -   the total number of interior faces is
        $ 2N * \prod_i {L_i} - 2 * \sum_i { \prod_{j!=i}{ L_j } } $
        or
        $ 2 * \sum_i { (L_i - 1) * \prod_{j!=i}{ L_j } }$
    """
    def __init__(
        self,
        dim_ncells: List[int],
        flow_src: bool,
        device: Union[torch.device, str] = 'cpu'
    ):
        super().__init__()
        self.dtype = torch.int64
        self.shape = 1
        self.device = torch.device(device)

        self.dim_ncells = dim_ncells

        # A reference direction for flow direction on the volume
        self.flow_src = flow_src
    
    def fully_load(self, device: Union[torch.device, str], replicated=False) -> torch.Tensor:
        return super().fully_load(device, replicated)
    


class CartesianProductConcatSymbolicDataLoaderBase(DataLoaderBase):
    """
    Each CP to concat is at most prod(L_i), but each may be subtracted with
    some panels (making it a CP of one axis/set being smaller).

    The result is flattened into 1-d, as concat (disjoint union) of many
    cartesian products is generally not a cartesian product anymore.

    for each original item, apply a symbolic expr `f` on it.
    """
    def __init__(self) -> None:
        super().__init__()

        concat_components: List[CartesianProductDataLoader] = []
        self.concat_components = concat_components
        # TODO assert ndim and kdim

        # self.shape = ('sum', 'kk')


    def minmax(self, region: _SimpleIndex) -> Tuple[Num, Num]:
        """
        Get minimum and maximum value within the `region` range of data source.

        TODO minmax() and count_unique() both accept *simple* index as range,
        i.e. tensor-typed index is not allowed.
        This indicates we may need to reimplement minmax()/count_unique() on a
        very detailed, tensor-indexed region if we provides such DataLoaders.
        """
        raise NotImplementedError()

    def count_unique(self, region: _SimpleIndex) -> int:
        """
        Count unique elements in the exact `region` range of data source.

        Used by Reducer.set_fullness()
        """
        raise NotImplementedError()
    


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


    def partially_load_by_range(self, region: _SimpleIndex) -> torch.Tensor:
        raise NotImplementedError()


    def partially_load_by_index(self, index: torch.Tensor, **kwargs
                                ) -> torch.Tensor:

        raise NotImplementedError()
    

    def fully_load(self, device: Union[torch.device, str], replicated=False) -> torch.Tensor:
        return super().fully_load(device, replicated)
    