# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Dict, List, Optional, Tuple, Type, Union, cast, \
    overload, TYPE_CHECKING
import torch.types
from typing_extensions import TypeAlias
import os
import dataclasses

# REMARK
# Because of our custom definition of `easier.sum` function,
# the builtin `sum()` is hidden.
# To use it e.g. in `EdgeTensor.dist_sync`,
# please call explicit `builtins.sum(...)` instead.
import builtins
from typing_extensions import Literal

import torch
from torch import nn
from torch import fx

import h5py

from easier.core.runtime.dist_env import get_cpu_dist_env
from easier.core.runtime.data_loader import \
    ArangeTensorLoader, DataLoaderBase, InMemoryTensorLoader, H5DataLoader, \
    FulledTensorLoader, ATTRIBUTE_PLACEHOLDER, torch_dtype_to_numpy_dtype
from easier.core.utils import logger, get_random_str


if TYPE_CHECKING:
    from easier.core.passes.tensor_group_partition import ElemPart
    from easier.core.passes.tensor_grouping import EasierTensorGroup


def hdf5(
    file: str, dataset: str,
    dtype: Optional[torch.dtype] = None,
    device: Union[torch.device, str] = 'cpu',
    **h5_file_kwargs
):
    """
    Create a handle to a HDF5 dataset.

    The specified dataset must be accessible from rank-0.
    """
    return H5DataLoader(file, dataset, dtype=dtype, device=device,
                        **h5_file_kwargs)


def full(size, fill_value, *,
         dtype: Optional[torch.dtype] = None, device=None):
    """
    Args:
    - dtype: Optional[torch.dtype]:
        If None, the default dtype is `torch.int64` for integer `fill_value`
        and `torch.float64` for floating-poin `fill_value`.
    - device: Optional[torch.Device]:
        If None, the default device is `"cpu"`.
    """
    if isinstance(fill_value, int):
        default_dtype = torch.int64
    elif isinstance(fill_value, float):
        default_dtype = torch.float64
    else:
        raise TypeError('fill_value must be integer or floating-point')

    if dtype is None:
        dtype = default_dtype

    if device is None:
        # TODO like torch.set_default_device()
        device = 'cpu'
    return FulledTensorLoader(fill_value, size, dtype, device)


def zeros(size, dtype=None, device=None):
    # TODO torch.zeros/ones can have `size` be both tuple and `*size:int`.
    """
    Args:
    - dtype: Optional[torch.dtype]:
        If None, the default dtype is `torch.int64`.
    """
    if dtype is None:
        dtype = torch.float64
    return full(size, 0, dtype=dtype, device=device)


def ones(size, dtype=None, device=None):
    """
    Args:
    - dtype: Optional[torch.dtype]:
        If None, the default dtype is `torch.int64`.
    """
    if dtype is None:
        dtype = torch.float64
    return full(size, 1, dtype=dtype, device=device)


@overload
def arange(end, dtype=None, device=None): ...
@overload
def arange(start, end, step=1, dtype=None, device=None): ...


def arange(*args, **kwargs):
    def _end_matcher(end, dtype=None, device=None):
        return (0, end, 1, dtype, device)

    def _start_end_matcher(start, end, step=1, dtype=None, device=None):
        return (start, end, step, dtype, device)

    def _resolve():
        try:
            return _end_matcher(*args, **kwargs)
        except TypeError:
            pass

        try:
            return _start_end_matcher(*args, **kwargs)
        except TypeError:
            pass

        raise TypeError(f'Unexpected arguments {args} to easier.arange')

    start, end, step, dtype, device = _resolve()

    for arg in [start, end, step]:
        if not isinstance(arg, (int, float)):
            raise TypeError(
                'argument to easier.arange must be integer or floating-point')

    promoted_value = start + end + step
    if isinstance(promoted_value, int):
        default_dtype = torch.int64
    elif isinstance(promoted_value, float):
        default_dtype = torch.float64
    else:
        raise TypeError(
            'argument to easier.arange must be integer or floating-point')

    if dtype is None:
        dtype = default_dtype

    if device is None:
        # TODO like torch.set_default_device()
        device = 'cpu'

    return ArangeTensorLoader(start, end, step, dtype, device)


def _resolve_data_loader(arg) -> DataLoaderBase:
    if isinstance(arg, DataLoaderBase):
        return arg
    elif isinstance(arg, Tensor) or hasattr(arg, ATTRIBUTE_PLACEHOLDER):
        # The .data is banned before it's an empty placeholder, even for cases
        # of InMemoryTensorLoader. We specifically insert a private
        # "easier_placeholder" attr on the placeholder tensors.
        raise TypeError(
            "Cannot use another easier.Tensor (or its .data property)"
            " to initialize easier.Tensor")
    elif isinstance(arg, torch.Tensor):
        return InMemoryTensorLoader(arg)
    else:
        raise TypeError(f"Unknown data type {type(arg)}")


def _validate_idx_dataloader(module: Union['Selector', 'Reducer']):
    dl = module.easier_data_loader
    cls_name = module.__class__.__name__

    cpu_dist_env = get_cpu_dist_env()
    if cpu_dist_env.rank == 0:
        try:
            try:
                iinfo = torch.iinfo(dl.dtype)
            except TypeError:
                raise TypeError(
                    f"The index tensor to {cls_name} must be an integer tensor"
                )

            idxmin: int = iinfo.max
            idxmax: int = iinfo.min
            for chunk in dl.partially_load_by_chunk(1024 * 1024 * 128):
                chunk_idxmin, chunk_idxmax = torch.aminmax(chunk)
                idxmin = builtins.min(int(chunk_idxmin), idxmin)
                idxmax = builtins.max(int(chunk_idxmax), idxmax)

            if not (0 <= idxmin):
                raise ValueError(
                    f"The minimum value of the {cls_name} index tensor {idxmin}"
                    f" must be greater than or equal 0"
                )
            if isinstance(module, Reducer):
                n = module.n
                if not isinstance(n, int):
                    raise TypeError(
                        f"The argument `n` to {cls_name} must be an integer"
                    )
                if not (idxmax < n):
                    raise ValueError(
                        f"The maximum value of the {cls_name} index tensor"
                        f" {idxmax} must be smaller than {n} the length of"
                        " the first dimension of the resultant tensor"
                    )
        except Exception as e:
            logger.exception(e)

            # Aborting one rank-0 will kill all processes and surpass barriers.
            cpu_dist_env.abort()

    else:  # rank != 0
        idxmax = -1  # does not matter

    # NOTE this bcast() also serves as a collective barrier.
    [idxmax] = cpu_dist_env.broadcast_object_list(0, [idxmax])

    if isinstance(module, Selector):
        module.idx_max = idxmax


def _dist_collect(tensor: 'Tensor') -> torch.Tensor:
    """
    Collect all distributed part of an `esr.Tensor` into its original device.

    Given that both the original device and the runtime device can be either
    CPU or GPU, and these two device choices can be different:
    Here we move tensor partitions to CPU and exchange them using CPU channels
    and then move the recovered full `torch.Tensor` to the original device.
    """
    assert tensor.elempart is not None

    cpu_dist_env = get_cpu_dist_env()
    elempart = tensor.elempart
    subshp = tuple(tensor.shape[1:])

    synced = torch.empty(
        (builtins.sum(elempart.lengths),) + subshp,  # type: ignore
        dtype=tensor.dtype, device='cpu')

    parts = cpu_dist_env.all_gather(
        tensor.data.to('cpu'),
        shapes=[(bs,) + subshp for bs in elempart.lengths])
    idxes = cpu_dist_env.all_gather(
        elempart.idx, shapes=[(bs,) for bs in elempart.lengths])

    for part, idx in zip(parts, idxes):
        synced[idx] = part

    return synced.to(device=tensor.easier_data_loader.device)


IdxStatus: TypeAlias = Literal['placeholder', 'partially_loaded', 'rewritten']


class Selector(nn.Module):
    def __init__(self, idx: Union[torch.Tensor, DataLoaderBase], **kwargs):
        super().__init__()

        # not going to be shown on the interface.
        if kwargs.get('easier_noncollective_idx', False) is True:
            # Both _resolve_data_loader() and _validate_idx() require to
            # be run collectively. In certain cases like EASIER-injected
            # reordering Selectors we need to skip those collective steps.
            assert isinstance(idx, torch.Tensor)
            assert not (
                isinstance(idx, Tensor) or hasattr(idx, ATTRIBUTE_PLACEHOLDER)
            )
            self.idx = idx
            self.easier_index_status = 'rewritten'

            # self.idx_max is not needed for noncollective Selectors

            # Let related passes to fill other JIT-only fields.

            return

        idx_dl = _resolve_data_loader(idx)
        self.easier_data_loader = idx_dl
        _validate_idx_dataloader(self)
        idx_ph: torch.Tensor = idx_dl.get_placeholder()

        self.idx: torch.Tensor = idx_ph

        # The maximum index of the original idx data loader,
        # collectively figured out in _validate_idx_dataloader
        self.idx_max: int

        self.easier_index_status: IdxStatus = 'placeholder'

        # ======
        # Fields filled during JIT compilation
        self.easier_tensor_group: 'EasierTensorGroup'

        # Which part of the data is initially loaded for dist_pass rewriting,
        # only needed in dist_pass.
        self.easier_idx_part_range: Optional[Tuple[int, int]] = None

        # `self.idx` is required to be registerd as a "buffer" so it's moveable
        # in non-JIT mode.
        # In contrast, halo indexes are only moved and moved once to
        # backend device during JIT.
        self.runtime_halos_local_idxes: List[torch.Tensor]
        self.runtime_halos_recv_lengths: List[int]

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.easier_index_status != 'rewritten':
            raise RuntimeError("Selector.idx not ready, run compile() first")

        # NOTE at backend==none runtime,
        # bad cases like `not(self.idx_max < tensor.shape[0])` will be
        # thrown and reported by the tensor indexing operation.
        return tensor[self.idx]

    def to(self, *args, **kwargs):
        raise TypeError(
            "easier.Selector does not support .to() method,"
            " consider defining a new easier.Selector instance.")


ReduceOp: TypeAlias = Literal['sum', 'prod', 'mean', 'amax', 'amin']


class Reducer(nn.Module):

    def __init__(self, idx: Union[torch.Tensor, DataLoaderBase],
                 n: int,
                 reduce: ReduceOp = 'sum'
                 ) -> None:
        super().__init__()

        self.n = n
        self.reduce = reduce

        idx_dl = _resolve_data_loader(idx)
        self.easier_data_loader = idx_dl
        _validate_idx_dataloader(self)
        idx_ph: torch.Tensor = idx_dl.get_placeholder()

        self.idx: torch.Tensor = idx_ph

        self.easier_index_status: IdxStatus = 'placeholder'

        self.set_fullness()

        # ======
        # Fields filled during JIT compilation
        self.easier_tensor_group: 'EasierTensorGroup'

        # Which part of the data is initially loaded for dist_pass rewriting,
        # only needed in dist_pass.
        self.easier_idx_part_range: Optional[Tuple[int, int]] = None

        self.easier_reordering_selector_idx: Optional[torch.Tensor] = None

        # `self.idx` is required to be registerd as a "buffer" so it's moveable
        # in non-JIT mode.
        # In contrast, halo indexes are only moved and moved once to
        # backend device during JIT.
        self.runtime_halos_local_idxes: List[torch.Tensor]
        self.runtime_halos_recv_lengths: List[int]

    def set_fullness(self):
        """
        Calculate and collectively set the score for fullness of
        the original definition of the Reducer.
        """
        dist_env = get_cpu_dist_env()

        if dist_env.rank == 0:
            nwrites = 0

            # Each time we count output elements with global IDs that fall in
            # the pack, in case the pack gets too big;
            # For each such pack, traverse all .idx data and "set the bit" and
            # count "bits".

            bitpack_maxlen = 1024 * 1024 * 128  # 128MB with bools
            bitpack_n, remainder = divmod(self.n, bitpack_maxlen)
            if remainder > 0:
                bitpack_n += 1

            # TODO use real bitmap and popcount instead of *bool*pack.
            for bitpack_i in range(bitpack_n):
                bitpack_min = bitpack_i * bitpack_maxlen
                bitpack_max = builtins.min(
                    (bitpack_i + 1) * bitpack_maxlen, self.n)

                bitpack = torch.zeros(
                    [bitpack_max - bitpack_min], dtype=torch.bool)

                for chunk in self.easier_data_loader.partially_load_by_chunk(
                    # load_by_chunk requires to run on rank-0
                    1024 * 1024 * 128
                ):
                    in_bitpack = torch.logical_and(
                        chunk >= bitpack_min, chunk < bitpack_max)
                    bitpack[chunk[in_bitpack] - bitpack_min] = 1

                bitpack_nnz = int(torch.count_nonzero(bitpack))
                nwrites += bitpack_nnz

            [nwrites] = dist_env.broadcast_object_list(0, [nwrites])
        else:
            [nwrites] = dist_env.broadcast_object_list(0)

        self.easier_fullness: float = float(nwrites) / float(self.n)

    def set_is_full(self):
        """
        Calculate and locally set the flag for if the local Reducer is full.

        Remark:
        The is_full flag can only be calculated after distribution rewriting
        or after fully index loading.
        """
        assert self.easier_index_status == 'rewritten', \
            "Reducer.idx must have been rewritten or loaded"

        unique_idx = self.idx.unique(sorted=True)

        self.easier_is_full: bool = \
            unique_idx.shape[0] == self.n and \
            unique_idx[0] == 0 and unique_idx[-1] == (self.n - 1)

    def forward(self, tensor: torch.Tensor,
                *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.easier_index_status != 'rewritten':
            raise RuntimeError("Reducer.idx not ready, run compile() first")

        shape = tensor.shape

        if out is None:
            out = torch.zeros(
                (self.n,) + shape[1:],
                dtype=tensor.dtype, device=tensor.device)

        if out.shape[0] != self.n:
            raise ValueError(
                f"{out.shape[0]} the length of the first dimension of `out`"
                f" and {self.n} the specified `Reducer.n` do not match")

        idx = self.idx[
            (...,) + (None,) * (len(shape) - 1)].expand(-1, *shape[1:])
        return out.scatter_reduce_(0, idx, tensor, self.reduce,
                                   include_self=False)

    def to(self, *args, **kwargs):
        raise TypeError(
            "easier.Reducer does not support .to() method,"
            " consider defining a new easier.Reducer instance.")


class Tensor(nn.Parameter):

    def __new__(cls,
                data: Union[torch.Tensor, DataLoaderBase],
                mode: Literal['partition', 'replicate'],
                requires_grad: bool = False) -> "Tensor":
        dl = _resolve_data_loader(data)
        if mode == 'partition':
            if not (len(dl.shape) >= 1):
                raise ValueError('Partition easier.Tensor must have ndim >= 1')

        data_ph: torch.Tensor = dl.get_placeholder()

        tensor = super().__new__(cls, data_ph, requires_grad)  # type: ignore

        # store the parsing results to
        # avoid parsing args/kwargs in __init__ again.
        tensor.easier_data_loader = dl
        return tensor

    def __init__(self,
                 data: Union[torch.Tensor, DataLoaderBase],
                 mode: Literal['partition', 'replicate'],
                 requires_grad: bool = False) -> None:
        # ======
        # Fields filled during __new__

        self.easier_data_loader: DataLoaderBase

        self.easier_data_ready: bool = False

        # TODO names like `is_partition` are potentially conflict to pytorch,
        # will `easier_is_partition` with namespace be better?
        self.is_partition = mode == 'partition'
        self.is_replica = mode == 'replicate'

        # ======
        # Fields filled during JIT compilation

        # Only distributed tensors have tensor groups
        # (no matter if they are referenced by `get_attr` Nodes),
        self.easier_tensor_group: 'EasierTensorGroup'

        # Only tensors that are distributed and used has this field set.
        self.elempart: 'Optional[ElemPart]' = None

    def __repr__(self) -> str:
        if self.is_replica:
            if not self.easier_data_ready:
                repr_str = ' ' + repr(self.easier_data_loader)
            else:
                repr_str = '\n' + repr(self.data)
            return 'Replicated easier.Tensor' + repr_str
        else:
            if not self.easier_data_ready:
                repr_str = ' ' + repr(self.easier_data_loader)
            else:
                repr_str = ''
            return 'Managed partitioned easier.Tensor' + repr_str

    def __setitem__(self, indices, val) -> 'Tensor':
        """
        This __setitem__ generally runs for setting replicated easier.Tensor
        content, outside the scope for compiled easier.Module.forward().
        """
        if not self.easier_data_ready:
            raise RuntimeError("Tensor data is not ready, run compile() first")
        self.data.__setitem__(indices, val)
        return self

    def to(self, *args, **kwargs):
        """
        `torch.Tensor.to()` returns self if the specified dtype/device is
        compatible with the old properties.
        Following this style implies that any in-place modification might be
        unconsciously reflected through on both `easier.Tensor` references.

        EASIER users might consider defining a new `easier.Tensor` instance
        with proper dtype settings.
        """
        raise TypeError(
            "easier.Tensor does not support .to() method,"
            " consider defining a new easier.Tensor instance.")

    def collect(self) -> torch.Tensor:
        if not self.easier_data_ready:
            raise RuntimeError("Tensor data is not ready, run compile() first")

        if self.elempart is not None:
            return _dist_collect(self)
        else:
            return self.data.to(self.easier_data_loader.device, copy=True)

    def save(self, h5_file_path, h5_dataset_path, **h5_file_kwargs) -> None:
        if not self.easier_data_ready:
            raise RuntimeError("Tensor data is not ready, run compile() first")

        cpu_dist_env = get_cpu_dist_env()
        if cpu_dist_env.rank == 0:

            h5_file_path = os.path.expanduser(h5_file_path)

            os.makedirs(os.path.dirname(h5_file_path), exist_ok=True)

            np_dtype = torch_dtype_to_numpy_dtype(self.dtype)
            orig_shape = self.easier_data_loader.shape

            # mode 'a', append: read/write if exists, create otherwise
            with h5py.File(h5_file_path, mode='a', **h5_file_kwargs) as h5f:
                # TODO this dataset is kept only, will it need to be
                # close and reopen (created only once) to avoid OOM?
                h5d = h5f.create_dataset(
                    h5_dataset_path, dtype=np_dtype, shape=orig_shape)

                if self.elempart is not None:
                    _dist_save(self, h5d)  # collectively save dist tensor
                else:
                    h5d[...] = self.data.to('cpu')  # replica

        else:
            if self.elempart is not None:
                _dist_save(self, None)  # collectively save dist tensor


def _dist_save(tensor: 'Tensor', h5d: Optional[h5py.Dataset]) -> None:
    """
    Save all distributed part of an `esr.Tensor` into a HDF5 dataset.

    The paths to the HDF5 dataset must be accessible from rank-0 process.

    On rank-0, each time it handles the range [chunk_size*i, chunk_size*(i+1))
    (chunk_size determines the maximum memory consumption on rank-0)
    and every element of this range is gathered from all ranks.
    Each time, all ranks only send the element in that range to rank-0.
    """
    assert tensor.elempart is not None

    cpu_dist_env = get_cpu_dist_env()
    chunk_size = 1024 * 1024 * 128  # roughly 128M elements

    orig_len = tensor.easier_data_loader.shape[0]
    sub_shape = tensor.easier_data_loader.shape[1:]

    nchunk, remainder = divmod(orig_len, chunk_size)
    if remainder > 0:
        nchunk += 1

    idx = tensor.elempart.idx  # already ordered, but discrete.

    # Count how many indexes each chunk has on this rank
    chunk_counts = torch.bincount(
        idx // chunk_size,  # determine chunk IDs of each elempart index
        minlength=nchunk)

    # TODO to bincount all chunked slices is a simple and direct approach.
    # An alternative approach is to `torch.searchsorted` since idx is ordered,
    # which is binary search underneath and may be more efficient (but save()
    # is called outside of computing workflow although)

    idx_start = 0

    for i in range(nchunk):
        slice_len = int(chunk_counts[i])
        idx_end = idx_start + slice_len

        idx_slice = idx[idx_start:idx_end].to('cpu')
        data_slice = tensor[idx_start:idx_end].to('cpu')

        idx_slices = cpu_dist_env.gather(0, idx_slice)
        data_slices = cpu_dist_env.gather(0, data_slice)

        idx_start = idx_end  # for next step

        if cpu_dist_env.rank == 0:
            assert isinstance(idx_slices, list)
            assert isinstance(data_slices, list)
            assert len(idx_slices) == len(data_slices)

            chuck_size_i = builtins.min(
                orig_len, chunk_size * (i+1)) - chunk_size * i
            assert builtins.sum(s.shape[0] for s in idx_slices) == chuck_size_i

            h5d_slice = torch.empty(
                (chuck_size_i,) + sub_shape, dtype=tensor.dtype)
            for idx_slice, data_slice in zip(idx_slices, data_slices):
                assert idx_slice.shape[0] == data_slice.shape[0]
                h5d_slice[idx_slice] = data_slice

            assert h5d is not None
            h5d[(chunk_size*i):(chunk_size*i+chuck_size_i)] = h5d_slice


class Module(nn.Module):

    def __init__(self):
        super().__init__()

        # Already pruned for pickle
        self.easier_raw_graph: Optional[fx.graph.Graph] = None

        # ======
        # Fields filled during JIT compilation

        # Only filled on top modules i.e. inputs to esr.compile()
        self.easier_jit_backend: Literal[
            'torch', 'cpu', 'gpu', 'none', None
        ] = None

        # Only has value when jit_backend in ['torch','cpu','gpu']
        self.partition_mode: Literal['metis', 'evenly']

        # Each Module shares the ElemPart dict of all Modules in JIT session.
        self.easier_elemparts: 'Dict[EasierTensorGroup, ElemPart]'

    def forward(self):
        pass

    def to(self, *args, **kwargs):
        raise TypeError(
            "easier.Module does not support .to() method,"
            " consider defining a new easier.Module instance.")


def _allreduce(op, tensor: torch.Tensor, *args, **kwargs):
    return op(tensor, *args, dim=0, keepdim=True, **kwargs)


def sum(tensor: torch.Tensor) -> torch.Tensor:
    """
    Equivalent to `torch.sum(vertex_tensor, dim=0, keepdim=True)`.

    Args:
    -   tensor:
        At JIT-time, must be a distributed tensor.
    """
    return _allreduce(torch.sum, tensor)


def prod(tensor: torch.Tensor) -> torch.Tensor:
    """
    Equivalent to `torch.prod(vertex_tensor, dim=0, keepdim=True)`.

    Args:
    -   tensor:
        At JIT-time, must be a distributed tensor.
    """
    return _allreduce(torch.prod, tensor)


def norm(tensor: torch.Tensor, p: Union[int, str] = 2) -> torch.Tensor:
    """
    Equivalent to `torch.norm(tensor, p, dim=0, keepdim=True)`.

    Args:
    -   tensor:
        At JIT-time, must be a distributed tensor.
    """
    return _allreduce(torch.norm, tensor, p)


def max(tensor: torch.Tensor) -> torch.Tensor:
    """
    Equivalent to `torch.max(tensor, dim=0, keepdim=True)`.

    Args:
    -   tensor:
        At JIT-time, must be a distributed tensor.
    """
    return _allreduce(torch.max, tensor)


def min(tensor: torch.Tensor) -> torch.Tensor:
    """
    Equivalent to `torch.min(tensor, dim=0, keepdim=True)`.

    Args:
    -   tensor:
        At JIT-time, must be a distributed tensor.
    """
    return _allreduce(torch.min, tensor)


easier_aggregators = (sum, prod, norm, max, min)
