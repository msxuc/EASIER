# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Dict, List, Optional, Tuple, Type, Union, cast, \
    overload, TYPE_CHECKING
import torch.types
from typing_extensions import TypeAlias
import os
import dataclasses
import json
import enum

# REMARK
# Because of our custom definition of `easier.sum` function,
# the builtin `sum()` is hidden.
# To use it e.g. in `EdgeTensor.dist_sync`,
# please call explicit `builtins.sum(...)` instead.
import builtins
from typing_extensions import Literal

import torch
from torch import nn

import h5py

from easier.core.runtime.dist_env import get_cpu_dist_env
from easier.core.runtime.data_loader import \
    DataLoaderBase, InMemoryTensorLoader, H5DataLoader, FulledTensorLoader, \
    ATTRIBUTE_PLACEHOLDER, torch_dtype_to_numpy_dtype
from easier.core.utils import logger


if TYPE_CHECKING:
    from easier.core.passes.dataflow_distribution.tensor_partition import \
        ElemPart
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
        raise TypeError()


def _validate_idx(dl: DataLoaderBase, cls_name: str, n: Optional[int]):
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
            if n is not None:
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

        cpu_dist_env.barrier()

    else:
        cpu_dist_env.barrier()


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


class Selector(nn.Module):
    def __init__(self, idx: Union[torch.Tensor, DataLoaderBase]) -> None:
        super().__init__()

        idx_dl = _resolve_data_loader(idx)
        _validate_idx(idx_dl, 'Selector', None)
        self.easier_data_loader = idx_dl
        idx_ph: torch.Tensor = idx_dl.get_placeholder()

        self.idx: torch.Tensor
        self.register_buffer('idx', idx_ph)

        self.easier_index_ready = False

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
        if not self.easier_index_ready:
            raise RuntimeError()

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

        idx_dl = _resolve_data_loader(idx)
        _validate_idx(idx_dl, 'Reducer', n)
        self.easier_data_loader = idx_dl
        idx_ph: torch.Tensor = idx_dl.get_placeholder()

        self.idx: torch.Tensor
        self.register_buffer('idx', idx_ph)

        self.easier_index_ready = False

        self.n = n
        self.reduce = reduce

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

    def forward(self, tensor: torch.Tensor,
                *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.easier_index_ready:
            raise RuntimeError()

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
                dist: Literal['partition', 'replicate'],
                requires_grad: bool = False) -> "Tensor":
        dl = _resolve_data_loader(data)
        data_ph: torch.Tensor = dl.get_placeholder()

        tensor = super().__new__(cls, data_ph, requires_grad)  # type: ignore

        # store the parsing results to
        # avoid parsing args/kwargs in __init__ again.
        tensor.easier_data_loader = dl
        return tensor

    def __init__(self,
                 data: Union[torch.Tensor, DataLoaderBase],
                 dist: Literal['partition', 'replicate'],
                 requires_grad: bool = False) -> None:
        # ======
        # Fields filled during __new__

        self.easier_data_loader: DataLoaderBase

        self.easier_data_ready: bool = False

        # TODO names like `is_partition` are potentially conflict to pytorch,
        # will `easier_is_partition` with namespace be better?
        self.is_partition = dist == 'partition'
        self.is_replica = dist == 'replicate'

        # ======
        # Fields filled during JIT compilation

        # Only distributed tensors have tensor groups
        # (and they must be used i.e. be referenced by `get_attr` Nodes),
        # so this field can have default value None, unlike Selector or Reducer.
        self.easier_tensor_group: 'Optional[EasierTensorGroup]' = None

        # Only tensors that are distributed and used has this field set.
        self.elempart: 'Optional[ElemPart]' = None

    def __repr__(self) -> str:
        if self.is_replica:
            return 'Replicated easier.Tensor\n' + repr(self.data)
        else:
            return 'Managed partitioned easier.Tensor'

    def __setitem__(self, indices, val) -> 'Tensor':
        """
        This __setitem__ generally runs for setting replicated easier.Tensor
        content, outside the scope for compiled easier.Module.forward().
        """
        if not self.easier_data_ready:
            raise RuntimeError()
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
        if self.elempart is not None:
            return _dist_collect(self)
        else:
            return self.data.to(self.easier_data_loader.device, copy=True)

    def save(self, h5_file_path, h5_dataset_path, **h5_file_kwargs) -> None:
        cpu_dist_env = get_cpu_dist_env()
        if cpu_dist_env.rank == 0:
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


@dataclasses.dataclass
class JitConfig:
    """
    JIT configuration for this rank and this local machine.
    """
    world_size: int
    rank: int
    local_rank: int


def _get_current_jit_config():
    cpu_dist_env = get_cpu_dist_env()
    config = JitConfig(
        world_size=cpu_dist_env.world_size,
        rank=cpu_dist_env.rank,
        local_rank=cpu_dist_env.local_rank,
    )
    return config


class _StopLoadingJITCache(Exception):
    pass


class JitDump:
    """
    Across multiple loaded esr.Modules, redundant Tensor/ElemPart instances
    may be deserialized.
    When entering dist_pass (e.g. when relating them to EasierTensorGroup),
    deduplicate those instances, e.g., arbitrarily take an instance.
    """

    def __init__(self) -> None:
        self.tensor_dumps: Dict[str, torch.Tensor] = {}
        self.elempart_dumps: Dict[str, Tuple[torch.Tensor, List[int]]] = {}
        self.selector_reducer_dumps: Dict[
            str,
            #     idx,          halo_lidxes,    halo_lengths,  reducer_n
            Tuple[torch.Tensor, List[torch.Tensor], List[int], Optional[int]]
        ] = {}


class Module(nn.Module):

    def __init__(self):
        super().__init__()

        self.easier_loaded_cache: Optional[JitDump] = None

        # ======
        # Fields filled during JIT compilation
        self.easier_jit_backend: Literal['torch', 'cpu', 'gpu', 'none']

    def forward(self):
        pass

    def to(self, *args, **kwargs):
        raise TypeError(
            "easier.Module does not support .to() method,"
            " consider defining a new easier.Module instance.")

    def dump(self, dump_dir: str) -> None:
        """
        Dump `easier.Tensor` data to rank-0, and internal JIT configurations to
        local directory.
        The dumped data is not for users to inspect.

        Users may dump only a subset of initially compiled `easier.Module`s.

        Args:
        -   dump_dir:
                Machine-local dir to store dumped data for this module.
                Many ranks on the same machine can share the same dir,
                as long as they are dumped in the same session and for the same
                timepoint.

        Usage:
        ```
        m1, m2, mro = Model1(), Model2(), ModelRunOnce()
        [m1, m2, mro] = easier.compile([m1, m2, mro])

        mro()
        m1()
        m1.submod()
        m2()
        m2.submod()

        # don't call mro.dump()
        m1.dump('/mnt/checkpoint1/')
        m2.dump('/mnt/checkpoint2/')
        ```

        Remarks:
        `easier.Tensor` data to dump is essentially:
        -   Partitioned `easier.Tensor` contents
            These contents will be collected to rank-0 and stored
            in the original order.

            On the other hand, even without `easier.Module.load()`,
            these dumped data can be used to initialize `easier.Tensor`s
            in the same EASIER program, just like users are running new EASIER
            sessions.

        -   Replicated `easier.Tensor` contents
            Only get dumped on rank-0

        and `easier.Tensor` data will be dumped to `${dump_dir}/tensor.hdf5`
        on rank-0, each `easier.Tensor` will become a HDF5 dataset whose name
        is the path of the `easier.Tensor` within this `easier.Module`;
        JIT configurations will be dumped to `${dump_dir}/jit_{rank}.hdf5`.
        """
        # Besides esr.Tensor data, we store dist_pass states:
        # - rewritten idx
        # - elempart.idx
        # on each local machine and for each rank
        # into `${dump_dir}/jit_${rank}.hdf5`
        #
        # When loading from this dump_dir,
        # we will try to store and reuse the rewritten index.
        #
        # If it's improper to reuse (e.g. world_size changes),
        # we invoke dist_pass again, taking `tensor.hdf5` on rank-0 and
        # using the original index DataLoader specified in EASIER programs.
        os.makedirs(dump_dir, exist_ok=True)

        cpu_dist_env = get_cpu_dist_env()
        rank = cpu_dist_env.rank
        log_info0: Callable[[str], None] = \
            logger.info if rank == 0 else logger.debug

        cls_name = self.__class__.__name__

        # Within this esr.Module, deduplicate multi-reference esr.Tensor etc.
        memo = {}

        def _dump_tensor_once(jit_f: Optional[h5py.File],
                              tensor_attrpath: str, tensor: Tensor):
            """
            On rank-0, save full data in the original order.
            On each rank, save the partition/replica locally

            MEMO RECORD:
            [('TENSOR', address of esr.Tensor)] => (rep. Module, tensor path)
            # kv-pair value is only for debug purpose.

            Write logs that each esr.Tensor instance:
            -   is dumped as what attribute within this esr.Module;
            -   is skipped because it's referenced by multiple attributes
                and has already been dumped;
            -   besides the (representative) attribute name,
                an esr.Tensor also has a unique memory address as a Python obj,
                the address can auxiliarly help users identify the esr.Tensor
                instance.
            """
            t_addr = id(tensor)
            memo_key = ('TENSOR', t_addr)
            if memo_key not in memo:
                # The relation/topology how an instance is shared among
                # multiple modules is the same for all ranks.
                memo[memo_key] = (self, tensor_attrpath)

                # Full data, original order, on rank-0 only
                tensor_fpath = os.path.join(dump_dir, 'tensor.hdf5')
                tensor.save(tensor_fpath, tensor_attrpath)

                # When dumping with JIT cache, parition/replica are dumped
                # even the dumped contents always appear in tensor.hdf5.
                # The dataset names "tensor/ATTRPATH" also serve as keys
                # where the data will be written to.
                # I.e. tensor.hdf5 play no role in JIT-cache-loading.
                if jit_f is not None:
                    jit_f.create_dataset(
                        f'tensor/{tensor_attrpath}',
                        data=tensor.data.to('cpu')
                    )

                # Print tensor dumping events as INFO on rank-0 only.
                log_info0(
                    'Dump Tensor'
                    f' {cls_name}.{tensor_attrpath} (at 0x{t_addr:016x})'
                )
            else:
                (recorded_mod, recorded_path) = memo[memo_key]
                recorded_cls_name = recorded_mod.__class__.__name__
                log_info0(
                    'Skip dumping Tensor'
                    f' {cls_name}.{tensor_attrpath} (at 0x{t_addr:016x})'
                    f'\n\tdumped as {recorded_cls_name}.{recorded_path}'
                )

        def _dump_elempart_once(jit_f: h5py.File, elempart: 'ElemPart',
                                bound_tensor_attrpath: str):
            """
            ElemParts (both the mesh partition results and the relation to
            esr.Tensors) need to be saved, to support post-loading
            `esr.Tensor.save()` calls.

            Many tensors may be bound with the same elempart;
            those tensors may be scattered among Modules, or have different
            attr paths, too.

            But no matter what attr paths the tensors bound to elempart have,
            the set of ElemPart instances a Module has is fixed.

            MEMO RECORD:
            [('ELEMPART', address of ElemPart)] => (rep. Module, tensor path)
            """
            ep_addr = id(elempart)
            memo_key = ('ELEMPART', ep_addr)
            if memo_key not in memo:
                memo[memo_key] = (self, bound_tensor_attrpath)

                # We simply identify the elempart with name derived from the
                # (arbitrarily) first-seen esr.Tensor, to which the elempart
                # is bound.
                # Because we can locate that esr.Tensor (even, in different
                # executed/dumped/loaded sessions) firstly by Tensor attrpath,
                # then by the equivalence relation of Tensor instances.
                jit_f.create_dataset(
                    f'elempart/{bound_tensor_attrpath}/idx',
                    data=elempart.idx)
                jit_f.create_dataset(
                    f'elempart/{bound_tensor_attrpath}/lengths',
                    data=elempart.lengths)

                logger.debug(
                    'Dump ElemPart bound to '
                    f' {cls_name}.{bound_tensor_attrpath}')

        def _dump_selector_reducer_once(jit_f: h5py.File,
                                        mod: Union[Selector, Reducer],
                                        mod_attrpath: str):
            mod_addr = id(mod)
            memo_key = ('DIST_PASS', mod_addr)
            if memo_key not in memo:
                memo[memo_key] = (self, mod_attrpath)

                jit_f.create_dataset(
                    f'dist_pass/{mod_attrpath}/idx',
                    data=mod.idx.to('cpu')
                )

                for w, local_idx in enumerate(mod.runtime_halos_local_idxes):
                    # local_idx tensor will be empty, but not None
                    jit_f.create_dataset(
                        f'dist_pass/{mod_attrpath}/halo_local_idxes/{w}',
                        data=local_idx.to('cpu')
                    )

                jit_f.create_dataset(
                    f'dist_pass/{mod_attrpath}/halo_recv_lengths',
                    data=mod.runtime_halos_recv_lengths
                )

                if isinstance(mod, Reducer):
                    jit_f.get(
                        f'dist_pass/{mod_attrpath}'
                    ).attrs['reducer_n'] = str(mod.n)

                logger.debug(f'Dump S/R {cls_name}.{mod_attrpath}')

        def _dump_jit_config(jit_f: h5py.File):
            config = _get_current_jit_config()
            config_str = json.dumps(dataclasses.asdict(config))
            jit_f.attrs['jit_config'] = config_str

        if self.easier_jit_backend == 'none':
            # Don't even create jit_RANK.hdf5 for backend=='none'.
            for attr_path, param in self.named_parameters(recurse=True):
                if isinstance(param, Tensor):
                    if not param.easier_data_ready:
                        raise RuntimeError(
                            f'{cls_name}.{attr_path}'
                            ' easier.Tensor not compiled yet')

                    _dump_tensor_once(None, attr_path, param)
            return

        jit_fpath = os.path.join(dump_dir, f'jit_{rank}.hdf5')
        with h5py.File(jit_fpath, 'w') as jit_f:

            _dump_jit_config(jit_f)

            for attr_path, param in self.named_parameters(recurse=True):
                if isinstance(param, Tensor):
                    if not param.easier_data_ready:
                        raise RuntimeError(
                            f'{cls_name}.{attr_path}'
                            ' easier.Tensor not compiled yet')

                    _dump_tensor_once(jit_f, attr_path, param)

                    # jit-none/fully-loaded and replica don't have this.
                    if param.elempart is not None:
                        _dump_elempart_once(jit_f, param.elempart, attr_path)

            for mod_path, mod in self.named_modules():
                if isinstance(mod, (Selector, Reducer)):
                    if not mod.easier_index_ready:
                        raise RuntimeError(
                            f'{cls_name}.{mod_path} {mod.__class__.__name__}'
                            ' not compiled yet')

                    _dump_selector_reducer_once(jit_f, mod, mod_path)

    def load(self, dump_dir: str):
        """
        Load dumped easier.Tensor data from rank-0, and JIT configurations from
        machine local directory, before calling `easier.compile()`.

        Usage:
        ```
        m1 = Model1()
        m2 = Model2()

        m1.load('/mnt/checkpoint1/')
        m2.load('/mnt/checkpoint2/')
        [jitted1, jitted2] = easier.compile([m1, m2])
        ```

        The `esr.Module`s to load should still be constructed and initialized
        first. This means any data source they read should be ready
        (e.g. HDF5 data files should be accessible on rank-0).

        During `easier.compile()`, if any of the loaded JIT configurations
        do not match the current JIT settings
        (e.g. `jit_{rank}.hdf5` not existing; world size changes;
        ranks permuted), these configurations will be ignored.
        `easier.compile()` will generate new JIT configurations.
        """
        if self.easier_loaded_cache is not None:
            raise RuntimeError()

        # We didn't dedup data across esr.Modules getting dumped,
        # so shared esr.Tensors etc. will have their data being overwritten.
        # But that's ok, the loaded data are the same.

        cpu_dist_env = get_cpu_dist_env()
        rank = cpu_dist_env.rank
        cls_name = self.__class__.__name__

        def _stop_if(rank_should_stop: bool, debug_msg: str) -> None:
            """
            Collectively decide if to stop reusing JIT cache.
            """
            if rank_should_stop:
                logger.debug(debug_msg)

            all_stop = cpu_dist_env.all_gather_into_tensor(
                torch.tensor([rank_should_stop], dtype=torch.int64)
            ).sum().item() > 0

            if all_stop:
                if cpu_dist_env.is_host:
                    logger.info(
                        f'Dumped JIT configurations for {cls_name} will not'
                        f' be loaded (reason: {debug_msg})')

                raise _StopLoadingJITCache()

        tensor_fpath = os.path.join(dump_dir, 'tensor.hdf5')
        if rank == 0:
            if not os.path.isfile(tensor_fpath):
                logger.error('tensor.hdf5 does not exist.')
                cpu_dist_env.abort()  # will skip barrier()
        cpu_dist_env.barrier()

        try:
            cache = JitDump()

            # Try loading JIT cache.
            #
            # It's ok that JIT cache is unavailable,
            # then just use tensor.hdf5 only and re-compile.
            # But all ranks should agree on the same result of whether to reuse
            # JIT cache or none.
            jit_fpath = os.path.join(dump_dir, f'jit_{rank}.hdf5')
            jit_cache_not_exist = not os.path.isfile(jit_fpath)
            _stop_if(
                jit_cache_not_exist, 'jit_RANK.hdf5 does not exist.')

            with h5py.File(jit_fpath, 'r') as jit_f:
                config_str = str(jit_f.attrs['jit_config'])
                loaded_config = JitConfig(**json.loads(config_str))
                cur_config = _get_current_jit_config()
                config_not_match = loaded_config != cur_config
                _stop_if(
                    config_not_match,
                    f'Dumped JIT configurations for {cls_name} do not match.')

                #
                # All checks passed
                #

                tensor_grp = cast(Union[h5py.Group, dict],
                                  jit_f.get('tensor', {}))
                for tensor_attrpath, tensor_dataset in tensor_grp.items():
                    cache.tensor_dumps[tensor_attrpath] = torch.from_numpy(
                        tensor_dataset[...])

                elempart_grp = cast(
                    Union[h5py.Group, dict], jit_f.get('elempart', {}))
                for bound_tensor_attrpath, subgrp in elempart_grp.items():
                    elempart_idx = torch.from_numpy(subgrp['idx'][...])
                    # [xxx][...] returns a np.array
                    elempart_lengths = subgrp['lengths'][...].tolist()
                    cache.elempart_dumps[bound_tensor_attrpath] = (
                        elempart_idx, elempart_lengths)

                distpass_grp = cast(
                    Union[h5py.Group, dict], jit_f.get('dist_pass', {}))
                for mod_attrpath, subgrp in distpass_grp.items():
                    mod_idx = torch.from_numpy(subgrp['idx'][...])
                    halo_local_idxes = []
                    for w in range(loaded_config.world_size):
                        halo_local_idx = torch.from_numpy(
                            subgrp['halo_local_idxes'][str(w)][...])
                        halo_local_idxes.append(halo_local_idx)
                    halo_recv_lengths = \
                        subgrp['halo_recv_lengths'][...].tolist()

                    if 'reducer_n' in subgrp.attrs:
                        reducer_n = int(subgrp.attrs['reducer_n'])
                    else:
                        reducer_n = None

                    cache.selector_reducer_dumps[mod_attrpath] = (
                        mod_idx, halo_local_idxes, halo_recv_lengths,
                        reducer_n)

            self.easier_loaded_cache = cache

        except _StopLoadingJITCache:
            self.easier_loaded_cache = None  # TODO can load again if failed?

        # No matter jit_RANK.hdf5 is loaded or matched, always rebind
        # esr.Tensors' data_loader.
        # NOTE tensor.hdf5 is only on rank-0
        if rank == 0:
            with h5py.File(tensor_fpath, 'r') as tensor_f:
                attrpaths = list(tensor_f.keys())
                cpu_dist_env.broadcast_object_list(0, attrpaths)
        else:
            attrpaths = cpu_dist_env.broadcast_object_list(0)
        for tensor_attrpath in attrpaths:
            t = cast(Tensor, self.get_parameter(tensor_attrpath))
            # tensor_attrpath e.g. 'a.b.c' will be the dataset name
            old_dl = t.easier_data_loader
            t.easier_data_loader = H5DataLoader(
                tensor_fpath, tensor_attrpath,
                dtype=old_dl.dtype, device=old_dl.device)


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
