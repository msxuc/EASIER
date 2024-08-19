# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import itertools
from types import FunctionType
from typing import Dict, Iterable, List, Optional, Sequence, Set, \
    Tuple, Union, cast
from typing_extensions import Literal, TypeVar, assert_never
import torch
from torch import LongTensor
from torch.fx.graph import Graph
from torch.fx.node import Node
from torch.nn.modules import Module
from easier.core.passes.tensor_grouping import \
    EasierTensorDef, EasierTensorGroup, get_node_tensor_group
from easier.core.passes.utils import EasierInterpreter, \
    normalize_reducer_call_into_args, normalize_selector_call_into_args, \
    vector_index_of
from easier.core.utils import logger
from easier.core.runtime.dist_env import get_cpu_dist_env, get_runtime_dist_env
from easier.core.runtime.modules import HaloExchanger, all_gather_into_tensor

import easier.core.module as esr
from easier.core.module import Module, Selector, Reducer
from easier.core.passes.dataflow_distribution.tensor_partition import \
    ElemPart, partition_tensor_groups, insert_noncomm_elemparts
from easier.core.passes.metadata_propagation.metadata import \
    EasierTensorMeta, Role, convert_scalar_type_to_torch_dtype, \
    set_node_meta, get_node_meta


def calculate_idx_and_halos(
    input_gidx: torch.Tensor,
    input_elempart: ElemPart,
    output_gidx: torch.Tensor,
    output_elempart: ElemPart,
):
    """
    Calculate the halo regions and data exchange for a Selector/Reducer call.

    We keep the elements of the output elempart on this worker and do not
    exchange them. Only exchange elements of the input element to form halo
    regions.

    This subprocedures abstract the Selector/Reducer instances into the
    global ID vector of their input/output.
    For example:

    For Selector, `output_idx` will be `[n*rank,n*(rank+1))`
    where n is the size of the evenly partitioned `Selector.idx`
    and halos are reordered into the order of `output_elempart`;

    For Reducer, it's the `input_idx` to be such a range,
    and halos are reordered into the order of `input_elempart`.
    """
    dist_env = get_cpu_dist_env()

    input_gidx_slices_to_scatter = []
    output_gidx_slices_to_scatter = []

    for u in range(dist_env.world_size):
        if dist_env.rank == u:
            output_elempart_u = dist_env.broadcast(
                u, output_elempart.idx)
        else:
            output_elempart_u = dist_env.broadcast(
                u, shape=(output_elempart.lengths[u],),
                dtype=output_elempart.idx.dtype)

        # Find the positions of `output_idx` that are writing to
        # `output_elempart_u` and the positions are discrete and ordered.
        # NOTE `idx_pos` is 0-based and relative to local `output_idx`.
        pos = torch.isin(output_gidx, output_elempart_u).argwhere().ravel()

        input_gidx_slice = input_gidx[pos]
        output_gidx_slice = output_gidx[pos]

        input_gidx_slices_to_scatter.append(input_gidx_slice)
        output_gidx_slices_to_scatter.append(output_gidx_slice)

    gathered_input_gidx_slices: List[torch.Tensor] = \
        dist_env.all_to_all(input_gidx_slices_to_scatter)
    gathered_output_gidx_slices: List[torch.Tensor] = \
        dist_env.all_to_all(output_gidx_slices_to_scatter)

    gathered_input_gidx = torch.concat(gathered_input_gidx_slices)
    gathered_output_gidx = torch.concat(gathered_output_gidx_slices)
    assert gathered_input_gidx.shape[0] == gathered_output_gidx.shape[0]

    # For runtime indexing
    halo_lidxes_to_this = []
    halo_gidxes_to_this = []

    for w in range(dist_env.world_size):
        if dist_env.rank == w:
            input_elempart_w = dist_env.broadcast(
                w, input_elempart.idx)

        else:
            input_elempart_w = dist_env.broadcast(
                w, shape=(input_elempart.lengths[w],),
                dtype=input_elempart.idx.dtype)

        # NOTE for over-simplified understanding of calculation,
        # for Selector, we can skip `halo_pos` for local input_elempart,
        # since the whole local input_elempart will be concat-ed.

        # calculation of halo
        halo_lidx = torch.isin(input_elempart_w, gathered_input_gidx
                               ).argwhere().ravel()
        halo_lidxes_to_this.append(halo_lidx)

        halo_gidx = input_elempart_w[halo_lidx]
        halo_gidxes_to_this.append(halo_gidx)

    # For indexing on local tensor and sending to others
    runtime_halos_lidxes: List[torch.Tensor] = \
        dist_env.all_to_all(halo_lidxes_to_this)

    return gathered_input_gidx, gathered_output_gidx, \
        halo_lidxes_to_this, halo_gidxes_to_this, \
        runtime_halos_lidxes


def rewrite_selector_instance(
    selector: Selector,
    input_elempart: ElemPart, output_elempart: ElemPart
):
    assert selector.idx.device.type == 'cpu', \
        "selector.idx tensor should be moved to cpu before distribution"
    assert selector.easier_idx_part_range is not None, \
        "selector.idx must have been partially loaded by rank"

    dist_env = get_cpu_dist_env()
    rank = dist_env.rank

    idx_part = selector.idx
    idx_part_start, idx_part_end = selector.easier_idx_part_range

    gathered_input_gidx, gathered_output_gidx, \
        halo_lidxes_to_this, halo_gidxes_to_this, \
        runtime_halos_lidxes = \
        calculate_idx_and_halos(
            idx_part, input_elempart,
            torch.arange(idx_part_start, idx_part_end), output_elempart)

    # NOTE this `sort` is specifically to reorder gathered idx-cells
    # according to local output_elempart (because it's ordered too)
    reordered_output_gidx, reorder_pos = torch.sort(gathered_output_gidx)
    # A halo-exchanged Selector 1:1 writes the local output_elempart
    assert torch.equal(reordered_output_gidx, output_elempart.idx)

    input_gidx = gathered_input_gidx[reorder_pos]

    # We simply expect the IR rewriting sub-pass to concat halos and local part
    # ** by the ranks ** of their workers.
    if halo_gidxes_to_this[rank].shape[0] != 0:
        # ... but if no local input_elempart element is ever read, don't bother
        # concat-ing it.
        halo_gidxes_to_this[rank] = input_elempart.idx

    # The space is concated by many idx pieces,
    # each idx piece has ordered elements but pieces are interleaving.
    # And some elements inherited from local input elempart may be not read.
    chunk_gidx_space = torch.concat(halo_gidxes_to_this)
    rewritten_idx = vector_index_of(input_gidx, chunk_gidx_space)

    selector.idx = rewritten_idx
    selector.easier_index_ready = True

    # For indexing on local tensor and sending to others
    selector.runtime_halos_local_idxes = runtime_halos_lidxes

    # unlike `input_idxes_runtime_halos`, this list still preserves
    # the flag whether `local_idxes_runtime_halos[rank] == 0`.
    halo_lengths_to_recv = [
        lidx_recv.shape[0] for lidx_recv in halo_lidxes_to_this]
    selector.runtime_halos_recv_lengths = halo_lengths_to_recv


def rewrite_reducer_instance(
    reducer: Reducer,
    input_elempart: ElemPart, output_elempart: ElemPart
):
    assert reducer.idx.device.type == 'cpu', \
        "reducer.idx tensor should be moved to cpu before distribution"
    assert reducer.easier_idx_part_range is not None, \
        "reducer.idx must have been partially loaded by rank"

    idx_part = reducer.idx
    idx_part_start, idx_part_end = reducer.easier_idx_part_range

    gathered_input_gidx, gathered_output_gidx, \
        halo_lidxes_to_this, halo_gidxes_to_this, \
        runtime_halos_lidxes = \
        calculate_idx_and_halos(
            torch.arange(idx_part_start, idx_part_end), input_elempart,
            idx_part, output_elempart)

    # Firstly reorder the "index rows"
    # (namely `zip(gathered_input_gidx, gathered_output_gidx)`)
    # by mathcing `gathered_input_gidx` and `halo_gidxes_to_this`
    # to match the layout of the chunk at runtime.
    chunk_gidx = torch.concat(halo_gidxes_to_this)
    pos = vector_index_of(chunk_gidx, gathered_input_gidx)
    reordered_output_gidx = gathered_output_gidx[pos]

    # Unlike Selector, A halo-exchanged Reducer 1:1 reads its concat-ed halo
    # (including the ** sliced ** local input_elempart) but randomly writes
    # the local output_elempart.
    # Additionally, some elements of output_elempart might be never written to.
    rewritten_idx = vector_index_of(reordered_output_gidx, output_elempart.idx)

    halo_lengths_to_recv = [
        lidx_recv.shape[0] for lidx_recv in halo_lidxes_to_this]

    reducer.idx = rewritten_idx
    reducer.easier_index_ready = True
    reducer.n = output_elempart.idx.shape[0]
    reducer.runtime_halos_local_idxes = runtime_halos_lidxes
    reducer.runtime_halos_recv_lengths = halo_lengths_to_recv


class HaloExchangerNameAllocator:
    def __init__(self) -> None:
        self.id = 0

    def alloc_name(self, root: torch.nn.Module) -> str:
        while True:
            name = f'haloexchanger{self.id}'
            self.id += 1
            if not hasattr(root, name):
                return name


class IdxAndConstMover(EasierInterpreter):
    """
    Move `Selector/Reducer.idx` (registered by `Module.register_buffer()`),
    and constant tensors to the device of the JIT backend.

    -   the `.idx` has been rewritten and always on CPU;
    -   distributed/replicated tensors are moved elsewhere.
    """

    def __init__(self, modules: Sequence[Module], graphs: Sequence[Graph],
                 runtime_device) -> None:
        super().__init__(modules, graphs)

        # Device for JIT, may be different from device for data loader
        # (which only matters under "torch" JIT backend)
        self.runtime_device = runtime_device

    def if_get_attr(self, submod_path: str, attr_name: str, p):
        if isinstance(p, esr.Tensor):
            pass  # no-op

        elif isinstance(p, torch.Tensor):  # constants
            # FX will treat tensors created ad hoc in `forward()` as
            # constant tensors and `setattr()` them into the root module
            # with attribute names like `_tensor_constant0`,
            # and such attributes are neither Module parameters or buffers.
            # We need to move those constant tensors to proper device, too.
            path: str = self.current_node.target  # type: ignore
            assert '.' not in path, \
                "constant tensors must be attrs of the root module"

            setattr(self.current_module, path,
                    # Ensure constants are contiguous so that these tensors
                    # have initial strides.
                    # And it's ok to call `.to(device)` multi times.
                    p.to(self.runtime_device).contiguous())

    def if_call_module(self, submod: Module):
        if isinstance(submod, (esr.Selector, esr.Reducer)):
            submod.idx = submod.idx.to(self.runtime_device)

            # HaloExchanger stores this List, so changing item of list can be
            # seen there, to match runtime devices for input and index.
            for i, t in enumerate(submod.runtime_halos_local_idxes):
                submod.runtime_halos_local_idxes[i] = t.to(
                    device=self.runtime_device)


class SubmodRewriter(EasierInterpreter):
    """
    Rewrite index tensors for Selector/Reducer instances.

    We need to follow the order of how submodules appear on IR,
    which is the same on all workers.
    """

    def __init__(self, modules, graphs,
                 elemparts: Dict[EasierTensorGroup, ElemPart],
                 device: torch.device):
        super().__init__(modules, graphs)

        self.elemparts = elemparts
        self.device = device

        self.haloxchg_name_allocator = HaloExchangerNameAllocator()
        self.instances_written: Set[Union[Selector, Reducer]] = set()

    def if_call_module(self, submod: Module):
        root = self.current_module
        node = self.current_node
        output_elempart = self.elemparts[
            get_node_tensor_group(node)]  # type: ignore

        if isinstance(submod, esr.Selector):
            input_node = normalize_selector_call_into_args(
                *node.args, **node.kwargs)
            input_elempart = self.elemparts[
                get_node_tensor_group(input_node)]  # type: ignore

        elif isinstance(submod, esr.Reducer):
            input_node, _out_node = normalize_reducer_call_into_args(
                *node.args, **node.kwargs)
            input_elempart = self.elemparts[
                get_node_tensor_group(input_node)]  # type: ignore

        else:
            assert False, "unreachable"  # have raised in preceding passes

        # Rewrite EASIER module instances
        if submod not in self.instances_written:
            self.instances_written.add(submod)

            if isinstance(submod, esr.Selector):
                rewrite_selector_instance(submod,
                                          input_elempart, output_elempart)
            elif isinstance(submod, esr.Reducer):
                rewrite_reducer_instance(submod,
                                         input_elempart, output_elempart)
            else:
                assert False, "unreachable"

        # Rewrite call_module Node
        meta: EasierTensorMeta = get_node_meta(node)  # type: ignore
        haloxchg_inst = HaloExchanger(
            is_for_selector=isinstance(submod, esr.Selector),
            input_elempart_length=input_elempart.idx.shape[0],
            runtime_halos_lidxes=submod.runtime_halos_local_idxes,
            runtime_recv_lengths=submod.runtime_halos_recv_lengths,
            element_tensor_shape=meta.shape[1:],
            dtype=convert_scalar_type_to_torch_dtype(meta.dtype)
        )
        haloxchg_modpath = self.haloxchg_name_allocator.alloc_name(root)
        root.add_module(haloxchg_modpath, haloxchg_inst)

        assert isinstance(input_node, Node)
        with node.graph.inserting_before(node):
            haloxchg_node = node.graph.call_module(
                haloxchg_modpath, (input_node,))
            node.replace_input_with(input_node, haloxchg_node)


class AllReducePrimitivesRewriter(EasierInterpreter):
    """
    To achieve the transformation:
    ```
    %a = esr.reduce(%x)
    ~~
    %b = continuation1(%a)
                       ~~
    %c = continuation2(%a)
                       ~~
    ```
    into
    ```
    %worker_local = esr.reduce(%x, *worker_local_args)
    ~~~~~~~~~~~~~
    %allgather = all_gather(%worker_local)
    %replica = esr.reduce(%allgather, *replica_args)

    %b = continuation1(%replica)
                       ~~~~~~~~
    %c = continuation2(%replica)
                       ~~~~~~~~
    ```
    All places of the original reference to `%a` will be rewritten.

    All reduction operations on vertex tensors are EASIER primitives,
    and they are equivalent to `torch.reduce(x, keepdim=True, dim=0)`.
    So the `node.args` and `node.kwargs` remain unchanged after
    the insertion of the second `reduce`, except the input tensor itself.
    """

    def if_call_function(self, func):
        if func not in [
            esr.sum, esr.prod, esr.norm, esr.max, esr.min
        ]:
            return

        node = self.current_node

        # try to normalize the possible keyword arguments:
        # NOTE the param name must match the definition, which is "tensor".
        input_tensor = node.kwargs.get('tensor', None)
        if input_tensor is None:
            # param `tensor` is a positional arg
            input_tensor = node.args[0]
            replica_allreduce_args = node.args[1:]
            replica_allreduce_kwargs = node.kwargs.copy()
        else:
            # param `vertex_tensor` is a keyword arg
            replica_allreduce_args = ()
            # If the 1st param `vertex_tensor` is passed as a keyword arg,
            # generally there will be no positional args.
            assert len(node.args) == 0
            replica_allreduce_kwargs = node.kwargs.copy()
            replica_allreduce_kwargs.pop('tensor')

        # WARNING
        # when using `inserting_before` and `inserting_after`,
        # please note it fixes the insert-point for all new insertion
        # under its scope, e.g.
        # `with inserting_after(n): insert [a,b,c]`
        # produces:
        # `..., n, *, c, b, a, ...`
        # where `*` indicates the insertion point, always relative to `n`.
        #
        # That's why we need to specify `inserting_before(node.next)`.
        with node.graph.inserting_before(node.next):
            dist_reduce_prim: FunctionType = node.target  # type: ignore
            # EASIER reduce primitives are equivalent to
            # `torch.reduce(..., keepdim=True, dim=0)`
            worker_local_reduce = node.graph.call_function(
                dist_reduce_prim, tuple(node.args), dict(node.kwargs))

            allgather = node.graph.call_function(
                all_gather_into_tensor, (worker_local_reduce,))

            replica_reduce_op: FunctionType = getattr(
                torch, dist_reduce_prim.__name__)
            replica_allreduce_kwargs['keepdim'] = True
            replica_allreduce_kwargs['dim'] = 0
            replica_reduce = node.graph.call_function(
                replica_reduce_op,
                (allgather,) + replica_allreduce_args,
                replica_allreduce_kwargs)

        node.replace_all_uses_with(replica_reduce)
        # Newly added Nodes have no metadata associated,
        # and since we replace uses with the newly added Node, we need to copy
        # node metadata for it.
        set_node_meta(replica_reduce, get_node_meta(node))
        node.graph.erase_node(node)


def load_selector_reducer_idx(modules: List[esr.Module], graphs: List[Graph]):
    """
    Partially load Selector/Reducer.idx to each rank.
    The loaded idx tensor is always on CPU.
    """
    class ModuleIdxLoader(EasierInterpreter):
        def __init__(self,
                     modules: Sequence[esr.Module],
                     graphs: Sequence[Graph]):
            super().__init__(modules, graphs)

        def if_call_module(self, module: Module):
            if isinstance(module, (esr.Selector, esr.Reducer)):
                if not module.easier_index_ready:
                    partial_idx, pstart, pend = \
                        module.easier_data_loader.partially_load_by_rank()
                    module.idx = partial_idx
                    module.easier_idx_part_range = (pstart, pend)
                    module.easier_index_ready = True

    ModuleIdxLoader(modules, graphs).run()


def load_partitioned_tensors_from_source(
    elemparts: Dict[EasierTensorGroup, ElemPart],
    runtime_device: torch.device
):
    """
    Partially load distributed tensor by elempart.idx
    to the specified backend device on each rank.
    """
    for tensor_group, elempart in elemparts.items():
        for tensor in tensor_group.tensor_defs:
            if isinstance(tensor, esr.Tensor):
                assert tensor.is_partition  # only dist tensors form group
                tensor.elempart = elempart

                tensor.data = \
                    tensor.easier_data_loader.partially_load_by_index(
                        elempart.idx).to(runtime_device)
                tensor.easier_data_ready = True


def load_replicated_tensors_from_source(
    modules: List[esr.Module], graphs: List[Graph],
    runtime_device: torch.device
):
    """
    Fully load replicated tensor to the specified backend device on each rank.
    """
    # This particularly cannot be done via `get_attr` handler, as a replica
    # may be accessed outside the JIT scope.
    for root in modules:
        for p in root.parameters(recurse=True):
            if isinstance(p, esr.Tensor) \
                    and p.is_replica and not p.easier_data_ready:
                p.data = p.easier_data_loader.fully_load(runtime_device)
                p.easier_data_ready = True


def reconstruct_elempart_from_dumps(
    loaded_top_modules: List[esr.Module]
) -> Dict[EasierTensorGroup, ElemPart]:

    elemparts = {}

    # "Orphan" esr.Tensors not involving in communication also form
    # a TensorGroup and got dumped with evenly partitioned ElemParts,
    # so ElemParts we read from dumps cover those "orphan" tensors too.
    for root in loaded_top_modules:
        assert root.easier_loaded_cache is not None
        for tensor_attrpath, (elempart_idx, elempart_lengths) \
                in root.easier_loaded_cache.elempart_dumps.items():

            t = cast(esr.Tensor, root.get_parameter(tensor_attrpath))

            # This is the TensorGroup obj created in this Python session
            g = t.easier_tensor_group
            assert g is not None

            # elempart is still duplicated across Module, we take any one
            elemparts[g] = ElemPart(elempart_idx, elempart_lengths)

    insert_noncomm_elemparts(loaded_top_modules, elemparts)

    return elemparts


def load_tensors_from_dumps(loaded_top_modules: List[esr.Module],
                            elemparts: Dict[EasierTensorGroup, ElemPart],
                            runtime_device: torch.device):

    for root in loaded_top_modules:
        assert root.easier_loaded_cache is not None
        for tensor_attrpath, tensor_dump \
                in root.easier_loaded_cache.tensor_dumps.items():

            t = cast(esr.Tensor, root.get_parameter(tensor_attrpath))
            if t.is_partition:
                t.elempart = elemparts[t.easier_tensor_group]  # type: ignore

            # may overwrite across Modules
            t.data = tensor_dump.to(runtime_device)

            t.easier_data_ready = True


def load_selectors_reducers_from_dumps(
    loaded_top_modules: List[esr.Module],
) -> Set[Union[Selector, Reducer]]:
    res = set()
    for root in loaded_top_modules:
        assert root.easier_loaded_cache is not None
        for mod_attrpath, (
                mod_idx, halo_local_idxes, halo_recv_lengths, reducer_n) \
                in root.easier_loaded_cache.selector_reducer_dumps.items():

            m = cast(Union[Selector, Reducer],
                     root.get_submodule(mod_attrpath))

            m.idx = mod_idx  # still CPU
            m.runtime_halos_local_idxes = halo_local_idxes
            m.runtime_halos_recv_lengths = halo_recv_lengths

            if isinstance(m, esr.Reducer):
                assert type(reducer_n) is int
                m.n = reducer_n

            m.easier_index_ready = True

            res.add(m)

    return res


def distribute_dataflow(modules: List[esr.Module], graphs: List[Graph]):
    runtime_device = get_runtime_dist_env().comm_device

    # We do filtering here:
    # In jit.py/compiler() we have checked it's either all are loaded or
    # all are not loaded;
    # and the `modules` here are not all top-level esr.Modules.
    # So if any of them is marked loaded, it means all (sub-)modules here
    # are loaded.
    loaded_top_modules = list(filter(
        (lambda m: m.easier_loaded_cache is not None),
        modules))

    if len(loaded_top_modules) == 0:
        load_selector_reducer_idx(modules, graphs)

        # Key TensorGroup contains all distributed and used esr.Tensors
        # (i.e. esr.Tensors that are accessed via `get_attr` Nodes)
        elemparts: Dict[EasierTensorGroup, ElemPart] = \
            partition_tensor_groups(modules, graphs)

        # Rewrite Selector/Reducer instances and call_module Nodes
        SubmodRewriter(modules, graphs, elemparts, runtime_device).run()

        # Rewrite EASIER allreduce Nodes
        AllReducePrimitivesRewriter(modules, graphs).run()

        load_partitioned_tensors_from_source(elemparts, runtime_device)

        load_replicated_tensors_from_source(modules, graphs, runtime_device)

        IdxAndConstMover(modules, graphs, runtime_device).run()

    else:
        elemparts: Dict[EasierTensorGroup, ElemPart] = \
            reconstruct_elempart_from_dumps(loaded_top_modules)

        # Load Selector/Reducer idx from dumps and rewrite call_modules Nodes
        sr = SubmodRewriter(modules, graphs, elemparts, runtime_device)
        sr.instances_written = \
            load_selectors_reducers_from_dumps(loaded_top_modules)
        sr.run()

        # Rewrite EASIER allreduce Nodes
        AllReducePrimitivesRewriter(modules, graphs).run()

        load_tensors_from_dumps(loaded_top_modules, elemparts, runtime_device)

        IdxAndConstMover(modules, graphs, runtime_device).run()

    return modules, graphs
