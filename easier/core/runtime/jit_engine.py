# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import dataclasses
import math
import operator
from typing import \
    Callable, Dict, List, Optional, Sequence, Tuple, TypeAlias, Union, cast
import numpy
from typing_extensions import Literal
import more_itertools
import pickle

import torch
from torch import nn
from torch.fx._symbolic_trace import Tracer
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
from torch.fx.proxy import Proxy
from torch.fx.graph import Graph

import easier as esr
from easier.core import passes
from easier.core import module as _EsrMod
from easier.core.passes.metadata_propagation import \
    StaticNodeMeta, get_static_node_metadata, Role, \
    get_runtime_tensor_metadata, set_runtime_tensor_metadata, RuntimeTensorMeta
from easier.core.passes.utils import \
    FX, get_easier_objects, tree_map, normalize_reducer_call_into_args
from easier.core.utils import EasierJitException, logger
from easier.core.runtime.dist_env import get_runtime_dist_env
from easier.core.runtime.modules import HaloExchanger


def _get_runtime_metadata_from_value(
    node_role: Role,
    val: Union[torch.Tensor, bool, int, float, None]
) -> RuntimeTensorMeta:
    if isinstance(val, torch.Tensor):
        return RuntimeTensorMeta(node_role, tuple(val.shape), val.dtype)

    if node_role == Role.DISTRIBUTED:
        raise EasierJitException(
            f"Unexpected distributed value {val} of type {type(val)}"
        )

    if val is None:
        # Replica Nodes: Output, nested esr.Module calls
        # may return None
        return RuntimeTensorMeta(Role.REPLICATED, (), torch.int32)
    elif isinstance(val, bool):
        return RuntimeTensorMeta(Role.REPLICATED, (), torch.bool)
    elif isinstance(val, int):
        # PyTorch Python wrapper isn't aware of Python int precision,
        # so we treat ints as current minimum int32 dtype
        # so they are compatible with any torch tensor with int-kind dtype.
        return RuntimeTensorMeta(Role.REPLICATED, (), torch.int32)
    elif isinstance(val, float):
        # Same as int32, treat Python float as current minimum float32.
        return RuntimeTensorMeta(Role.REPLICATED, (), torch.float32)
    else:
        # NOTE for types that cannot explicitly appear on `Node.args`,
        # (`torch.Tensor` is one of such types), their metadata is always
        # propagated and carried by their corresponding `Node[op='get_attr']`.
        # We don't expect to see them here.
        raise EasierJitException(
            f'Value {val} of type {type(val)} cannot have associated metadata'
        )

class _Skipped:
    """
    We use a special runtime object `_skipped = _Skipped()` to represent
    values of skipped Nodes.
    """
    pass

_skipped = _Skipped()


_RuntimeValue: TypeAlias = Union[
    torch.Tensor,
    Sequence['_RuntimeValue']

    # NOTE it's possible that FX trace `Tensor.item()` call which results in
    # a pure int/float scalar rather than a [0]-shape tensor.
]
RuntimeValue: TypeAlias = Union[
    _RuntimeValue,
    None,  # output Nodes, nested esr.Module calls
    _Skipped  # Skipped won't be nested
]

def get_aggregator_neutral_value(aggregator, dtype: torch.dtype):
    if dtype.is_complex:
        raise NotImplementedError()

    if dtype.is_floating_point:
        finfo = torch.finfo(dtype)
        vmax = finfo.max
        vmin = finfo.min
    else:
        iinfo = torch.iinfo(dtype)
        vmax = iinfo.max
        vmin = iinfo.min

    vneutral = {
        esr.sum: 0,
        esr.prod: 1,
        esr.norm: 0,
        esr.max: vmin,
        esr.min: vmax
    }[aggregator]
    return vneutral



def exchange_meta_for_halo_exchanger(
    halo_xchg: HaloExchanger,
    input: Union[torch.Tensor, _Skipped]
) -> Tuple[Tuple[int, ...], torch.dtype]:
    """
    Exchange shape/dtype info for recv buffers of HaloExchangers.

    On some workers the batch size of the input ElemPart is zero and the input
    Node is skipped, then we cannot get a valid input Tensor to the halo_xchg.
    However, the halo_xchg may need to receive, then it need valid shape/dtype
    info the allocate the recv buffers.
    For such cases, we'll exchange the shape/dtype info from other ranks.

    NOTE
    -   HaloExchanger itself is not a fully collective call, it may be called
        on some ranks and not on others.
        Therefore the call to this function may not be a fully collective call.

    -   Keep using P2P with the same src/dst ranks as the halo_xchg.
        Because on the ranks without HaloExchangers, they may have entered
        and been waiting in dist.all_gather_into_tensor (the high-level API)
        or ncclAllGathr (the low-level API) etc.
        Any communication APIs than P2P may incorrectly be mixed with them.
    """
    dist_env = get_runtime_dist_env()

    # Flags reflecting halo_xchg's original P2P connectivity.
    #
    # Remarkably, if a rank has zero-batch-size input, it cannot send or
    # be received-from (but if can recv from others).
    can_send_to = torch.zeros((dist_env.world_size,), dtype=torch.bool)
    can_recv_from = torch.zeros((dist_env.world_size,), dtype=torch.bool)
    for u in range(dist_env.world_size):
        lidx = halo_xchg.runtime_halos_lidxes[u]
        if u != dist_env.rank:
            if lidx.shape[0] > 0:
                can_send_to[u] = True
    for u in range(dist_env.world_size):
        recv_len = halo_xchg.runtime_recv_lengths[u]
        if u != dist_env.rank:
            if recv_len > 0:
                can_recv_from[u] = True

    def _exchange(
        to_send: torch.Tensor,
    ):
        """
        Always exclude the self rank.
        The result will have the shape `(world_size,) + tosend.shape`.
        """
        to_send = to_send.to(dist_env.comm_device)
        recv_buffer = torch.empty(
            (dist_env.world_size,) + to_send.shape,
            dtype=to_send.dtype, device=dist_env.comm_device
        )
        p2p_ops = []

        for u in range(dist_env.world_size):
            if can_send_to[u]:
                isend = dist_env.def_isend(to_send, u, tag=u)
                p2p_ops.append(isend)

        for u in range(dist_env.world_size):
            if can_recv_from[u]:
                irecv = dist_env.def_irecv(
                    recv_buffer[u], u, tag=dist_env.rank
                )
                p2p_ops.append(irecv)
        
        for req in dist_env.batch_isend_irecv(p2p_ops):
            req.wait()
        return recv_buffer.cpu()
    
    #
    # Exchange dtype and ndim
    # both have relatively constant sizes
    #

    # A big enough buffer to store the serialized dtype.
    # buffer[0] is the length of bytes.
    dtype_buffer = torch.zeros((1000,), dtype=torch.int64)
    ndim = 0

    if isinstance(input, torch.Tensor):

        dtype_bytes = pickle.dumps(input.dtype)
        assert len(dtype_bytes) < dtype_buffer.shape[0] - 1
        dtype_buffer[0] = len(dtype_bytes)
        dtype_buffer[1:(1 + len(dtype_bytes))] = torch.from_numpy(
            numpy.frombuffer(dtype_bytes, dtype=numpy.uint8).copy()
        )

        ndim = input.ndim

    else:
        if not isinstance(input, _Skipped):
            raise EasierJitException(
                f"runtime value {input} of type {type(input)} is not expected"
            )
        
        if not torch.any(can_recv_from):
            raise EasierJitException(
                "Unexpected HaloExchanger without any input"
            )

    dtypes_buffer = _exchange(dtype_buffer)
    ndims_buffer = _exchange(torch.tensor([ndim], dtype=torch.int64))

    # nzep stands for Non-Zero ElemPart
    nzep_dtypes = set()
    if can_recv_from.any():
        # split returns an empty tensor if input is empty
        for nzep_dtype_buffer in dtypes_buffer[can_recv_from].split(1, dim=0):
            # dtypes_buffer: (N, 1000)
            # split: [(1,1000), (1,1000), ...]
            nzep_dtype_buffer = nzep_dtype_buffer[0]
            u8_len = int(nzep_dtype_buffer[0])
            nzep_dtypes.add(pickle.loads(
                nzep_dtype_buffer[
                    1:(1 + u8_len)
                ].to(torch.uint8).numpy(force=True).tobytes()
            ))

    #
    # - Validate dtype and ndim are consistent among workers
    #   involved in the halo_xchg;
    # - Exchange shape, the size is depended on ndim
    #
    if isinstance(input, torch.Tensor):
        # Validate dtype with others, if there are any
        if not all(d == input.dtype for d in nzep_dtypes):
            raise EasierJitException(
                "dtypes of HaloExchanger are not the same:"
                f" {nzep_dtypes}"
            )
        dtype = input.dtype

        # Validate ndim with others, if there are any
        if not torch.all(ndims_buffer[can_recv_from] == ndim):
            raise EasierJitException(
                "ndim of HaloExchanger are not the same:"
                f" {ndims_buffer[can_recv_from]}"
            )

        shape_buffer = torch.tensor(input.shape, dtype=torch.int64)

    else:
        assert isinstance(input, _Skipped)

        # Unique dtype
        if len(nzep_dtypes) != 1:
            raise EasierJitException(
                "dtypes of HaloExchanger are not the same:"
                f" {nzep_dtypes}"
            )
        dtype = nzep_dtypes.pop()

        # Unique ndim
        nzep_ndims = ndims_buffer[can_recv_from].unique()
        if nzep_ndims.shape[0] > 1:
            raise EasierJitException(
                "ndim of HaloExchanger are not the same:"
                f" {nzep_ndims}"
            )
        
        ndim = int(nzep_ndims[0])
        shape_buffer = torch.full((ndim,), -1, dtype=torch.int64)


    shapes_buffer = _exchange(shape_buffer)
    if isinstance(input, torch.Tensor):
        # Validate shape[1:] with others, if there are any
        if not torch.all(
            shapes_buffer[can_recv_from][:, 1:] == shape_buffer[1:]
        ):
            raise EasierJitException(
                "shape[1:] of HaloExchanger are not the same:"
                f" {shapes_buffer[can_recv_from][:, 1:]}"
            )
        subshape = tuple(input.shape[1:])
    
    else:
        assert isinstance(input, _Skipped)

        # Unique shape[1:]
        nzep_subshape_buffer = \
            shapes_buffer[can_recv_from][:, 1:].unique(dim=0)
        if nzep_subshape_buffer.shape != (1, ndim - 1,):
            raise EasierJitException(
                "shape[1:] of HaloExchanger are not the same:"
                f" {nzep_subshape_buffer}"
            )
        subshape = tuple(nzep_subshape_buffer[0].tolist())

    return subshape, dtype

def allgather_meta_for_collective_input(
    input: Union[torch.Tensor, _Skipped]
) -> Tuple[Tuple[int, ...], torch.dtype]:
    """
    This will be a fully collective call, all ranks must be involved.
    
    Available scenarios include EASIER aggregators and Reducers.
    """
    dist_env = get_runtime_dist_env()

    if isinstance(input, torch.Tensor):
        arg_skipped = torch.tensor([0], device=dist_env.comm_device)

    elif isinstance(input, _Skipped):
        arg_skipped = torch.tensor([1], device=dist_env.comm_device)

    else:
        raise EasierJitException(
            f"runtime value {input} of type {type(input)} is not expected"
        )

    # The first communication API must be fully collective, to avoid getting
    # mixed with P2P etc. 
    arg_skipped_flags = dist_env.all_gather_into_tensor(arg_skipped)

    # at least one rank has shape info
    info_sender = (arg_skipped_flags == 0).argwhere().ravel()[0]

    if info_sender == dist_env.rank:
        [dtype, subshape] = dist_env.broadcast_object_list(
            info_sender,
            [input.dtype, input.shape[1:]]  # type: ignore
        )
    else:
        [dtype, subshape] = dist_env.broadcast_object_list(
            info_sender
        )

    if isinstance(input, torch.Tensor):
        if input.shape[1:] != subshape:
            raise EasierJitException(
                "shape[1:] of collective inputs are not the same:"
                f" {input.shape[1:]} and {subshape}"
            )
        if input.dtype != dtype:
            raise EasierJitException(
                "dtype of collective inputs are not the same:"
                f" {input.dtype} and {dtype}"
            )

    return tuple(subshape), dtype


class EvaluationHandlerBase:
    """
    A Handler can be registered into the JitEngine to serve as
    pre/post-evaluation hooks of a single Node in the runtime.

    All handlers are run in a recursion manner. The NodeEvaluationHandler
    that really evaluates the Node (i.e. resulting in Tensor(s)) must be
    the innermost of the recursion.

    For implementers:

    Any Handler if_xxx method (e.g. if_call_function) can call
    `super().if_call_function()` to enter the recursion, like:
    ```
    class MyHandler(EvaluationHandlerBase):
        def if_call_function(self, function, args, kwargs):
            # pre hooks
            # derived Handlers may inspect the IR Node here
            if function in easier.aggregator:
                # may change the args
                args = (0,) + args[1:]

            # enter the recursion
            # the `if_call_function` method of the next Handler will be called
            result = super().if_call_function(self, function, args, kwargs)
            # ~~~~                                            ~~~~~~~~~~~~
            # get the result                     when needed, transform the arg

            # post hooks
            print(result)

            return result
    ```

    NOTE
    -   self.dispatch_node() will also enter recursion.
    -   Positional arguments can be passed in as keyword args.
        For Selector/Reducer, we can use `normalize_selector_call_into_args`
        etc. to _normalize_ `*args **kwargs` into all positional args.
    """
    def __init__(self) -> None:
        # Let JitEngine initialize and wire up all registered Handlers.
        self.next: Optional[EvaluationHandlerBase] = None
        self.current_module: esr.Module
        self.current_graph: Graph

        # 
        # Context variables, the lifetime is the scope of `Module.forward()`
        # =================
        #
        # All Handlers share the same stackframe dict, which is normally
        # created and managed by the JitEngine
        self.stackframe: Dict[Node, RuntimeValue]

        # 
        # Context variables, the lifetime is the execution period of a Node.
        # =================
        self.current_node: Node

    def _dispatch_next(self, args, kwargs):
        assert self.next is not None, \
            "The innermost Handler shouldn't call super().if_xxx method" \
            " (normally we need to put a NodeEvaluationHandler innermost)"

        result = self.next.dispatch_node(self.current_node, args, kwargs)

        assert self.current_node is self.next.current_node, \
            "Currently we don't expect Handlers rebind self.current_node"
        # TODO if we expect recursively run Handlers rebind self.current_node
        # we may enable:
        # self.current_node = self.next.current_node

        return result

    def dispatch_node(
        self,
        node: Node, 
        args: Tuple[RuntimeValue, ...],
        kwargs: Dict[str, RuntimeValue]
    ) -> RuntimeValue:
        self.current_node = node

        root = self.current_module

        if node.op == FX.GET_ATTR:
            assert isinstance(node.target, str)
            val = self.if_get_attr(node.target)

        elif node.op == FX.CALL_FUNCTION:
            assert callable(node.target)
            val = self.if_call_function(node.target, args, kwargs)

        elif node.op == FX.CALL_METHOD:
            assert isinstance(node.target, str)
            val = self.if_call_method(node.target, args, kwargs)

        elif node.op == FX.CALL_MODULE:
            submod_path = cast(str, node.target)
            callee = root.get_submodule(submod_path)
            val = self.if_call_module(callee, args, kwargs)
        
        elif node.op == FX.OUTPUT:
            return None

        else:
            assert False, f"Unexpected FX Node op {node.op}"

        # TODO certain Handlers like NodeEvalHandler may skip super().dispatch
        # and immediately return _Skipped(), making the setting and the check
        # of self.current_node not working.
        # However NodeEvalHandler is the innermost Handler so it may be OK.
        # But it will be better if we make such checks _pre/_post hook of
        # dispatch_node() method itself, just like DistEnv and DataLoader.
        assert self.current_node is node, \
            "Currently we don't expect Handlers rebind self.current_node"

        return val

    def if_get_attr(self, attr_path: str) -> RuntimeValue:
        # As a result of not passing attr_path, function, submod etc.
        # into _dispatch_next, we make these variables auxiliary only
        # and Handler methods cannot change them within the recursion.
        # TODO if we want to make function/submod arg hookable, we may add
        # an extra route to bypass dispatch_node().
        return self._dispatch_next((), {})

    def if_call_function(
        self,
        function: Callable,
        args: Tuple[RuntimeValue, ...],
        kwargs: Dict[str, RuntimeValue]
    ) -> RuntimeValue:
        return self._dispatch_next(args, kwargs)
    
    def if_call_method(
        self,
        method_name: str,
        args: Tuple[RuntimeValue, ...],
        kwargs: Dict[str, RuntimeValue]
    ) -> RuntimeValue:
        return self._dispatch_next(args, kwargs)
    
    def if_call_module(
        self,
        submod: torch.nn.Module,
        args: Tuple[RuntimeValue, ...],
        kwargs: Dict[str, RuntimeValue]
    ) -> RuntimeValue:
        return self._dispatch_next(args, kwargs)


class NodeEvaluationHandler(EvaluationHandlerBase):
    """
    The most essential Handler to evaluate each Node.

    If a Node has StaticNodeMeta(Role.DISTRIBUTED, batch_size=0) it will be
    skipped and a debug marker object _Skipped() will be put to the stackframe.
    (But HaloExchangers are still evaluated)

    Must be the innermost Handler.
    No more dispatch_next() i.e. super().if_xxx() is called.
    """
    def dispatch_node(self, node: Node, args, kwargs):
        if self.is_skippable(node):
            # Skip according to the static metadata, so the result will be
            # consistent among sessions.
            res = _skipped

        else:
            res = super().dispatch_node(node, args, kwargs)

            # TODO handle _SKipped here
            self.handle_result_runtime_metadata(node, res)

        self.stackframe[node] = res

        return res
    
    def is_skippable(self, node: Node):
        is_halo_exchanger = \
            node.op == FX.CALL_MODULE and isinstance(
                self.current_module.get_submodule(cast(str, node.target)),
                HaloExchanger
            )

        node_meta = get_static_node_metadata(node)
        
        return node_meta.role == Role.DISTRIBUTED \
            and node_meta.batch_size == 0 \
            and (not is_halo_exchanger)
    
    def handle_result_runtime_metadata(self, node, res):
        node_meta = get_static_node_metadata(node)
        prev_runtime_meta = get_runtime_tensor_metadata(node)

        result_runtime_meta = tree_map(
            res, lambda x: _get_runtime_metadata_from_value(node_meta.role, x)
        )

        # TODO if we support meta changes on the fly, we can just check
        # if batch sizes change.
        if prev_runtime_meta != result_runtime_meta:
            raise EasierJitException(
                "The properties of the result value of the operation"
                f" {node.target} changes:"
                f" {prev_runtime_meta} => {result_runtime_meta}"
            )

    def if_get_attr(self, attr_path: str) -> RuntimeValue:
        submod_path, _sep, attr_name = attr_path.rpartition(".")
        submod = self.current_module.get_submodule(submod_path)
        obj = getattr(submod, attr_name)
        return obj

    def if_call_function(self, function, args, kwargs) -> RuntimeValue:
        if function in _EsrMod.easier_aggregators:
            if isinstance(args[0], _Skipped):
                dist_env = get_runtime_dist_env()
                result_runtime_meta = get_runtime_tensor_metadata(
                    self.current_node
                )
                assert isinstance(result_runtime_meta, RuntimeTensorMeta)
                assert result_runtime_meta.role == Role.REPLICATED

                vneutral = get_aggregator_neutral_value(
                    function, result_runtime_meta.dtype
                )

                # TODO We may consider to create the neutral value
                # (per node, also per run if k-dim shape can change)
                # only once.
                # Maybe we can store such values for skipped inputs in
                # JitEngine (also HaloExchanger.zero_length_input can be
                # unified in this way).
                # However given that previous Nodes are skipped, it may have
                # saved enough time to create on the fly.
                arg_neutral = torch.full(
                    result_runtime_meta.shape,
                    fill_value=vneutral,
                    dtype=result_runtime_meta.dtype,
                    device=dist_env.comm_device
                )
                args = (arg_neutral,) + args[1:]

        res = function(*args, **kwargs)

        if function is operator.setitem:
            # By default operator.setitem will return None
            # Since the args[0] may be both DISTRIBUTED and REPLICA,
            # we'd better return the concrete value to avoid dealing with None
            # (None is normally identified as REPLICA-only)
            # and align with the Node's role.
            return args[0]

        return res
    
    def if_call_method(self, method_name, args, kwargs) -> RuntimeValue:
        this, *other_args = args

        if not isinstance(this, torch.Tensor):
            # TODO any cases in FX that non-tensor methods are called?
            # maybe `a.split().index(3)` -- `tuple.index` is called?
            raise EasierJitException(
                "expect a method of torch.Tensor to be called,"
                f" but method '{method_name}' of {type(this)} is called"
            )
        
        this_method = getattr(this, method_name)
        # `getattr` on the instance `this` already binds the method to the obj
        # so we don't pass `this` as an argument anymore.
        res = this_method(*other_args, **kwargs)
        return res
    
    def if_call_module(self, submod, args, kwargs) -> RuntimeValue:
        if isinstance(submod, HaloExchanger):
            # Inserted by EASIER, input always on args[0]
            arg = args[0]
            if isinstance(arg, _Skipped):
                args = (submod.zero_length_input,) + args[1:]
        
        if isinstance(submod, esr.Reducer):
            input, opt_out = normalize_reducer_call_into_args(*args, **kwargs)
            if isinstance(input, _Skipped):
                # Pre-Reducer HaloExchanger won't be skipped;
                # Empty input ElemPart won't have reordering Selector.
                if opt_out is not None:
                    return opt_out

                else:
                    dist_env = get_runtime_dist_env()
                    result_runtime_meta = get_runtime_tensor_metadata(
                        self.current_node
                    )
                    assert isinstance(result_runtime_meta, RuntimeTensorMeta)
                    assert result_runtime_meta.role == Role.DISTRIBUTED

                    # Same as Reducer.forward()
                    out = torch.zeros(
                        result_runtime_meta.shape,
                        dtype=result_runtime_meta.dtype,
                        device=dist_env.comm_device
                    )

                    return out
            
            # P.S. local Reducer.idx may be zero-length, but forward() and
            # scatter_reduce_() within can handle it.
        
        res = submod(*args, **kwargs)
        return res


class FisrtRunNodeEvaluationHandler(NodeEvaluationHandler):
    def is_skippable(self, node: Node):
        """
        Additional to HaloExchanger, in the first run we need to exchange
        meta for possible skipped inputs to Reducers, so we enforce the
        evaluation of Reducers too.
        """
        is_reducer = \
            node.op == FX.CALL_MODULE and isinstance(
                self.current_module.get_submodule(cast(str, node.target)),
                esr.Reducer
            )
         
        return (not is_reducer) and super().is_skippable(node)

    def handle_result_runtime_metadata(self, node, res):
        node_meta = get_static_node_metadata(node)
        result_runtime_meta = tree_map(
            res, lambda x: _get_runtime_metadata_from_value(node_meta.role, x)
        )
        set_runtime_tensor_metadata(node, result_runtime_meta)

    def if_call_function(self, function, args, kwargs):
        if function in _EsrMod.easier_aggregators:
            # Set resultant runtime tensor metadata in advance.
            # If on some ranks the input Node is skipped, NodeEvalHandler
            # will use that resultant runtime metadata to create a neutral
            # input.
            subshape, dtype = allgather_meta_for_collective_input(args[0])
            set_runtime_tensor_metadata(
                self.current_node,
                RuntimeTensorMeta(
                    Role.REPLICATED, (1,) + subshape, dtype
                )
            )

        return super().if_call_function(function, args, kwargs)
        

    def if_call_module(self, submod, args: tuple, kwargs: dict):
        if isinstance(submod, HaloExchanger):
            subshape, dtype = exchange_meta_for_halo_exchanger(submod, args[0])
            submod.prepare_buffers(subshape, dtype)
        
        if isinstance(submod, esr.Reducer):
            subshape, dtype = allgather_meta_for_collective_input(args[0])
            set_runtime_tensor_metadata(
                self.current_node,
                RuntimeTensorMeta(
                    Role.DISTRIBUTED, (submod.n,) + subshape, dtype
                )
            )

        return super().if_call_module(submod, args, kwargs)



class JitEngine:
    def __init__(self, module: esr.Module, graph: Graph) -> None:
        self.module = module
        self.graph = graph

        self.run_count = 0

        self.first_run_handlers: List[EvaluationHandlerBase] = [
            FisrtRunNodeEvaluationHandler()  # always last
        ]
        self.runtime_handlers: List[EvaluationHandlerBase] = [
            NodeEvaluationHandler()  # always last
        ]

    def _update_handlers(self, handlers: List[EvaluationHandlerBase]):
        """
        Wire up Handlers.
        Bind Handlers with latest instances of self.module/graph -- in case
        the instances are changed by the JIT passes.
        """
        for i in range(len(handlers) - 1):
            prev = handlers[i]
            next = handlers[i + 1]
            prev.next = next
        
        for h in handlers:
            h.current_module = self.module
            h.current_graph = self.graph

    
    def forward(self):
        if self.run_count == 0:
            ms, gs = [self.module], [self.graph]

            ms, gs = passes.propagate_static_node_metadata(ms, gs)

            [self.module], [self.graph] = ms, gs

            self._update_handlers(self.first_run_handlers)
            handlers = self.first_run_handlers

        else:
            handlers = self.runtime_handlers
        outermost_handler = handlers[0]

        
        stackframe: Dict[Node, RuntimeValue] = {}
        for h in handlers:
            h.stackframe = stackframe

        # TODO make eval_args_kwargs Handler recursion-style methods
        def _eval(x):
            # instead sf.get(x,x) we want to also check type consistency.
            if isinstance(x, Node):
                return stackframe[x]
            else:
                return x
        
        for node in list(self.graph.nodes):
            args = tuple(tree_map(v, _eval) for v in node.args)
            kwargs = {k: tree_map(v, _eval) for k, v in node.kwargs.items()}

            outermost_handler.dispatch_node(node, args, kwargs)  # type: ignore

        if self.run_count == 0:
            ms, gs = [self.module], [self.graph]

            # ms, gs = passes.fuse(ms, gs)
            # ms, gs = passes.codegen(ms, gs)

            [self.module], [self.graph] = ms, gs
            self._update_handlers(self.runtime_handlers)

        stackframe.clear()

        self.run_count += 1

