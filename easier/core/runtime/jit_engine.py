# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import dataclasses
import math
import operator
from types import ModuleType
from typing import \
    Callable, Dict, Iterator, List, Optional, Sequence, Tuple, TypeAlias, Union, cast
from typing_extensions import Literal
import more_itertools
import os

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
    FX, get_easier_objects, EasierInterpreter, tree_map
from easier.core.utils import EasierJitException, logger
from easier.core.runtime.dist_env import get_runtime_dist_env
from easier.core.runtime.modules import HaloExchanger, tuple_getitem



def _get_runtime_metadata_from_value(
    node_role: Role,
    val: Union[torch.Tensor, int, float]
) -> RuntimeTensorMeta:
    if isinstance(val, torch.Tensor):
        return RuntimeTensorMeta(node_role, tuple(val.shape), val.dtype)

    if node_role != Role.REPLICATED:
        raise EasierJitException()

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
        raise EasierJitException(f'Value {val} cannot have associated metadata')

class _Skipped:
    pass

_skipped = _Skipped()


# NOTE it's possible that FX trace `Tensor.item()` call which results in
# a pure int/float scalar rather than a [0]-shape tensor.
RuntimeValue: TypeAlias = Union[
    torch.Tensor,
    _Skipped,
    Sequence['RuntimeValue']
]

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
        def if_call_function(self, function):
            # pre hooks
            # derived Handlers may inspect the IR Node here
            if function in easier.aggregator:
                function = easier.runtime.my_new_function

                # may change the self.current_node
                self.current_node.target = function

            # enter the recursion
            # the `if_call_function` method of the next Handler will be called
            result = super().if_call_function(self, function)
            # ~~~~                                  ~~~~~~~~
            # get the result                when needed, transform the arg

            # post hooks
            print(result)

            return result
    ```

    P.S. self.dispatch_node() will also enter recursion.
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

    def _dispatch_next(self):
        assert self.next is not None, \
            "The innermost Handler shouldn't call super().if_xxx method" \
            " (normally we need to put a NodeEvaluationHandler innermost)"

        result = self.next.dispatch_node(self.current_node)

        assert self.current_node is self.next.current_node, \
            "Currently we don't expect Handlers rebind self.current_node"
        # TODO if we expect recursively run Handlers rebind self.current_node
        # we may enable:
        # self.current_node = self.next.current_node

        return result

    
    def eval_arg_node(self, arg: Node) -> RuntimeValue:
        return self.stackframe[arg]
    
    def eval_args_kwargs(self) -> Tuple[
        Tuple[RuntimeValue, ...], Dict[str, RuntimeValue]
    ]:
        # NOTE positional arguments can be passed in as keyword args.
        # For Selector/Reducer, we can use `normalize_selector_call_into_args`
        # etc. to _normalize_ `*args **kwargs` into all positional args.

        def _eval(x):
            if isinstance(x, Node):
                return self.eval_arg_node(x)
            else:
                return x
        args = tuple(tree_map(v, _eval) for v in self.current_node.args)
        kwargs = {
            k: tree_map(v, _eval)
            for k, v in self.current_node.kwargs.items()
        }
        return args, kwargs  # type: ignore
    
    def dispatch_node(self, node: Node) -> RuntimeValue:
        self.current_node = node

        root = self.current_module

        if node.op == FX.GET_ATTR:
            assert isinstance(node.target, str)
            val = self.if_get_attr(node.target)

        elif node.op == FX.CALL_FUNCTION:
            assert callable(node.target)
            val = self.if_call_function(node.target)

        elif node.op == FX.CALL_METHOD:
            assert isinstance(node.target, str)
            val = self.if_call_method(node.target)

        elif node.op == FX.CALL_MODULE:
            submod_path = cast(str, node.target)
            callee = root.get_submodule(submod_path)
            val = self.if_call_module(callee)

        else:
            assert False, f"Unexpected FX Node op {node.op}"

        # The same Node key will be written multiple times by all Handlers
        self.stackframe[self.current_node] = val
        return val

    def if_get_attr(self, attr_path: str) -> RuntimeValue:
        return self._dispatch_next()

    def if_call_function(self, function: Callable) -> RuntimeValue:
        return self._dispatch_next()
    
    def if_call_method(self, method_name: str) -> RuntimeValue:
        return self._dispatch_next()
    
    def if_call_module(self, submod: torch.nn.Module) -> RuntimeValue:
        return self._dispatch_next()


class NodeEvaluationHandler(EvaluationHandlerBase):
    """
    The most essential Handler to evaluate each Node.

    If a Node has StaticNodeMeta(Role.DISTRIBUTED, batch_size=0) it will be
    skipped and a debug marker object _Skipped() will be put to the stackframe.
    (But HaloExchangers are still evaluated)

    Must be the innermost Handler.
    No more dispatch_next() i.e. super().if_xxx() is called.
    """
    def dispatch_node(self, node: Node):
        is_halo_exchanger = \
            node.op == FX.CALL_MODULE and isinstance(
                self.current_module.get_submodule(cast(str, node.target)),
                HaloExchanger
            )

        node_meta = get_static_node_metadata(node)
        if node_meta.role == Role.DISTRIBUTED \
        and node_meta.batch_size == 0 \
        and (not is_halo_exchanger):
            # Skip according to the static metadata, so the result will be
            # consistent among sessions.
            return _skipped

        prev_runtime_meta = get_runtime_tensor_metadata(node)

        val = super().dispatch_node(node)

        new_runtime_meta = tree_map(
            val, lambda x: _get_runtime_metadata_from_value(node_meta.role, x)
        )
        if prev_runtime_meta != new_runtime_meta:
            raise EasierJitException(
                "The properties of the result value of the operation"
                f" {node.target} changes:"
                f" {prev_runtime_meta} => {new_runtime_meta}"
            )

        return val

    def if_get_attr(self, attr_path: str) -> RuntimeValue:
        submod_path, _sep, attr_name = attr_path.rpartition(".")
        submod = self.current_module.get_submodule(submod_path)
        obj = getattr(submod, attr_name)
        return obj

    def if_call_function(self, function) -> RuntimeValue:
        args, kwargs = self.eval_args_kwargs()
        res = function(*args, **kwargs)

        if function is operator.setitem:
            # by default operator.setitem will return None
            return args[0]
        
        if function in _EsrMod.easier_aggregators:
            # Inserted by EASIER, input always on args[0]
            (arg, *_args), *_kwargs = self.eval_args_kwargs()
            if isinstance(arg, _Skipped):
                dist_env = get_runtime_dist_env()

                prev_rmeta = get_runtime_tensor_metadata(self.current_node)
                assert isinstance(prev_rmeta, RuntimeTensorMeta)
                vneutral = get_aggregator_neutral_value(
                    function, prev_rmeta.dtype
                )
                v = torch.full(
                    prev_rmeta.shape, fill_value=vneutral,
                    dtype=prev_rmeta.dtype, device=dist_env.comm_device
                )
                return v

        return res
    
    def if_call_method(self, method_name) -> RuntimeValue:
        (this, *args), kwargs = self.eval_args_kwargs()

        if isinstance(this, torch.Tensor):
            # TODO any cases in FX that non-tensor methods are called?
            # maybe `a.split().index(3)` -- `tuple.index` is called?
            raise EasierJitException(
                "expect a method of torch.Tensor to be called"
            )
        
        method = getattr(this, method_name)
        # `getattr` on the instance `this` already binds the method to the obj
        # so we don't pass `this` as an argument.
        res = method(*args, **kwargs)
        return res
    
    def if_call_module(self, submod) -> RuntimeValue:
        args, kwargs = self.eval_args_kwargs()

        if isinstance(submod, HaloExchanger):
            # Inserted by EASIER, input always on args[0]
            arg = args[0]
            if isinstance(arg, _Skipped):
                args = (submod.zero_length_input,) + args[1:]

        res = submod(*args, **kwargs)
        return res



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

class FisrtRunNodeEvaluationHandler(NodeEvaluationHandler):
    def dispatch_node(self, node: Node):
        smeta = get_static_node_metadata(node)
        if smeta.role == Role.DISTRIBUTED and smeta.batch_size == 0:
            # Skip this node. Store a debug marker object
            return _skipped  # type: ignore

        v = super().dispatch_node(node)
        
        rmeta = tree_map(v, _get_runtime_metadata_from_value)
        node.meta[KEY__JITENGINE_RUNTIMEMETA] = rmeta

        return v

    def if_call_function(self, function):
        dist_env = get_runtime_dist_env()

        if function in _EsrMod.easier_aggregators:
            # The input to he aggregator may have been skipped,
            # we need to broadcast the shape[1:] from other Node.
            (arg, *_args), *_kwargs = self.eval_args_kwargs()
            if isinstance(arg, torch.Tensor):
                arg_skipped = torch.tensor([0], device=dist_env.comm_device)

            elif isinstance(arg, _Skipped):
                arg_skipped = torch.tensor([1], device=dist_env.comm_device)

            else:
                raise EasierJitException(
                    f"runtime value {arg} is not expected"
                )

            # TODO note the comm primitive to broadcast the shape info here
            # it cannot be mixed with e.g. P2P because if some ranks have
            # empty partition, they will not invoke P2P calls,
            # and this will cause the P2P to exchange shapes and P2P to
            # exchange halos.
            arg_skipped_flags = dist_env.all_gather_into_tensor(arg_skipped)
            if arg_skipped_flags.sum() > 0:
                # at least one rank has shape info
                info_sender = (arg_skipped_flags == 0).argwhere().ravel()[0]

                if info_sender == dist_env.rank:
                    [dtype, subshape] = dist_env.broadcast_object_list(
                        info_sender,
                        [arg.dtype, arg.shape[1:]]  # type: ignore
                    )
                else:
                    [dtype, subshape] = dist_env.broadcast_object_list(
                        info_sender
                    )

            if arg_skipped == 0:
                v = super().if_call_function(function)
            else:
                vneutral = get_aggregator_neutral_value(function, dtype)

                v = torch.full(
                    (1,) + subshape,
                    fill_value=vneutral,
                    dtype=dtype,
                    device=dist_env.comm_device
                )

            return v

        return super().if_call_function(function)
        

    def if_call_module(self, submod):
        # TODO HaloExchanger input may be 0-length. exchange subshape like aggregators
        if isinstance(submod, HaloExchanger):
            (arg, *_args), *_kwargs = self.eval_args_kwargs()
            if isinstance(arg, torch.Tensor):
                arg_skipped = torch.tensor([0], device=dist_env.comm_device)

            elif isinstance(arg, _Skipped):
                arg_skipped = torch.tensor([1], device=dist_env.comm_device)

            else:
                raise EasierJitException(
                    f"runtime value {arg} is not expected"
                )
            pass 
        return super().if_call_module(submod)


class RewriteTupleGetItemHandler(EvaluationHandlerBase):
    def eval_arg_node(self, arg: Node):
        """
        Ensure any resultant tensor tuples have always been unpacked first.

        So we can make the assertions:
        1.  when nested structure appear, e.g. `torch.stack([a,b,c])`,
            we can be sure that the argument list/tuple/tree
            (which is a list/tuple/tree of Nodes)
            is exactly showing that structure of argument Tensors.

        2.  every Node,
            except the operator itself that is known to return a tuple,
            exactly represents a Tensor.

        We do the check when a Node is used, so that it covers the cases of
        call_function/call_method.
        """
        v = super().eval_arg_node(arg)

        # Use is-not-Tensor check, since the structure may be list/tuple/etc.
        if not isinstance(v, torch.Tensor):
            if self.current_node.op == FX.CALL_FUNCTION \
            and self.current_node.target is tuple_getitem:
                # We are unpacking the tuple, this is the only allowed case
                return v
            else:
                raise EasierJitException("TODO unpack first!")

    def if_call_function(self, function):
        if self.current_node.op == FX.CALL_FUNCTION \
        and self.current_node.target is operator.getitem:
            # here we can be sure that arguments are definitely positional
            (this_val, pos_val), _kwargs = self.eval_args_kwargs()
            assert len(_kwargs) == 0, \
                "Both FX tuple getitem and tensor slicing do not have kwargs"

            # Use is-not-Tensor check, since it may be list/tuple/etc.
            if not isinstance(this_val, torch.Tensor):
                self.current_node.target = tuple_getitem

        return super().if_call_function(function)



class JitEngine:
    def __init__(self, module: esr.Module, graph: Graph) -> None:
        self.module = module
        self.graph = graph

        self.run_count = 0

        self.first_run_handlers: List[EvaluationHandlerBase] = [
            RewriteTupleGetItemHandler(),
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
        
        stackframe = {}
        for h in handlers:
            h.stackframe = stackframe
        
        outermost_handler = handlers[0]
        for node in list(self.graph.nodes):
            outermost_handler.dispatch_node(node)

        if self.run_count == 0:
            ms, gs = [self.module], [self.graph]

            # ms, gs = passes.fuse(ms, gs)
            # ms, gs = passes.codegen(ms, gs)

            [self.module], [self.graph] = ms, gs
            self._update_handlers(self.runtime_handlers)

        stackframe.clear()

        self.run_count += 1

