# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import dataclasses
import math
import operator
from types import ModuleType
from typing import \
    Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union, cast
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
from easier.core.passes.static_metadata_propagation import \
    StaticTensorMeta, get_static_metadata, Role
from easier.core.passes.utils import \
    FX, get_easier_objects, EasierInterpreter, tree_map
from easier.core.utils import EasierJitException, logger
from easier.core.runtime.dist_env import get_runtime_dist_env
from easier.core.runtime.modules import HaloExchanger, tuple_getitem

KEY__JITENGINE_RUNTIMEMETA = 'easier_jitEngine_runtimeMeta'


@dataclasses.dataclass
class RuntimeTensorMetadata:
    """
    For distributed Nodes only.
    May be nested.
    """
    shape: Tuple[int, ...]
    dtype: torch.dtype


def get_runtime_metadata(node: Node) -> RuntimeTensorMetadata:
    return node.meta[KEY__JITENGINE_RUNTIMEMETA]


def get_runtime_metadata_from_value(
    val: Union[torch.Tensor, int, float]
) -> RuntimeTensorMetadata:
    if isinstance(val, torch.Tensor):
        return RuntimeTensorMetadata(tuple(val.shape), val.dtype)
    elif isinstance(val, bool):
        return RuntimeTensorMetadata((), torch.bool)
    elif isinstance(val, int):
        # PyTorch Python wrapper isn't aware of Python int precision,
        # so we treat ints as current minimum int32 dtype
        # so they are compatible with any torch tensor with int-kind dtype.
        return RuntimeTensorMetadata((), torch.int32)
    elif isinstance(val, float):
        # Same as int32, treat Python float as current minimum float32.
        return RuntimeTensorMetadata((), torch.float32)
    else:
        # NOTE for types that cannot explicitly appear on `Node.args`,
        # (`torch.Tensor` is one of such types), their metadata is always
        # propagated and carried by their corresponding `Node[op='get_attr']`.
        # We don't expect to see them here.
        raise EasierJitException(f'Value {val} cannot have associated metadata')



class EvaluationHandlerBase:
    def __init__(self) -> None:
        # Let JitEngine initialize and wire up all registered Handlers.
        self.next: Optional[EvaluationHandlerBase] = None
        self.current_module: esr.Module
        self.current_graph: Graph

        # IR-level session data
        self.stackframe: Dict[Node, torch.Tensor]

        # Node-level session data
        self.current_node: Node

    def _dispatch_next(self):
        assert self.next is not None, \
            "The innermost Handler shouldn't super().if_xxx method" \
            " (normally the innermost Handler is the NodeEvaluationHandler)"
        return self.next.dispatch_node(self.current_node)
    
    def eval_arg_node(self, arg: Node):
        return self.stackframe[arg]
    
    def eval_args_kwargs(self):
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
        return args, kwargs
    
    def dispatch_node(self, node: Node):
        self.current_node = node

        root = self.current_module

        if node.op == FX.GET_ATTR:
            path = cast(str, node.target)
            submod_path, _sep, attr_name = path.rpartition(".")
            submod = root.get_submodule(submod_path)
            obj = getattr(submod, attr_name)
            val = self.if_get_attr(submod_path, attr_name, obj)

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

        # NOTE handler subprocedure may change the self.current_node
        self.stackframe[self.current_node] = val
        return val

    def if_get_attr(self, submod_path: str, attr_name: str, attr_val):
        return self._dispatch_next()

    def if_call_function(self, function):
        return self._dispatch_next()
    
    def if_call_method(self, method_name):
        return self._dispatch_next()
    
    def if_call_module(self, submod):
        return self._dispatch_next()


class NodeEvaluationHandler(EvaluationHandlerBase):
    """
    Must be the innermost Handler.
    No more dispatch_next() i.e. super().if_xxx() is called.
    """
    def dispatch_node(self, node: Node):
        smeta = get_static_metadata(node)
        if smeta.role == Role.DISTRIBUTED and smeta.batch_size == 0:
            # Skip according to the static metadata, so the result will be
            # consistent among sessions.
            return _skipped  # type: ignore

        prev_rmeta = get_runtime_metadata(node)

        v = super().dispatch_node(node)

        new_rmeta = tree_map(v, get_runtime_metadata_from_value)
        if prev_rmeta != new_rmeta:
            raise EasierJitException(
                "The properties of the result value of the operation"
                f" {node.target} changes: {prev_rmeta} => {new_rmeta}"
            )

        return v

    def if_get_attr(self, submod_path: str, attr_name: str, attr_val):
        return attr_val

    def if_call_function(self, function):
        args, kwargs = self.eval_args_kwargs()
        res = function(*args, **kwargs)

        if function is operator.setitem:
            # by default operator.setitem will return None
            return args[0]
        
        if function in _EsrMod.easier_aggregators:
            # The input to he aggregator may have been skipped,
            # we need to broadcast the shape[1:] from other Node.
            (arg, *_args), *_kwargs = self.eval_args_kwargs()
            if isinstance(arg, _Skipped):
                dist_env = get_runtime_dist_env()

                prev_rmeta = get_runtime_metadata(self.current_node)
                vneutral = get_aggregator_neutral_value(
                    function, prev_rmeta.dtype
                )
                v = torch.full(
                    prev_rmeta.shape, fill_value=vneutral,
                    dtype=prev_rmeta.dtype, device=dist_env.comm_device
                )
                return v

        return res
    
    def if_call_method(self, method_name):
        (this, *args), kwargs = self.eval_args_kwargs()

        if isinstance(this, torch.Tensor):
            # TODO are there any cases pytorch returns a non-Tensor object?
            raise EasierJitException(
                "expect a method of torch.Tensor to be called"
            )
        
        method = getattr(this, method_name)
        # `getattr` on the instance `this` already binds the method to the obj
        # so we don't pass `this` as an argument.
        res = method(*args, **kwargs)
        return res
    
    def if_call_module(self, submod):
        args, kwargs = self.eval_args_kwargs()


        if isinstance(submod, HaloExchanger):
            arg = args[0]
            if isinstance(arg, torch.Tensor):
                arg_skipped = torch.tensor([0], device=dist_env.comm_device)

            elif isinstance(arg, _Skipped):
                arg_skipped = torch.tensor([1], device=dist_env.comm_device)

            else:
                raise EasierJitException(
                    f"runtime value {arg} is not expected"
                )

        res = submod(*args, **kwargs)
        return res


class _Skipped:
    pass

_skipped = _Skipped()


def get_aggregator_neutral_value(aggregator, dtype: torch.dtype):
    if dtype.is_floating_point:
        finfo = torch.finfo(dtype)
        vmax = finfo.max
        vmin = finfo.min
    else:  # TODO exclude complex dtype, it's not float too
        iinfo = torch.iinfo(dtype)
        vmax = iinfo.max
        vmin = iinfo.min

    vneutral = {
        esr.sum: 0,
        esr.prod: 1,
        esr.norm: 0,
        esr.max: vmax,
        esr.min: vmin
    }[aggregator]
    return vneutral

class FisrtRunNodeEvaluationHandler(NodeEvaluationHandler):
    def dispatch_node(self, node: Node):
        smeta = get_static_metadata(node)
        if smeta.role == Role.DISTRIBUTED and smeta.batch_size == 0:
            # Skip this node. Store a debug marker object
            return _skipped  # type: ignore

        v = super().dispatch_node(node)
        
        rmeta = tree_map(v, get_runtime_metadata_from_value)
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
        self._init_handlers(self.first_run_handlers)

        self.runtime_handlers: List[EvaluationHandlerBase] = [
            NodeEvaluationHandler()  # always last
        ]
        self._init_handlers(self.runtime_handlers)
    
    def _init_handlers(self, handlers: List[EvaluationHandlerBase]):
        for h in handlers:
            h.current_module = self.module
            h.current_graph = self.graph

        last = handlers[-1]
        for h in reversed(handlers[:-1]):
            h.next = last
            last = h

    
    def forward(self):
        if self.run_count == 0:
            handlers = self.first_run_handlers
        else:
            handlers = self.runtime_handlers
        
        stackframe = {}
        for h in handlers:
            h.stackframe = stackframe
        
        outermost_handler = handlers[0]
        for node in list(self.graph.nodes):
            outermost_handler.dispatch_node(node)

        self.run_count += 1
        stackframe.clear()

