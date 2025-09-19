# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import contextlib
import dataclasses
from typing import Callable, Dict, List, Optional, Self, Sequence, Tuple, TypeAlias, cast

from torch.fx import Node, Graph
from torch.nn.modules import Module

from easier.core.jit import EasierTracer
import easier.core.module as esr
from easier.core.passes.utils import \
    FX, EasierInterpreter, OrderedSet, get_easier_objects, normalize_reducer_call_into_args, \
    get_node_inplace_arg, get_easier_tensors
from easier.core.utils import EasierJitException


class Jvp(esr.Module):
    """
    """
    def __init__(self):
        super().__init__()

        inputs: Sequence[esr.Tensor]
        outputs: Sequence[esr.Tensor]

        vector: Sequence[esr.Tensor]
        product: Sequence[esr.Tensor]
        # TODO how about tangents_in tangents_out?
    
    def forward(self):
        raise EasierJitException()
    

"""
TODO
-   Calculate connectivity between each op to a specified input/ouput,
    ops that aren't connected are not transformed to forward AD.
    -   all "sources/sinks" are in the middle of the graph.
    -   Nodes to carry dual numbers are connected to at least one input and at least one output.
    -   user-defined esr.Tensors, even not specified as input/output,
        as long as it's involved, we need to create tangent Tensor for it,
        as it might be cross-nested-module (for those not, can be totally immediate
        -- does fusion allow to write immediate vars?), but just not exposed via Jvp.
"""




# class TangentFlowPropagation(EasierInterpreter):
#     def __init__(
#         self, modules, graphs,
#         inputs: Sequence[esr.Tensor],
#         outputs: Sequence[esr.Tensor]
#     ):
#         super().__init__(modules, graphs)

#         self.inputs = OrderedSet(inputs)
#         self.outputs = OrderedSet(outputs)

#         # Being tangent carrier or not is time-dependent, some esr.Tensors
#         # may only be treated as carriers, after they are written with values
#         # that have tangents paired.
#         # Alternatively, we can see it as carrier can be trivial -- its tangent
#         # values are all 0s.
#         self.carriers: OrderedSet[TangentCarrier] = OrderedSet()

#         # Adders are callables, effectively delayed the real addition to
#         # `carriers` set. The addition will happen when the dataflow reaches
#         # an JVP output.
#         self.adders: Dict[Node, TangentCarrierAdder] = {}
    
#     def for_each_node(self) -> TangentCarrierAdder:
#         adder = super().for_each_node()
#         self.adders[self.current_node] = adder
#         return adder

#     def _get_carrier_adder(self) -> TangentCarrierAdder:
#         _captured_this = self.current_node

#         arg_adders = list(map(
#             self.adders.__getitem__, self.current_node.all_input_nodes
#         ))

#         def _adder():
#             self.carriers.add(
#                 TangentCarrier(_captured_this, _captured_this)
#             )
#             for arg_adder in arg_adders:
#                 arg_adder()

#         return _adder

#     def if_get_attr(self, submod_path: str, attr_name: str, attr_val) -> TangentCarrierAdder:
#         if attr_val in self.inputs:
#             return self._get_carrier_adder()
#         else:
#             # For non-JVP input Tensor, at the moment it's get-attr-ed,
#             # it's not a tangent carrier yet.
#             # It's only after it's written with values that have tangent paired
#             # does it becomes a carrier.
#             return lambda: None

#     def _on_operation(self, written_node: Optional[Node]) -> TangentCarrierAdder:
#         """
#         Args:
#         -   mutable: only regarding the 1st arg

#         NOTE about views:
#         We can only tell if the input Node is a derived view or not
#         in runtime by checking the memory address of the tensor.
#         (e.g. `y = x.reshape().reshape().reshape()`, it's hard to tell
#         if `y` is a view of `x`)

#         This difficulty is eliminated since we have `data_dependency_analysis`
#         pass in the runtime, it requires explicit `clone()` calls after
#         operations that create views. So we don't need to worry about views
#         even in pre-`compile()` subprocedures (like `jvp()`) or AOT passess.
        
#         So, we can be confident that by checking the immediate Node written
#         by some operation, we can identify the Tensor instance that's written:
#         -   If input Node is 'get_attr', an esr.Tensor instance is written;
#         -   Otherwise, an immediate result Tensor is written;
#         """
#         if written_node is not None:
#             if written_node.op == FX.GET_ATTR:
#                 attr = self.modules[0].get_parameter(
#                     cast(str, written_node.target)
#                 )
#                 if attr in self.outputs:  # type: ignore
#                     1
        
#         return self._get_carrier_adder()
        
#         # Because of the enforced clone-view rule, no matter this inplace Node
#         # has users (user must be a `clone` call) or not, it's not a view.
#         # I.e. the tangent in this moment must be cloned too.
    
#     def if_call_function(self, function) -> TangentCarrierAdder:
#         out = get_node_inplace_arg(self.current_node)
#         return self._on_operation(out)
    
#     def if_call_method(self, method_name: str) -> TangentCarrierAdder:
#         if method_name.endswith('_'):
#             out = self.current_node.args[0]
#             assert isinstance(out, Node)
#         else:
#             out = None
#         return self._on_operation(out)
    
#     def if_call_module(self, submod: Module) -> TangentCarrierAdder:
#         if isinstance(submod, esr.Module):
#             """
#             Given an esr.Tensor `p` both used in parent Module and submod,
#             if submod ever does `p[:] = f(tangent_carrier)`, `p` becomes a
#             tangent carrier for parent Module too.

#             However, `p`'s tangent Tensor is only used after the submod call.
#             This the same as normal attribute Tensors too.
#             """
#             def _submod_carriers_adder():
#                 return None
            
#             return _submod_carriers_adder

#         else:
#             if isinstance(submod, esr.Reducer):
#                 input, out = normalize_reducer_call_into_args(
#                     *self.current_node.args, **self.current_node.kwargs
#                 )
#                 assert isinstance(out, Node)
#             else:
#                 out = None
#             return self._on_operation(out)


@dataclasses.dataclass
class _PropStackTrace:
    """
    This is basically the call stack plus the IR offset.

    However, offsets (Nodes) are not recorded for all stackframe. Because we
    are taking the union of propagated tangent flows if a Module is called
    multiple times, only the first involvement of a Tensor in called-once
    Module makes sense.
    """
    stackframe: Sequence[esr.Module]
    current_node: Node  # a Node in the innermost Node

class TangentFlowPropCtx:
    def __init__(self, primal_srcs: OrderedSet[esr.Tensor]) -> None:
        #
        # Effectively constant data
        # =============
        # fx.Graphs for primal Modules.
        # NOTE these graphs are generated by EasierTracer once and discarded,
        # only for creating the JVP Module. With JIT 'torch' backend these
        # Modules will be traced again.
        self.graphs: Dict[esr.Module, Graph] = {}
        # All esr.Tensors accessible to one Module and its submodules.
        self.accessible_tensors: Dict[esr.Module, OrderedSet[esr.Tensor]] = {}

        #
        # Propagator-populated data
        # =============
        # How many times a submod instance is called, if one has been called
        # multiple times, we need to union the tangent flow on its graph.
        self.call_counts: Dict[esr.Module, int] = {}

        self.prop_flow: Dict[esr.Module, OrderedSet[Node]] = {}

        # The snapshot of involved primal esr.Tensor to a Module at the
        # beginning of one turn of the propagation on it:
        # - recursively including Tensors in further nested sub Modules
        # - not including Tensors not accessible from this Module and
        #   its sub Modules, i.e. the intersection
        # If the snapshot doesn't change, given the static nature of EASIER
        # graphs, we don't need to propagate again.
        self.primal_srcs_snapshot: Dict[esr.Module, OrderedSet[esr.Tensor]] = {}

        # The dynamically increased set of primal esr.Tensors that are ever
        # involved in the global tangent flow.
        self.primal_srcs = OrderedSet(primal_srcs)

        # Use tuple instead of list for its immutability, so new StackTrace
        # can safely store it without explicit copy.
        self.prop_stackframe: Tuple[esr.Module, ...] = ()

        self.primal_earliest_tangent_pairing: Dict[esr.Tensor, _PropStackTrace] = {}
    
    def init_module_once(self, module: esr.Module):
        if module not in self.graphs:
            graph = EasierTracer().trace(module)
            self.graphs[module] = graph
        
        if module not in self.accessible_tensors:
            self.accessible_tensors[module] = OrderedSet(get_easier_tensors([module]))

        # Let caller increment.
        self.call_counts.setdefault(module, 0)

        self.prop_flow.setdefault(module, OrderedSet())
        self.primal_srcs_snapshot.setdefault(module, OrderedSet())
    
    @contextlib.contextmanager
    def new_prop_stackframe(self, module: esr.Module):
        self.prop_stackframe += (module,)
        yield
        self.prop_stackframe = self.prop_stackframe[:-1]


class TangentFlowPropagator(EasierInterpreter):
    def __init__(
        self, module: esr.Module, graph: Graph,
        ctx: TangentFlowPropCtx,
        reverse=False
    ):
        super().__init__([module], [graph], reverse)

        self.module = module

        self.ctx = ctx

    
    # TODO if a esr.Tensor involved in tangent flow but is not used across
    # submodules, we don't really need a persistent tangent esr.Tensor for it.

    def run(self) -> Self:
        ctx = self.ctx
        ctx.init_module_once(self.module)
        ctx.call_counts[self.module] += 1

        with ctx.new_prop_stackframe(self.module):

            latest_primals_snapshot = ctx.primal_srcs_snapshot[self.module]

            # set difference
            assert len(latest_primals_snapshot - ctx.primal_srcs) == 0, \
                "Tangent sources should grow incrementally on the same Module"

            if len(ctx.primal_srcs - latest_primals_snapshot) > 0:
                # Enter nested propagation only when tangent sources are updated
                # on the root submodule.
                super().run()
            
            accessible = ctx.accessible_tensors[self.module]
            new_primals_snapshot = OrderedSet(
                ps
                for ps in ctx.primal_srcs
                if ps in accessible
            )
            ctx.primal_srcs_snapshot[self.module] = new_primals_snapshot

        return self

    def if_get_attr(self, submod_path: str, attr_name: str, attr_val):
        # When an esr.Tensor is just 'get_attr'-ed, it may not be regarded to
        # be paired with tangent yet. This may happen on esr.Tensors that are
        # not specified in `jvp()` arguments.
        # But in the middle of the graph, the Tensor may be written with
        # values that carry tangents.
        # At that timing, we mark the Node as tangent flow, and record the
        # Tensor to be paired with tangent, 
        ctx = self.ctx
        if attr_val in ctx.primal_srcs:
            ctx.prop_flow[self.module].add(self.current_node)
            ctx.primal_earliest_tangent_pairing.setdefault(
                attr_val,
                _PropStackTrace(ctx.prop_stackframe, self.current_node)
            )
    
    def if_call_function(self, function: Callable):
        inplace_arg = get_node_inplace_arg(self.current_node)
        self._on_operation(inplace_arg)
    
    def if_call_method(self, method_name: str):
        inplace_arg = self.current_node.args[0]
        assert isinstance(inplace_arg, Node)
        self._on_operation(inplace_arg)

    
    def if_call_module(self, submod: Module):
        if isinstance(submod, esr.Module):
            """
            Given an esr.Tensor `p` both used in parent Module and submod,
            if submod ever does `p[:] = f(tangent_carrier)`, `p` propagates
            tangent into parent Module too.

            However, `p`'s tangent Tensor is only used after the submod call.
            This the same as normal attribute Tensors too.
            """
            submod_g = self.ctx.graphs[submod]
            sub_prop = TangentFlowPropagator(
                submod, submod_g, self.ctx, self.reverse
            ).run()
        
        else:
            if isinstance(submod, esr.Reducer):
                input, out = normalize_reducer_call_into_args(
                    *self.current_node.args, **self.current_node.kwargs
                )
                assert out is None or isinstance(out, Node)
            else:
                out = None

            self._on_operation(out)
    
    def _on_operation(self, written_node: Optional[Node]):
        """
        NOTE about views:
        We can only tell if the input Node is a derived view or not
        in runtime by checking the memory address of the tensor.
        (e.g. `y = x.reshape().reshape().reshape()`, it's hard to tell
        if `y` is a view of `x`)

        This difficulty is eliminated since we have `data_dependency_analysis`
        pass in the runtime, it requires explicit `clone()` calls after
        operations that create views. So we don't need to worry about views
        even in pre-`compile()` subprocedures (like `jvp()`) or AOT passess.
        
        So, we can be confident that by checking the immediate Node written
        by some operation, we can identify the Tensor instance that's written:
        -   If input Node is 'get_attr', an esr.Tensor instance is written;
        -   Otherwise, an immediate result Tensor is written;
        """
        if written_node is not None:
            if written_node.op == FX.GET_ATTR:
                attr_tensor = self.modules[0].get_parameter(
                    cast(str, written_node.target)
                )
            else:
                1
                # write to immediate tensors
        
        
        # Because of the enforced clone-view rule, no matter this inplace Node
        # has users (user must be a `clone` call) or not, it's not a view.
        # I.e. the tangent in this moment must be cloned too.


class JvpTransformer(EasierInterpreter):
    """
    Recursive transformation from a pair of Module/Graph for primal calculation
    to a Module/Graph for jvp.

    The recursion happens on JvpTransformer rather than `easier.jvp()`,
    the result of this is that there won't be explicit easier.Tensors to
    store tangents at the boundary of nested easier.Modules.
    """
    def __init__(self, root_jvp: esr.Module, module: esr.Module):
        # All referenced nested esr.Modules will be created a paired sub
        # Jvp module, cached by instance.
        self.root_jvp = root_jvp
        graph = EasierTracer().trace(module)

        super().__init__([module], [graph])


    def if_call_module(self, submod: Module):
        if isinstance(submod, esr.Module):
            # Nested easier.Module, must be JVP-ed.
            sub_jvp_transformer = JvpTransformer(self.root_jvp, submod).run()



        else:
            1

        return super().if_call_module(submod)

def jvp(
    module: esr.Module,
    inputs: Sequence[esr.Tensor],
    outputs: Sequence[esr.Tensor]
) -> Jvp:
    """
    Remarks:
    -   Input and output easier.Tensors are all mutable.
        Even, for example, `input[:] = f(output)`.

    -   If the same easier.Tensor is specified as both input and output,
        the tangent easier.Tensor for it will also be both included in both
        input and output tangent easier.Tensors.
    
    -   In `inputs` or `outputs`, the same easier.Tensor instance cannot be
        specified twice.
    """
    def _check_dup_arg(args: Sequence[esr.Tensor], param_name: str):
        counts = {}
        for arg in args:
            counts.setdefault(arg, 0)
            counts[arg] += 1
        for arg, c in counts.items():
            if c > 1:
                pos = args.index(arg)
                raise ValueError(
                    f"The {pos}-th easier.Tensor gets specified {c} times"
                    f" in {param_name}"
                )
    _check_dup_arg(inputs, 'inputs')
    _check_dup_arg(outputs, 'outputs')



    # TODO the resultant Jvp module should be disconnected from `module`
    # in a way that get_easier_object(jvp) does not include moudle.
    # TODO assign meaningful names to:
    # - Jvp.inputs, like `x` if input is InputModule.x
    # - Jvp.vector, like `tan_x`


    jvp_transformer = JvpTransformer(module).run()

    return Jvp()