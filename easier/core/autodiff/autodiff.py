# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import contextlib
import dataclasses
import operator
from typing import Callable, Collection, Dict, List, Literal, Optional, Self, Sequence, Tuple, TypeAlias, Union, cast
from typing_extensions import OrderedDict

import more_itertools
import torch
from torch.fx import Node, Graph, GraphModule
from torch.fx.node import Argument as FxArg
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.modules import Module

from easier.core.jit import EasierTracer
import easier.core.module as esr
from easier.core.passes.collective_initialization import collectively_initialize_and_validate
from easier.core.passes.tensor_grouping import group_tensors, get_node_tensor_group
from easier.core.runtime.jit_engine.jit_engine import get_value_runtime_info
from easier.core.runtime.metadata import Role, RuntimeTensorMeta, StructuredTensorMeta, collect_meta, set_node_meta, get_node_meta, get_runtime_metadata_from_scalar
from easier.core.passes.utils import \
    FX, EasierInterpreter, OrderedSet, SubmodNameAllocator, get_easier_objects, isinst_checker, normalize_reducer_call_into_args, \
    get_torch_func_inplace_arg, get_easier_tensors, get_attr_value, fx_normalize_function_variant_into_kwargs, tree_map, \
    normalize_selector_call_into_args
    
from easier.core.utils import EasierJitException
from easier.core.autodiff.utils import simplify_torchfunc_fx_graph

from .torch_jvp import DiffRuleBase, Differentiability, tangent_rule_registry, differentiabilities

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


FxConst: TypeAlias = Union[int, float, str]



class JvpTransformer(EasierInterpreter):
    """
    Recursive transformation from a pair of Module/Graph for primal calculation
    to a Module/Graph for jvp.

    The recursion happens on JvpTransformer rather than `easier.jvp()`,
    the result of this is that there won't be explicit easier.Tensors to
    store tangents at the boundary of nested easier.Modules.
    """
    def __init__(
        self,
        module: esr.Module,
        jvp_module: Jvp,
        initial_tangent_tensors: Dict[esr.Tensor, esr.Tensor]
    ):
        # TODO move outside
        [module], [graph] = collectively_initialize_and_validate([module])
        [module], [graph] = group_tensors([module], [graph])


        _getattr_vals = set()
        class _SingleGetAttrValidator(EasierInterpreter):
            # torch.fx would likely deduplicate GET_ATTR for multi-aliases
            # attribute Tensor, but the correctness of AD explicitly relies on
            # such a property, therefore we need to validate it.
            def if_get_attr(self, submod_path: str, attr_name: str, attr_val):
                if attr_val in _getattr_vals:
                    raise NotImplementedError(
                        "EASIER currently does not accept that in FX Graph"
                        " two 'get_attr' Nodes point to the same instance"
                    )
                _getattr_vals.add(attr_val)
        _SingleGetAttrValidator([module], [graph]).run()


        super().__init__([module], [graph])

        self.jvp_module = jvp_module
        self.initial_tangent_tensors = initial_tangent_tensors

        self.jvp_graph = Graph()
        self.nodemap_raw2primal: Dict[Node, Node] = {}
        self.nodemap_raw2tangent: Dict[Node, Node] = {}

        # Not only supporting submod, any kinds of attributes are OK.
        self.tangent_name_allocator = SubmodNameAllocator('tangent')



    def _fake_eval_meta_ctor(self, shape, dtype):
        # Special function needed by get_value_runtime_info, to provide
        # Role in TensorMeta.
        ng = get_node_tensor_group(self.current_node)
        if ng is None:
            role = Role.REPLICATED

        else:
            role = Role.DISTRIBUTED
            shape = (1000,) + shape[1:]

        return RuntimeTensorMeta(role, shape, dtype)
    
    def _create_zero_val(
        self, raw_node_arg: Union[Node, Sequence[Node]]
    ) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        """
        `raw_node_arg` is an argument of raw Graph Node, it may be a nested
        structure,
        e.g. torch.cat Node may have `args[0] == [x1, x2, x3]`.
        
        The result may also be a nested structure of many zero tensors.
        """
        def _make(x):
            assert isinstance(x, Node), \
                "In a list, Node and scalar are not expected to be mixed"

            meta = get_node_meta(x)
            assert isinstance(meta, RuntimeTensorMeta), \
                "Value of arg Node cannot be nested structure"

            return torch.zeros(meta.shape, dtype=meta.dtype)

        return tree_map(raw_node_arg, _make)  # type: ignore
    

    def if_get_attr(self, submod_path: str, attr_name: str, attr_val) -> None:
        runtime_meta = get_value_runtime_info(
            self.current_node, attr_val, self._fake_eval_meta_ctor
        )
        set_node_meta(self.current_node, runtime_meta)

        primal_node = self.jvp_graph.node_copy(self.current_node)
        self.nodemap_raw2primal[self.current_node] = primal_node

        if attr_val in self.initial_tangent_tensors:
            # Simplified: it's unlikely the same primal tensor has GET_ATTR
            # Nodes multiple times, therefore it's OK to inject GET_ATTR Nodes
            # for tangent each time.
            tangent_tensor = self.initial_tangent_tensors[attr_val]
            tangent_name = self.tangent_name_allocator.alloc_name(self.jvp_module, attr_name)
            setattr(self.jvp_module, tangent_name, tangent_tensor)

            # Inject tangent node
            tangent_node = self.jvp_graph.get_attr(tangent_name)

            # Bind primal-tangent
            self.nodemap_raw2tangent[self.current_node] = tangent_node



    def if_call_function(self, function: Callable) -> None:
        self._handle_operation(function)
        
        
    def _handle_operation(self, function: Callable):
        """
        Basically we treat all inplace ops as "non-inplace ops plus setitem".

        An important extra property for inplace ops is that the target Node,
        esr.Tensor instance or immediately result, may not be bound with
        tangent initially.
        It's at this inplace Node does the target starts to have tangent.
        """
        #
        # Op category
        #
        if function in tangent_rule_registry:
            rule_cls = tangent_rule_registry[function]
            rule = rule_cls(self.current_node, function, self._fake_eval_meta_ctor)

            raw_node_diff_args = rule.input_differentiability()
        
        else:
        
            # operator.xxx only appears in non-inplace CALL_FUNCTION Nodes.
            # simply convert it to torch.xxx op.
            if getattr(operator, function.__name__, None) is function:
                function = getattr(torch, function.__name__)

            #
            # Handle using torch.func.jvp and tracing.
            # This is not purely symbolic, must have proper shapes to work with.
            #

            if function not in differentiabilities:
                raise NotImplementedError(
                    f"Differentiation for operator {function} is not registered"
                )
            

            # NOTE torch inplace ops with `out` param can be normalized, but are
            # not supported by torch.jvp.

            # TODO torch.einsum is not FX-norm-able, but should be handled by
            # torch.jvp, as it's expanded to a lot of tensor manipulations
            # and mul/sum calls.

            # keys are strs, values are FX arguments, of RAW Graph
            raw_node_normalized_kwargs: Dict[str, FxArg] = fx_normalize_function_variant_into_kwargs(
                function, self.current_node.args, self.current_node.kwargs
            )  # type: ignore
            dfbs = differentiabilities[function]
            for dfb in dfbs:
                if dfb.all_param_names() == set(raw_node_normalized_kwargs.keys()):
                    break
            else:
                assert False, \
                    "Failed to resolve overloading:" \
                    f" with Differentiabilities {dfbs}, get {raw_node_normalized_kwargs}"
            
            raw_node_diff_args = raw_node_normalized_kwargs.fromkeys(
                dfb.diffable_params
            )  # type: ignore
        
        # endif op category


        # Mandatory context from all categories
        raw_node_diff_args: Dict[str, Union[FxConst, Node, Sequence[Node]]]

        
        # If a diffable param happens to carry tangent
        # -- tangent on non-diffable param is not counted.
        #
        # Also, in this noninplace handler we DO NOT count tangent appearance
        # on inplace target. This is handled by the inplace handler.
        tangent_appear_flag: List[bool] = []
        for raw_node_diff_arg in raw_node_diff_args.values():
            if isinstance(raw_node_diff_arg, (Node, Sequence)):
                arg_flags = collect_meta(
                    raw_node_diff_arg,
                    self.nodemap_raw2tangent.__contains__,
                    leaf_type=Node
                )
                tangent_appear_flag.extend(arg_flags)

        if not any(tangent_appear_flag):
            #
            # No tangent computation for the noninplace part.
            #
            jvp_primal_node = self.jvp_graph.node_copy(
                self.current_node, arg_transform=self.nodemap_raw2primal.__getitem__
            )
            self.nodemap_raw2primal[self.current_node] = jvp_primal_node
            return


        #
        # Prepare primal and tangent Nodes for JVP calculation on this call.
        #

        # input_primal_nodes: Dict[str, Union[FxConst, Node, Sequence[Node]]] = {}
        # input_tangent_nodes: Dict[str, Union[FxConst, Node, Sequence[Node]]] = {}

        # Must be in the same (whatever) order as `raw_node_diff_args`
        input_primal_nodes: List[Union[FxConst, Node, Sequence[Node]]] = []
        input_tangent_nodes: List[Union[FxConst, Node, Sequence[Node]]] = []

        for raw_node_diff_arg in raw_node_diff_args.values():

            if not isinstance(raw_node_diff_arg, (Node, Sequence)):
                # const scalars
                assert isinstance(raw_node_diff_arg, (int, float, str)), \
                    "If a differentiable parameter is not a fx.Node" \
                    " or Node list, it must be a scalar"

                # For scalar primal input to jvp, we **keep** the arg scalar.
                input_primal_nodes.append(raw_node_diff_arg)
                input_tangent_nodes.append(0.)
                # TODO if any JVP rule exploits arg being tensors this won't work.

                # # Zero tangent for scalar
                # #
                # # NOTE it's safer to create replicated zero tensors, because
                # # to generate JVP sub-Graph we use zero tensors (diff_argval)
                # # so the sub-Graph may exploit the assumption that all input
                # # tangents are tensors. And it eases our copying since all
                # # elements are Nodes -- no need to inline scalars.
                # # Although literal `1.0` may also work, computationally.
                # input_tangent_nodes[diff_param] = self.jvp_graph.call_function(
                #     torch.zeros, ([],), { 'dtype': torch.float32 }
                # )
            
            else:  # Node or Node list
                input_primal_nodes.append(tree_map(
                    raw_node_diff_arg, self.nodemap_raw2primal.__getitem__
                ))

                def _prepare_tangent_node(raw: Node) -> Node:
                    assert isinstance(raw, Node)
                    if raw in self.nodemap_raw2tangent:
                        return self.nodemap_raw2tangent[raw]
                    else:
                        # Create zero tangent
                        primal = self.nodemap_raw2primal[raw]
                        return self.jvp_graph.call_function(
                            torch.zeros_like, (primal,)
                        )
                input_tangent_nodes.append(tree_map(
                    raw_node_diff_arg, _prepare_tangent_node
                ))

        #
        # Materialize JVP sub-Graph and copy into JVP full Graph
        #
        if function in tangent_rule_registry:
            rule: DiffRuleBase

            output_primal = self.jvp_graph.node_copy(self.current_node, self.nodemap_raw2primal.__getitem__)

            output_tangent = rule.invoke(output_primal, list(raw_node_diff_args.keys()), input_tangent_nodes)
            assert get_node_meta(self.current_node), \
                "Rule should set metadata on the raw Node"

        
        else:
            raw_node_normalized_kwargs: Dict[str, FxArg]

            gm, y_meta = self._generate_jvp_subgraph_using_torchfunc(
                function, dfb, raw_node_normalized_kwargs
            )
            subg = simplify_torchfunc_fx_graph(gm.graph)
            copier = _TorchJvpSubGraphCopier(
                gm, subg, self.jvp_graph, diff_arg_names, input_primal_nodes, input_tangent_nodes
            ).run()

            set_node_meta(self.current_node, y_meta)

            assert isinstance(copier.output_primal, Node), "no multi-res support yet"
            assert isinstance(copier.output_tangent, Node), "no multi-res support yet"

            self.nodemap_raw2primal[self.current_node] = copier.output_primal
            self.nodemap_raw2tangent[self.current_node] = copier.output_tangent

        # return (copier.output_primal, copier.output_tangent)
    
    def if_call_method(self, method_name: str):
        function = getattr(torch, method_name)
        kwargs = fx_normalize_function_variant_into_kwargs(function, self.current_node.args, self.current_node.kwargs)

        assert False

    def if_call_module(self, submod: Module):
        if isinstance(submod, esr.Module):
            raise NotImplementedError()
            # Nested easier.Module, must be JVP-ed.
            sub_jvp_transformer = JvpTransformer(self.root_jvp, submod).run()


        elif isinstance(submod, esr.Selector):
            input = normalize_selector_call_into_args(*self.current_node.args, **self.current_node.kwargs)

        elif isinstance(submod, esr.Reducer):
            if submod.reduce != 'sum':
                raise NotImplementedError()

            assert 'out' not in self.current_node.kwargs, "TODO"

            input, out = normalize_reducer_call_into_args(*self.current_node.args, **self.current_node.kwargs)
            assert out is None, "TODO"

        else:
            assert False, 'unreachable'

        assert isinstance(input, Node)
        imeta = get_node_meta(input)
        assert isinstance(imeta, RuntimeTensorMeta)
        ometa = self._fake_eval_meta_ctor(imeta.shape, imeta.dtype)
        set_node_meta(self.current_node, ometa)
        
        primal_node = self.jvp_graph.node_copy(self.current_node, self.nodemap_raw2primal.__getitem__)
        self.nodemap_raw2primal[self.current_node] = primal_node

        if input in self.nodemap_raw2tangent:
            tangent_node = self.jvp_graph.node_copy(self.current_node, self.nodemap_raw2tangent.__getitem__)
            self.nodemap_raw2tangent[self.current_node] = tangent_node

    def _generate_jvp_subgraph_using_torchfunc(
        self,
        function: Callable, dfb: Differentiability, raw_normalized_kwargs: Dict[str, FxArg],
    ) -> Tuple[GraphModule, StructuredTensorMeta]:
        """
        We need to note that torch.func.jvp API only leverage the order of
        parameters -- via `diff_arg_names: List[str]`,
        but FX normalization results in `Dict[str, Node]`.

        We need to convert between list and dict.

        An extra conversion is to **flatten** the nested arg e.g. on torch.cat,
        and the identities of the nested elements are only told by their
        positions in the nested structure.
        """
        # Some diffable parameters are given as literals e.g. "add(x, 3)"
        # and we need to convert them to replicated tensors to simplify
        # the generated tangent sub-Graph.
        jvp_diff_primal_vals: \
            Dict[str, Union[torch.Tensor, Sequence[torch.Tensor]]] = {}
        jvp_tangent_vals: \
            Dict[str, Union[torch.Tensor, Sequence[torch.Tensor]]] = {}


        for diff_param in dfb.diffable_params:
            diff_param: str
            raw_node_diff_arg = raw_normalized_kwargs[diff_param]

            # e.g. cat([A, B, C])  -> jvp(lambda A,B,C: cat([A, B, C])
            # There Nodes ABC in the tuple should be captured indivdually
            # into the callable to jvp().
            assert not isinstance(raw_node_diff_arg, Sequence), \
                "Nested input to torch.cat etc. must be bijectively flattened"

            if not isinstance(raw_node_diff_arg, (Node, Sequence)):
                # const scalars
                assert isinstance(raw_node_diff_arg, (int, float, str))
                
                # If diffable, literal integer is treated as float.
                # Take a small-sized float32 dtype.
                # And wrap scalars into []-shape tensors, this dramatically
                # simplifies torch.jvp sub-Graph.
                # Otherwise, jvp sub-Graph will have internal zeros allocation
                # Nodes which are harder for EASIER to handle.
                diff_argval = torch.zeros([], dtype=torch.float32)

                jvp_diff_primal_vals[diff_param] = diff_argval
                jvp_tangent_vals[diff_param] = diff_argval.clone()

            else:  # Node or Node list

                # for a list-typed arg, we assume no mix of Node and scalar.
                jvp_diff_primal_vals[diff_param] = \
                    self._create_zero_val(raw_node_diff_arg)
                jvp_tangent_vals[diff_param] = \
                    self._create_zero_val(raw_node_diff_arg)

        #
        # Non-differentiable parts of raw inputs, primal inputs
        #
        
        # TODO if a non-diffable raw-input Node is captured via closure to jvp,
        # it's represented by a sub-Graph Node e.g. GET_ATTR[tensor_constant0]
        # We need to map it back to primal Nodes.
        # TODO jvp_nondiff_env_nodes = {}

        jvp_nondiff_env_vals: Dict[str, Union[torch.Tensor, FxConst]] = {}
        for nondiff_param, default_arg in dfb.other_params:
            raw_node_nondiff_arg = raw_normalized_kwargs[nondiff_param]

            if isinstance(raw_node_nondiff_arg, Sequence):
                raise NotImplementedError("Nested non-differentiable arg")
                # PyTorch unlikely has this.
            
            if isinstance(raw_node_nondiff_arg, Node):
                raise NotImplementedError("Captured non-differentiable arg")
                jvp_nondiff_val = self._create_zero_val(raw_node_nondiff_arg)
            else:
                assert isinstance(raw_node_nondiff_arg, (int, float, str))
                jvp_nondiff_val = raw_node_nondiff_arg

            # TODO this takes effect even user specifies the value to be
            # explicitly None.
            # It seems ok that no common operators offer a default value
            # (especially when the argument is omitted at callsite)
            # that is not None.
            if jvp_nondiff_val is None:
                if nondiff_param in self.current_node.kwargs:
                    raise NotImplementedError(
                        "User explicitly specifies None arg, may indicate that"
                        " an internal assumption is broken"
                    )
                jvp_nondiff_val = default_arg
            
            jvp_nondiff_env_vals[nondiff_param] = jvp_nondiff_val

        
        # Differentiable Tensor-type parameters, must be passed via jvp()
        # API param list.
        # The names and the zero Tensors must be in the same order.
        diff_arg_names: List[str] = dfb.diffable_params
        diff_primal_vals = tuple(map(jvp_diff_primal_vals.__getitem__, diff_arg_names))
        tangent_vals = tuple(map(jvp_tangent_vals.__getitem__, diff_arg_names))



        # Other non-differentiable parameters must NOT be passed via jvp()
        # API param list, but via function closure.

        def _primal_func(*args: torch.Tensor):
            # There preparations are not part of FX Proxy and won't be traced
            kw = {}
            for diff_arg_name, v in zip(diff_arg_names, args):
                kw[diff_arg_name] = v
            for other_arg_name, v in jvp_nondiff_env_vals.items():
                kw[other_arg_name] = v
            
            # The core operation to let torch.func.jvp to analyze
            return function(**kw)
        
        def _jvp(
            primals: Tuple[torch.Tensor, ...],
            tangents: Tuple[torch.Tensor, ...]
        ):
            return torch.func.jvp(_primal_func, primals, tangents)
        
        # FX Graph, including make_fx, does not respect nested inputs,
        # so we need to manually flatten any nested args and maintain the
        # mapping between before/after flattening.
        gm: GraphModule = make_fx(_jvp)(diff_primal_vals, tangent_vals)
        y, jvp_res = gm(diff_primal_vals, tangent_vals)
        
        ng = get_node_tensor_group(self.current_node)
        if ng is None:
            role = Role.REPLICATED
        else:
            role = Role.DISTRIBUTED

        y_meta = get_value_runtime_info(self.current_node, y, self._fake_eval_meta_ctor)
        jvp_meta = get_value_runtime_info(self.current_node, jvp_res, self._fake_eval_meta_ctor)
        assert y_meta == jvp_meta

        # TODO return bijective position mapping
        return gm, y_meta

class _TorchJvpSubGraphCopier(EasierInterpreter):
    def __init__(
        self, subgm: GraphModule, subg: Graph, jvp_graph: Graph,
        diff_arg_names: List[str],
        jvp_input_primals: Dict[str, Union[FxConst, Node, Sequence[Node]]],
        jvp_input_tangents: Dict[str, Union[Node, Sequence[Node]]]         
    ) -> None:
        super().__init__([subgm], [subg])  # type: ignore

        self.jvp_graph = jvp_graph
        self.diff_arg_names = diff_arg_names
        self.jvp_input_primals = jvp_input_primals
        self.jvp_input_tangents = jvp_input_tangents

        self.nested_arg_bijective_pos = {}

        self.placeholder_i = 0  # totally 2*len(diff_arg_names)

        self.nodemap_subg2jvp: Dict[Node, Union[Node, int, float]] = {}

        self.output_primal: Union[Node, Sequence[Node]]
        self.output_tangent: Union[Node, Sequence[Node]]

        
    def if_placeholder(self, param_name: str):
        """
        Same as torch.func.jvp requirements, the PLACEHOLDER Nodes only differ
        in their positions and the order.
        """
        # param_name would be "primals_1" "tangents_2" (from `def _jvp` above)
        # and not usable.
        is_primal = self.placeholder_i < len(self.diff_arg_names)

        if is_primal:
            pos = self.placeholder_i
            jvp_primal = self.jvp_input_primals[self.diff_arg_names[pos]]
            self.nodemap_subg2jvp[self.current_node] = jvp_primal
        else:
            pos = self.placeholder_i - len(self.diff_arg_names)
            jvp_tangent = self.jvp_input_tangents[self.diff_arg_names[pos]]
            self.nodemap_subg2jvp[self.current_node] = jvp_tangent

        self.placeholder_i += 1
    
    def if_get_attr(self, submod_path: str, attr_name: str, attr_val):
        raise NotImplementedError(
            "Captured values, likely non-differentiable primal inputs"
        )
    
    def if_call_function(self, function):
        jvp_node = self.jvp_graph.node_copy(self.current_node, self.nodemap_subg2jvp.__getitem__)

        self.nodemap_subg2jvp[self.current_node] = jvp_node
    
    def if_output(self):
        """
        TODO When without AD, a multi-res Node stands individually, it's from
        tensor metadata level can we know it's multi-res.

        However, in torch.func.jvp sub-Graph, the primal multi-res Node will
        first be unpacked, then all primal result items are put in OUTPUT
        Node's args[0] list, making it NO LONGER an individual Node.
        When copying the sub-Graph into jvp Graph, we need to handle this.

        TODO Discard raw getitem Nodes and keep getitem Nodes in the sub-Graph.
        """
        subg_primals, subg_tangents = cast(
            Sequence[Union[Node, Sequence[Node]]], self.current_node.args[0]
        )

        convert = self.nodemap_subg2jvp.__getitem__
        self.output_primal = tree_map(subg_primals, convert)
        self.output_tangent = tree_map(subg_tangents, convert)
    
    def if_call_method(self, method_name: str):
        raise NotImplementedError(
            "It's unlikely that torch.func.jvp generates method calls"
        )
    
    def if_call_module(self, submod: Module):
        raise NotImplementedError(
            "It's unlikely that torch.func.jvp generates module calls"
        )


class LazyJvpModule(esr.Module):
    def __init__(self, primal_module: esr.Module, inputs: Sequence[esr.Tensor], outputs: Sequence[esr.Tensor]):
        super().__init__()

        self._check_dup_args(inputs, 'inputs')
        self._check_dup_args(outputs, 'outputs')

        
        # TODO the primal module may also be a placeholder JvpModule
        self.easier_primal_module: esr.Module = primal_module


        # nn.ParamList is not actually a Sequence[Tensor] because it lacks
        # __contains__.
        self.inputs: Sequence[esr.Tensor] = \
            torch.nn.ParameterList(inputs)  # type: ignore
        self.outputs: Sequence[esr.Tensor] = \
            torch.nn.ParameterList(outputs)  # type: ignore


        # TODO how about tangents_in tangents_out?
        self.vectors: Sequence[esr.Tensor] = torch.nn.ParameterList(
            self._args_zerolike(inputs, 'inputs')
                          )  # type: ignore
        self.products: Sequence[esr.Tensor] = torch.nn.ParameterList(
            self._args_zerolike(outputs, 'outputs')
        )  # type: ignore
    
    
    def _check_dup_args(self, args: Sequence[esr.Tensor], param_name: str):
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
    
    def _args_zerolike(self, args: Sequence[esr.Tensor], param_name: str):
        for pos, arg in enumerate(args):
            if not arg.dtype.is_floating_point:
                raise ValueError(
                    f"The {pos}-th easier.Tensor does not have floating-point"
                    f" dtype in {param_name}"
                )

            yield esr.Tensor(
                esr.zeros_like(arg),
                mode=('partition' if arg.is_partition else 'replicate')
            )
    
    def forward(self):
        assert False, "not callable"


def jvp(
    module: esr.Module,
    inputs: Sequence[esr.Tensor],
    outputs: Sequence[esr.Tensor]
) -> LazyJvpModule:
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

    jvpm = LazyJvpModule(module, inputs, outputs)

    # TODO assign meaningful names to:
    # - Jvp.inputs, like `x` if input is InputModule.x
    # - Jvp.vector, like `tan_x`

    return jvpm