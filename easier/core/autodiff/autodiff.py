# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import contextlib
import dataclasses
import itertools
import operator
from types import EllipsisType
from typing import Callable, Collection, Dict, List, Literal, Optional, Self, Sequence, Tuple, TypeAlias, Union, cast
from typing_extensions import OrderedDict

import more_itertools
import torch
from torch.fx import Node, Graph, GraphModule
from torch.fx.node import Argument as FxArg, BaseArgumentTypes as _FxConstBase
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


# e.g. int, float, dtype, device, slice, range, etc.
FxConst: TypeAlias = Union[_FxConstBase, slice, range]



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
        jvp_module: 'Jvp',
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
        if function is operator.getitem:
            container = self.current_node.args[0]
            assert isinstance(container, Node)
            imeta = get_node_meta(container)

            if isinstance(imeta, Sequence):
                # This is the raw Node for unpacking a tuple, has been handled
                # when handling the multi-res raw Node.
                return

        self._handle_operation(function)
        
    def _try_init_tangent_for_inplace_target(self, raw_target: Node):
        if raw_target not in self.nodemap_raw2tangent:
            primal = self.nodemap_raw2primal[raw_target]

            # This also covers later initial write to esr.Tensor. Currently,
            # we have _SingleGetAttrValidator to ensure Node-Tensor 1:1 map.
            tangent_init = self.jvp_graph.call_function(torch.zeros_like, (primal,))

            self.nodemap_raw2tangent[raw_target] = tangent_init
        
        return self.nodemap_raw2tangent[raw_target]


    def _try_init_tangent_for_inplace_op(self, function):
        if function is operator.setitem:
            target, index, input = self.current_node.args
        elif function.__name__.endswith('_'):
            if len(self.current_node.args) > 0:
                target = self.current_node.args[0]
            else:
                target = self.current_node.kwargs['input']
        else:
            return
        
        assert isinstance(target, Node)
        self._try_init_tangent_for_inplace_target(target)


    def _is_tangent_involved(self, raw_node_diff_args: Dict[str, Union[FxConst, Node, Sequence[Node]]]) -> bool:
        """
        By op category:

        -   For operators that EASIER manually handles, DiffRule should parse
            the raw Node and tell what raw ARGUMENTs are differentiable.
            It's recorded in `raw_node_diff_args`.
            
            -   If the op is not differentiable, DiffRule unconditionally
                returns empty dict.
        
        -   For auto-generated operators that are handled by torch.jvp,
            the Differentiability record tells what PARAMETERS are statically
            differentiable, and _handle_operation merge the static info and
            actual info in the raw Node, and generate `raw_node_diff_args`.

        Remarks:

        -   Tangent on non-diffable param is not counted.
        """
        tangent_appear_flag: List[bool] = []
        for raw_node_diff_arg in raw_node_diff_args.values():
            if isinstance(raw_node_diff_arg, (Node, Sequence)):
                arg_flags = collect_meta(
                    raw_node_diff_arg,
                    self.nodemap_raw2tangent.__contains__,
                    leaf_type=Node
                )
                tangent_appear_flag.extend(arg_flags)
        
        return any(tangent_appear_flag)
    
    def _prepare_diffable_primals_and_tangents(self, raw_node_diff_args: Dict[str, Union[FxConst, Node, Sequence[Node]]]):

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

        return input_primal_nodes, input_tangent_nodes
        
    def _handle_operation(self, function: Callable):
        """
        Handle the raw Node:
        -   copy the raw Node to the primal Node into jvp Graph;
        -   if the op is differentiable AND tangent flow appears on raw Node,
            inject jvp sub-Graph.


        An important extra property for inplace ops is:
        the target Node, esr.Tensor instance or immediately result, may not
        be bound with tangent initially.
        It's at this inplace Node does the target starts to have tangent.
        """
        #
        # Op category
        #
        if function in tangent_rule_registry:
            rule_cls = tangent_rule_registry[function]
            rule = rule_cls(self.current_node, function, self._fake_eval_meta_ctor)

            #
            # primal
            #
            output_primal = self.jvp_graph.node_copy(self.current_node, self.nodemap_raw2primal.__getitem__)

            out_meta = rule.invoke_output_meta()
            set_node_meta(self.current_node, out_meta)

            # TODO multi-res primal Node has this, but there is no strictly
            # a multi-res tanget Node
            self.nodemap_raw2primal[self.current_node] = output_primal

            if isinstance(out_meta, Sequence):  # multi-res Node
                for raw_getitem in self.current_node.users:
                    assert raw_getitem.target is operator.getitem
                    _, item_i = raw_getitem.args
                    assert isinstance(item_i, int)

                    primal_getitem = self.jvp_graph.node_copy(raw_getitem, self.nodemap_raw2primal.__getitem__)

                    item_meta = out_meta[item_i]
                    set_node_meta(raw_getitem, item_meta)

                    self.nodemap_raw2primal[raw_getitem] = primal_getitem 

            # TODO invoke_jvp may insert getitem Node again

            #
            # tangent
            #
            raw_node_diff_args = rule.input_differentiability(*self.current_node.args, **self.current_node.kwargs)
            tangent_involved = self._is_tangent_involved(raw_node_diff_args)

            if tangent_involved:

                # Initialize tangent after checking tangent_involved -- because
                # inplace target may not have a tangent, we need to prepare a
                # zero tensor and bind it in the raw2tangent map.
                self._try_init_tangent_for_inplace_op(function)
                    
                _, input_tangent_nodes = self._prepare_diffable_primals_and_tangents(raw_node_diff_args)

                # Rule is only charge of generating tangent calculations
                output_tangent = rule.inject_jvp_subgraph(
                    output_primal,
                    list(raw_node_diff_args.keys()),
                    self.nodemap_raw2primal,
                    input_tangent_nodes
                )

                if isinstance(out_meta, RuntimeTensorMeta):
                    assert isinstance(output_tangent, Node)
                    self.nodemap_raw2tangent[self.current_node] = output_tangent

                else:
                    assert isinstance(output_tangent, Sequence)
                    # May have None item for non-diff-able items, say
                    # resultant indices in sort()
                    #
                    # The multi-res raw Node does not have tangent Node,
                    # but its unpacking getitem Nodes have.

                    for raw_getitem in self.current_node.users:
                        assert raw_getitem.target is operator.getitem
                        _, item_i = raw_getitem.args
                        assert isinstance(item_i, int)

                        tangent_item = output_tangent[item_i]
                        if tangent_item is not None:
                            # The unpacking getitem Nodes are required to be
                            # generated by DiffRule.jvp() -- this is the most
                            # general case, a multi-res tangent Node may not
                            # always be available.
                            self.nodemap_raw2tangent[raw_getitem] = output_tangent[item_i]
                # endif output is multi-res
            # endif tangent_involved

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
            
            raw_node_diff_args: Dict[
                str, Union[FxConst, Node, Sequence[Node]]
            ] = {
                p: raw_node_normalized_kwargs[p] for p in dfb.diffable_params
            }  # type: ignore

            tangent_involved = self._is_tangent_involved(raw_node_diff_args)

            # Unconditionally invoke torch.jvp to generate primal+jvp subgraph
            # NOTE this op may be a undifferentiable op, but we still need its
            # primal part
            gm, out_meta, flatten_tree = self._generate_jvp_subgraph_using_torchfunc(
                function, dfb, raw_node_normalized_kwargs
            )
            set_node_meta(self.current_node, out_meta)

            if not tangent_involved:
                # maybe:
                # - the op is not differentiable at all;
                # - no input differentiable arguments are with tangents
                # then copy the primal Node only.
                output_primal = self.jvp_graph.node_copy(self.current_node, self.nodemap_raw2primal.__getitem__)
                self.nodemap_raw2primal[self.current_node] = output_primal
            
            else:
                self._try_init_tangent_for_inplace_op(function)

                # Unflattened structure
                input_primal_nodes, input_tangent_nodes = self._prepare_diffable_primals_and_tangents(raw_node_diff_args)

                # copy the whole torch.func.jvp sub-Graph, it includes both
                # the primal part and the tangent part
                subg = simplify_torchfunc_fx_graph(gm.graph)
                copier = _TorchJvpSubGraphCopier(
                    gm, subg, self.jvp_graph, flatten_tree, input_primal_nodes, input_tangent_nodes
                ).run()

                # torch.jvp sub-Graph is likely to result in explicit unpacking
                # getitem Nodes on both primal and tangent.

                out_diff = dfb.output_differentiability

                if isinstance(out_diff, bool):
                    assert isinstance(copier.output_primal, Node)
                    assert isinstance(copier.output_tangent, Node)
                    assert out_diff == True
                    self.nodemap_raw2primal[self.current_node] = copier.output_primal
                    self.nodemap_raw2tangent[self.current_node] = copier.output_tangent
                
                else:
                    assert isinstance(copier.output_primal, Sequence)
                    assert isinstance(copier.output_tangent, Sequence)
                    assert isinstance(out_diff, Sequence)

                    # NOTE it's likely the RAW multi-res Node doesn't have
                    # explicit primal/tangent counterpart Nodes, so we can only
                    # set up binding on the unpacking getitem Nodes.

                    for raw_getitem in self.current_node.users:
                        assert raw_getitem.target is operator.getitem
                        _, item_i = raw_getitem.args
                        assert isinstance(item_i, int)

                        self.nodemap_raw2primal[raw_getitem] = copier.output_primal[item_i]

                        # If an output item is not differentiable, torch.jvp
                        # generates zero tangent. We don't bind it.
                        if out_diff[item_i]:
                            self.nodemap_raw2tangent[raw_getitem] = copier.output_tangent[item_i]
                # endif out_diff
            # endif tangent_involved
        
        # endif op category

    
    def if_call_method(self, method_name: str):
        function = getattr(torch, method_name)
        self._handle_operation(function)

    def if_call_module(self, submod: Module):
        if isinstance(submod, esr.Module):
            raise NotImplementedError()
            # Nested easier.Module, must be JVP-ed.
            # sub_jvp_transformer = JvpTransformer(self.root_jvp, submod).run()

        # TODO make Selector/Reducer rules.

        elif isinstance(submod, esr.Selector):
            input = normalize_selector_call_into_args(*self.current_node.args, **self.current_node.kwargs)
            assert isinstance(input, Node)

            # Copy primal
            primal_node = self.jvp_graph.node_copy(self.current_node, self.nodemap_raw2primal.__getitem__)
            self.nodemap_raw2primal[self.current_node] = primal_node
            
            # NOTE the fake batch size is not affected by S.idx.shape[0]
            set_node_meta(self.current_node, get_node_meta(input))

            # Inject JVP if tangent appears
            if input in self.nodemap_raw2tangent:
                tangent_node = self.jvp_graph.node_copy(self.current_node, self.nodemap_raw2tangent.__getitem__)
                self.nodemap_raw2tangent[self.current_node] = tangent_node

        elif isinstance(submod, esr.Reducer):
            if submod.reduce != 'sum':
                raise NotImplementedError()

            input, out = normalize_reducer_call_into_args(*self.current_node.args, **self.current_node.kwargs)
            assert isinstance(input, Node)
            assert out is None or isinstance(out, Node)

            # Copy primal
            primal_node = self.jvp_graph.node_copy(self.current_node, self.nodemap_raw2primal.__getitem__)
            self.nodemap_raw2primal[self.current_node] = primal_node

            # NOTE the fake batch size is not affected by S.idx.shape[0]
            set_node_meta(self.current_node, get_node_meta(input))

            # Inject JVP if tangent appears
            if input in self.nodemap_raw2tangent or out in self.nodemap_raw2tangent:

                if out is not None:
                    self._try_init_tangent_for_inplace_target(out)

                tangent_node = self.jvp_graph.node_copy(self.current_node, self.nodemap_raw2tangent.__getitem__)
                self.nodemap_raw2tangent[self.current_node] = tangent_node


        else:
            assert False, 'unreachable'



    def _generate_jvp_subgraph_using_torchfunc(
        self,
        function: Callable, dfb: Differentiability, raw_normalized_kwargs: Dict[str, FxArg],
    ) -> Tuple[GraphModule, StructuredTensorMeta, List[List[int]]]:
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
        diff_primal_vals = list(map(jvp_diff_primal_vals.__getitem__, diff_arg_names))
        tangent_vals = list(map(jvp_tangent_vals.__getitem__, diff_arg_names))

        def _flatten(vals: List[Union[torch.Tensor, Sequence[torch.Tensor]]]):
            def _flatten_pos(
                i_val: Tuple[int, Union[torch.Tensor, Sequence[torch.Tensor]]]
            ) -> List[Tuple[List[int], torch.Tensor]]:
                i, val = i_val
                if isinstance(val, torch.Tensor):
                    return [([i], val)]
                else:
                    i_ii_vals = []
                    for ii, item in enumerate(val):
                        i_ii_vals.append(([i, ii], item))
                    return i_ii_vals

            i_ii_vals = list(
                more_itertools.flatten(map(_flatten_pos, enumerate(vals)))
            )
            positions = list(map(lambda tp: tp[0], i_ii_vals))
            items = tuple(map(lambda tp: tp[1], i_ii_vals))
            return positions, items
        
        def _unflatten(positions: List[List[int]], items: Sequence[torch.Tensor]):
            unfltd_primals: List[Union[torch.Tensor, List[torch.Tensor]]] = []
            for pos, val in zip(positions, items):
                if len(pos) == 1:
                    [i] = pos
                    # single-elem pos is always not nested
                    unfltd_primals.append(val)

                else:
                    [i, ii] = pos
                    # double-elem pos is always nested, and the inner list
                    # may have only 1 item.
                    if ii == 0:
                        unfltd_primals.append([])
                    unfltd_primals[i].append(val)  # type: ignore
            
            return unfltd_primals

            

        flattened_primal_tree, flattened_primal_vals = _flatten(diff_primal_vals)
        flattened_tangent_tree, flattened_tangent_vals = _flatten(tangent_vals)
        assert flattened_primal_tree == flattened_tangent_tree

        # Other non-differentiable parameters must NOT be passed via jvp()
        # API param list, but via function closure.

        def _primal_func(*flattened_primals: torch.Tensor):
            # There preparations are not part of FX Proxy and won't be traced
            unfltd_primals = _unflatten(flattened_primal_tree, flattened_primals)

            kw = {}
            for diff_arg_name, v in zip(diff_arg_names, unfltd_primals):
                kw[diff_arg_name] = v
            for other_arg_name, v in jvp_nondiff_env_vals.items():
                kw[other_arg_name] = v
            
            # The core operation to let torch.func.jvp to analyze
            return function(**kw)
        
        def _jvp(
            flattened_primals: Tuple[torch.Tensor, ...],
            flattened_tangents: Tuple[torch.Tensor, ...]
        ):
            return torch.func.jvp(
                _primal_func, flattened_primals, flattened_tangents
            )
        
        # FX Graph, including make_fx, does not respect nested inputs,
        # so we need to manually flatten any nested args and maintain the
        # mapping between before/after flattening.
        gm: GraphModule = make_fx(_jvp)(flattened_primal_vals, flattened_tangent_vals)
        y, jvp_res = gm(flattened_primal_vals, flattened_tangent_vals)

        assert isinstance(y, torch.Tensor), "TODO multi-res"

        ng = get_node_tensor_group(self.current_node)
        if ng is None:
            role = Role.REPLICATED
        else:
            role = Role.DISTRIBUTED

        y_meta = get_value_runtime_info(self.current_node, y, self._fake_eval_meta_ctor)
        jvp_meta = get_value_runtime_info(self.current_node, jvp_res, self._fake_eval_meta_ctor)
        assert y_meta == jvp_meta

        # TODO return bijective position mapping
        return gm, y_meta, flattened_primal_tree

class _TorchJvpSubGraphCopier(EasierInterpreter):
    def __init__(
        self, subgm: GraphModule, subg: Graph, jvp_graph: Graph,
        flatten_tree: List[List[int]],
        jvp_input_primals: List[Union[FxConst, Node, Sequence[Node]]],
        jvp_input_tangents: List[Union[FxConst, Node, Sequence[Node]]],
    ) -> None:
        super().__init__([subgm], [subg])  # type: ignore

        self.jvp_graph = jvp_graph
        # self.diff_arg_names = diff_arg_names
        self.jvp_input_primals = jvp_input_primals
        self.jvp_input_tangents = jvp_input_tangents
        assert len(jvp_input_primals) == len(jvp_input_tangents)

        self.flatten_tree = flatten_tree

        self.nested_arg_bijective_pos = {}

        self.placeholder_i = 0  # totally 2*len(diff_arg_names)

        # TODO as JvpTransformer._prepare_diffable_primals_and_tangents, we
        # allow constants, but the subgraph assumes all inputs are Nodes,
        # making it erroneous to call torch function/method on constants.
        self.nodemap_subg2jvp: Dict[Node, Union[Node, FxConst]] = {}

        self.output_primal: Union[Node, Sequence[Node]]
        self.output_tangent: Union[Node, Sequence[Node]]

        
    def if_placeholder(self, param_name: str):
        """
        Same as torch.func.jvp requirements, the PLACEHOLDER Nodes only differ
        in their positions and the order.

        All placeholders are formed by first flattening the args of the raw
        Node, e.g. torch.cat([A,B,C]) will have 3 placeholders A, B, C.
        """
        # param_name would be "primals_1" "tangents_2" (from `def _jvp` above)
        # and not usable.
        is_primal = self.placeholder_i < len(self.jvp_input_primals)

        def _from_flatten(jvp_inputs: List[Union[FxConst, Node, Sequence[Node]]], tree: List[int]) -> Union[Node, FxConst]:
            if len(tree) == 1:
                [i] = tree
                return jvp_inputs[i]  # type: ignore
            else:
                [i, ii] = tree
                return jvp_inputs[i][ii] # type: ignore

        if is_primal:
            ph_pos = self.placeholder_i

            jvp_primal = _from_flatten(self.jvp_input_primals, self.flatten_tree[ph_pos])
            self.nodemap_subg2jvp[self.current_node] = jvp_primal
        else:
            ph_pos = self.placeholder_i - len(self.jvp_input_primals)

            jvp_tangent = _from_flatten(self.jvp_input_tangents, self.flatten_tree[ph_pos])
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


class Jvp(esr.Module):
    def __init__(self, inputs: Sequence[esr.Tensor], outputs: Sequence[esr.Tensor]):
        super().__init__()

        self._check_dup_args(inputs, 'inputs')
        self._check_dup_args(outputs, 'outputs')


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

    
    def get_initial_tangent_tensors_map(self) -> Dict[esr.Tensor, esr.Tensor]:
        mapping = {}
        for i, v in zip(self.inputs, self.vectors):
            mapping[i] = v
        for o, p in zip(self.outputs, self.products):
            mapping[o] = p
        return mapping
    
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


def jvp(
    module: esr.Module,
    inputs: Sequence[esr.Tensor],
    outputs: Sequence[esr.Tensor]
) -> Tuple[Jvp, Sequence[esr.Tensor], Sequence[esr.Tensor]]:
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

    # Create a local class for each jvp call.
    class _Jvp(Jvp):
        pass

    jvpm = _Jvp(inputs, outputs)
    jvp_transformer = JvpTransformer(
        module, jvpm, jvpm.get_initial_tangent_tensors_map()
    ).run()

    jvpm.forward = GraphModule(jvpm, jvp_transformer.jvp_graph).forward

    # TODO assign meaningful names to:
    # - Jvp.inputs, like `x` if input is InputModule.x
    # - Jvp.vector, like `tan_x`

    return jvpm, jvpm.vectors, jvpm.products