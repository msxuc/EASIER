# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import operator
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import more_itertools
import torch
from torch.fx import Node, Graph, GraphModule
from torch.fx.node import Argument as FxArg
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.modules import Module

import easier.core.module as esr
from easier.core.passes.collective_initialization import \
    collectively_initialize_and_validate
from easier.core.passes.tensor_grouping import \
    group_tensors, get_node_tensor_group
from easier.core.passes.utils import \
    EasierInterpreter, SubmodNameAllocator, normalize_reducer_call_into_args, \
    fx_normalize_function_variant_into_kwargs, tree_map, \
    normalize_selector_call_into_args

from easier.core.runtime.metadata import \
    Role, RuntimeTensorMeta, StructuredTensorMeta, \
    collect_meta, set_node_meta, get_node_meta
    
from easier.core.autodiff.autodiff_rule import \
    Differentiability, RequiredParam, \
    tangent_rule_registry, differentiabilities
from easier.core.autodiff.utils import simplify_torchfunc_fx_graph, FxConst



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
    ):
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

        self.jvp_graph = Graph()
        self.nodemap_raw2primal: Dict[Node, Node] = {}
        self.nodemap_raw2tangent: Dict[Node, Node] = {}

        self.tensormap_primal2tangent: Dict[esr.Tensor, esr.Tensor] = {}
        
        # Primal inputs/outputs, tangent inputs/outputs have been setattr-ed
        # to Jvp Module in its `.inputs/.outputs/.vectors/.products`
        # torch.nn.ParamList fields, we need to rebind the attr paths.
        #
        # NOTE currently if an esr.Tensor/GET_ATTR-Node gets written with
        # tangent, we only 1) ensure its GET_ATTR Node is unique; 2) inject a
        # zero_like Node for GET_ATTR Node.
        # I.e. we don't allocate an esr.Tensor for such immediate tangents.
        self.tensors_attrpaths: Dict[esr.Tensor, str] = {}

        for i, (input, vector) in enumerate(zip(
            jvp_module.inputs, jvp_module.vectors
        )):
            self.tensormap_primal2tangent[input] = vector

            self.tensors_attrpaths[input] = f'inputs.{i}'
            self.tensors_attrpaths[vector] = f'vectors.{i}'


        for i, (output, product) in enumerate(zip(
            jvp_module.outputs, jvp_module.products
        )):
            self.tensormap_primal2tangent[output] = product

            self.tensors_attrpaths[output] = f'outputs.{i}'
            self.tensors_attrpaths[product] = f'products.{i}'


        # There are still esr.Tensors not in jvp.inputs/outputs, but also
        # primal, we allocate names and bind them lazily.
        self.primal_name_allocator = SubmodNameAllocator('primal')

        # Attr name for Selector/Reducer to setattr on jvp
        self.primitive_name_allocator = SubmodNameAllocator('esrprim')


    def _ensure_jvp_attr_obj(
        self, raw_obj: Union[esr.Tensor, esr.Selector, esr.Reducer],
        name_allocator: SubmodNameAllocator,
        attrname_hint: str
    ) -> str:
        if raw_obj in self.tensors_attrpaths:
            return self.tensors_attrpaths[raw_obj]  # type: ignore

        # Lazily bound non-IO primal tensors, or S/R.
        # These attributes will be directly in the Jvp Module's fields.
        for primal_attrname, primal_obj in self.jvp_module.__dict__.items():
            if primal_obj is raw_obj:
                break
        else:
            primal_attrname = name_allocator.alloc_name(
                self.jvp_module, attrname_hint
            )
            setattr(self.jvp_module, primal_attrname, raw_obj)

        return primal_attrname

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
        # Avoid circle import
        from easier.core.runtime.jit_engine.jit_engine import \
            get_value_runtime_info

        runtime_meta = get_value_runtime_info(
            self.current_node, attr_val, self._fake_eval_meta_ctor
        )
        set_node_meta(self.current_node, runtime_meta)

        primal_attrname = self._ensure_jvp_attr_obj(
            attr_val, self.primal_name_allocator, attr_name
        )
        primal_node = self.jvp_graph.get_attr(primal_attrname)

        self.nodemap_raw2primal[self.current_node] = primal_node

        if attr_val in self.tensormap_primal2tangent:
            # Simplified: it's unlikely the same primal tensor has GET_ATTR
            # Nodes multiple times, therefore it's OK to inject GET_ATTR Nodes
            # for tangent each time.
            tangent_tensor = self.tensormap_primal2tangent[attr_val]

            # Inject tangent node
            tangent_attrname = self.tensors_attrpaths[tangent_tensor]
            tangent_node = self.jvp_graph.get_attr(tangent_attrname)

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
            tangent_init = self.jvp_graph.call_function(
                torch.zeros_like, (primal,)
            )

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

    def _is_tangent_involved(
        self,
        raw_node_diff_args: Dict[str, Union[FxConst, Node, Sequence[Node]]]
    ) -> bool:
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

        -   Not necessarily every diffable param has tangent.
        """
        involved = False
        for raw_node_diff_arg in raw_node_diff_args.values():
            if isinstance(raw_node_diff_arg, (Node, Sequence)):
                raws_tangents = collect_meta(
                    raw_node_diff_arg,
                    lambda raw: (raw, self.nodemap_raw2tangent.get(raw, None)),
                    leaf_type=Node
                )
        
                for raw_in, tangent in raws_tangents:
                    if tangent is not None:
                        involved = True

                        imeta = cast(RuntimeTensorMeta, get_node_meta(raw_in))
                        assert imeta.dtype.is_floating_point, \
                            "Tangent carrier must have floating-point dtype"
        
        return involved
    
    def _prepare_diffable_primals_and_tangents(
        self,
        raw_node_diff_args: Dict[str, Union[FxConst, Node, Sequence[Node]]]
    ):
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
                # TODO if any JVP rule exploits arg being tensors this won't
                # work.

                # # Zero tangent for scalar
                # #
                # # NOTE it's safer to create replicated zero tensors, because
                # # to generate JVP sub-Graph we use zero tensors (diff_argval)
                # # so the sub-Graph may exploit the assumption that all input
                # # tangents are tensors. And it eases our copying since all
                # # elements are Nodes -- no need to inline scalars.
                # # Although literal `1.0` may also work, computationally.
                # input_tangent_nodes.append(self.jvp_graph.call_function(
                #     torch.zeros, ([],), { 'dtype': torch.float32 }
                # ))
            
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
    
    def _prepare_constant_tensor_for_operator(self):
        enforced_dtype = None

        for arg in self.current_node.args:
            if arg in self.nodemap_raw2tangent:
                # Given this is an syntactic operator, if any operand
                # has tangent, other constants must be of floating-point dtype
                # to be handled by torch.func.jvp().
                enforced_dtype = torch.float32

        for arg_i, const in enumerate(list(self.current_node.args)):
            if isinstance(const, (int, float)):
                with self.current_graph.inserting_before(self.current_node):
                    raw_const_node = self.current_graph.call_function(
                        torch.full, ((), const), {'dtype': enforced_dtype}
                    )
                    self.current_node.update_arg(arg_i, raw_const_node)
                    set_node_meta(
                        raw_const_node,
                        RuntimeTensorMeta(
                            Role.REPLICATED,
                            (),
                            torch.full((), const, dtype=enforced_dtype).dtype
                        )
                    )
                
                self.nodemap_raw2primal[raw_const_node] = \
                    self.jvp_graph.call_function(
                        torch.full, ((), const), {'dtype': enforced_dtype}
                    )
        
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
            rule = rule_cls(
                self.current_node, function, self._fake_eval_meta_ctor
            )

            #
            # primal
            #
            output_primal = self.jvp_graph.node_copy(
                self.current_node, self.nodemap_raw2primal.__getitem__
            )

            out_meta = rule.invoke_output_meta()
            set_node_meta(self.current_node, out_meta)

            # TODO multi-res primal Node has this, but there may not be
            # a multi-res tanget Node strictly paired with this, e.g.
            # the 2nd result item of torch.sort() is not diff-able
            self.nodemap_raw2primal[self.current_node] = output_primal


            if isinstance(out_meta, Sequence):  # multi-res Node
                rule_jvp_result = []

                for user_i, raw_getitem in enumerate(self.current_node.users):
                    assert raw_getitem.target is operator.getitem
                    _, item_i = raw_getitem.args
                    assert isinstance(item_i, int)
                    assert item_i == user_i

                    primal_getitem = self.jvp_graph.node_copy(
                        raw_getitem, self.nodemap_raw2primal.__getitem__
                    )

                    item_meta = out_meta[item_i]
                    set_node_meta(raw_getitem, item_meta)

                    self.nodemap_raw2primal[raw_getitem] = primal_getitem 

                    rule_jvp_result.append(primal_getitem)
                
                rule_jvp_result = tuple(rule_jvp_result)
            else:
                rule_jvp_result = output_primal

            # TODO invoke_jvp may insert getitem Node again

            #
            # tangent
            #
            raw_node_diff_args = rule.input_differentiability(
                *self.current_node.args, **self.current_node.kwargs
            )
            tangent_involved = self._is_tangent_involved(raw_node_diff_args)

            if tangent_involved:

                # Initialize tangent after checking tangent_involved -- because
                # inplace target may not have a tangent, we need to prepare a
                # zero tensor and bind it in the raw2tangent map.
                self._try_init_tangent_for_inplace_op(function)
                    
                _, input_tangent_nodes = \
                    self._prepare_diffable_primals_and_tangents(
                        raw_node_diff_args
                    )

                # Rule is only charge of generating tangent calculations
                output_tangent = rule.inject_jvp_subgraph(
                    rule_jvp_result,
                    list(raw_node_diff_args.keys()),
                    self.nodemap_raw2primal,
                    input_tangent_nodes
                )

                if isinstance(out_meta, RuntimeTensorMeta):
                    assert isinstance(output_tangent, Node)
                    self.nodemap_raw2tangent[self.current_node] = \
                        output_tangent

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
                            self.nodemap_raw2tangent[raw_getitem] = \
                                output_tangent[item_i]
                # endif output is multi-res
            # endif tangent_involved

        else:
        
            # operator.xxx only appears in non-inplace CALL_FUNCTION Nodes.
            # simply convert it to torch.xxx op.
            if getattr(operator, function.__name__, None) is function:
                if function is operator.truediv:
                    function = torch.div  # torch does not have truediv
                else:
                    function = getattr(torch, function.__name__)

                self._prepare_constant_tensor_for_operator()

            #
            # Handle using torch.func.jvp and tracing.
            # This is not purely symbolic, must have proper shapes to work with
            #

            if function not in differentiabilities:
                raise NotImplementedError(
                    f"Operator {function} is not registered in EASIER AutoDiff"
                )
            

            # NOTE torch inplace ops with `out` param can be normalized, but are
            # not supported by torch.jvp.

            # TODO torch.einsum is not FX-norm-able, but should be handled by
            # torch.jvp, as it's expanded to a lot of tensor manipulations
            # and mul/sum calls.

            # keys are strs, values are FX arguments, of RAW Graph
            raw_node_normalized_kwargs: Dict[str, FxArg] = \
                fx_normalize_function_variant_into_kwargs(
                    function, self.current_node.args, self.current_node.kwargs
                )  # type: ignore
            dfbs = differentiabilities[function]
            for dfb in dfbs:
                if dfb.all_param_names() == set(
                    raw_node_normalized_kwargs.keys()
                ):
                    break
            else:
                assert False, \
                    "Failed to resolve overloading:" \
                    f" with Differentiabilities {dfbs}," \
                    f" get {raw_node_normalized_kwargs}"
            
            raw_node_diff_args: Dict[
                str, Union[FxConst, Node, Sequence[Node]]
            ] = {
                p: raw_node_normalized_kwargs[p] for p in dfb.diffable_params
            }  # type: ignore

            tangent_involved = self._is_tangent_involved(raw_node_diff_args)


            if not tangent_involved:
                kwvals = {
                    k: tree_map(
                        v, self._create_zero_val
                    ) if isinstance(v, Node) else v
                    for k, v in raw_node_normalized_kwargs.items()
                }
                fake_res = function(**kwvals)
                
                from easier.core.runtime.jit_engine.jit_engine import \
                    get_value_runtime_info
                out_meta = get_value_runtime_info(
                    self.current_node, fake_res, self._fake_eval_meta_ctor
                )
                set_node_meta(self.current_node, out_meta)

                # maybe:
                # - the op is not differentiable at all;
                # - no input differentiable arguments are with tangents
                # then copy the primal Node only.
                output_primal = self.jvp_graph.node_copy(
                    self.current_node, self.nodemap_raw2primal.__getitem__
                )
                self.nodemap_raw2primal[self.current_node] = output_primal
            
            else:
                # Unconditionally invoke torch.jvp to generate primal+jvp subgraph
                # NOTE this op may be a undifferentiable op, but we still need its
                # primal part
                gm, out_meta, flatten_tree, nondiff_inputs = \
                    self._generate_jvp_subgraph_using_torchfunc(
                        function, dfb, raw_node_normalized_kwargs
                    )
                set_node_meta(self.current_node, out_meta)

                self._try_init_tangent_for_inplace_op(function)

                # Unflattened structure
                input_primal_nodes, input_tangent_nodes = \
                    self._prepare_diffable_primals_and_tangents(
                        raw_node_diff_args
                    )

                out_diff = dfb.output_differentiability

                n_flattened_primal_outputs = 1
                if isinstance(out_diff, Sequence):
                    n_flattened_primal_outputs = len(out_diff)

                # copy the whole torch.func.jvp sub-Graph, it includes both
                # the primal part and the tangent part
                subg = simplify_torchfunc_fx_graph(gm)
                copier = _TorchJvpSubGraphCopier(
                    gm, subg, self.jvp_graph,
                    flatten_tree, input_primal_nodes, input_tangent_nodes,
                    n_flattened_primal_outputs,
                    nondiff_inputs
                ).run()

                # torch.jvp sub-Graph is likely to result in explicit unpacking
                # getitem Nodes on both primal and tangent.


                if isinstance(out_diff, bool):
                    assert isinstance(copier.output_primal, Node)
                    assert isinstance(copier.output_tangent, Node)
                    assert out_diff == True
                    self.nodemap_raw2primal[self.current_node] = \
                        copier.output_primal
                    self.nodemap_raw2tangent[self.current_node] = \
                        copier.output_tangent
                
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

                        self.nodemap_raw2primal[raw_getitem] = \
                            copier.output_primal[item_i]

                        # If an output item is not differentiable, torch.jvp
                        # generates zero tangent. We don't bind it.
                        if out_diff[item_i]:
                            self.nodemap_raw2tangent[raw_getitem] = \
                                copier.output_tangent[item_i]
                # endif out_diff
            # endif tangent_involved
        
        # endif op category

    
    def if_call_method(self, method_name: str):
        function = getattr(torch.ops.aten, method_name)
        self._handle_operation(function)

    def if_call_module(self, submod: Module):

        jvpmod_submod_attrname = self._ensure_jvp_attr_obj(
            submod,  # type: ignore
            self.primitive_name_allocator,
            cast(str, self.current_node.target)
        )

        if isinstance(submod, esr.Module):
            raise NotImplementedError()
            # Nested easier.Module, must be JVP-ed.
            # sub_jvp_transformer = JvpTransformer(self.root_jvp, submod).run()

        # TODO make Selector/Reducer rules.

        elif isinstance(submod, esr.Selector):
            input = normalize_selector_call_into_args(
                *self.current_node.args, **self.current_node.kwargs
            )
            assert isinstance(input, Node)

            # Copy primal
            primal_node = self.jvp_graph.node_copy(
                self.current_node, self.nodemap_raw2primal.__getitem__
            )
            primal_node.target = jvpmod_submod_attrname

            self.nodemap_raw2primal[self.current_node] = primal_node
            
            # NOTE the fake batch size is not affected by S.idx.shape[0]
            set_node_meta(self.current_node, get_node_meta(input))

            # Inject JVP if tangent appears
            if input in self.nodemap_raw2tangent:
                tangent_node = self.jvp_graph.node_copy(
                    self.current_node, self.nodemap_raw2tangent.__getitem__
                )
                tangent_node.target = jvpmod_submod_attrname

                self.nodemap_raw2tangent[self.current_node] = tangent_node

        elif isinstance(submod, esr.Reducer):
            if submod.reduce != 'sum':
                raise NotImplementedError()

            input, out = normalize_reducer_call_into_args(
                *self.current_node.args, **self.current_node.kwargs
            )
            assert isinstance(input, Node)
            assert out is None or isinstance(out, Node)

            # Copy primal
            primal_node = self.jvp_graph.node_copy(
                self.current_node, self.nodemap_raw2primal.__getitem__
            )
            primal_node.target = jvpmod_submod_attrname

            self.nodemap_raw2primal[self.current_node] = primal_node

            # NOTE the fake batch size is not affected by S.idx.shape[0]
            set_node_meta(self.current_node, get_node_meta(input))

            # Inject JVP if tangent appears
            if input in self.nodemap_raw2tangent \
            or out in self.nodemap_raw2tangent:

                if out is not None:
                    self._try_init_tangent_for_inplace_target(out)

                tangent_node = self.jvp_graph.node_copy(
                    self.current_node, self.nodemap_raw2tangent.__getitem__
                )
                tangent_node.target = jvpmod_submod_attrname

                self.nodemap_raw2tangent[self.current_node] = tangent_node


        else:
            assert False, 'unreachable'



    def _generate_jvp_subgraph_using_torchfunc(
        self,
        function: Callable,
        dfb: Differentiability,
        raw_normalized_kwargs: Dict[str, FxArg],
    ) -> Tuple[
        GraphModule,
        StructuredTensorMeta,
        List[List[int]]
    ]:
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
                    self._create_zero_val(raw_node_diff_arg)  # type: ignore
                jvp_tangent_vals[diff_param] = \
                    self._create_zero_val(raw_node_diff_arg)  # type: ignore

        #
        # Non-differentiable parts of raw inputs, primal inputs
        #
        
        # If a non-diffable raw-input Node is captured via closure to jvp,
        # it's represented by a sub-Graph Node e.g. GET_ATTR[tensor_constant0]
        # We need to map it back to primal Nodes.
        #
        # The order should be strictly maintained by jvp-resultant
        # GraphModule's _tensor_constantsN fields.
        nondiff_val2node: Dict[torch.Tensor, Node] = {}

        jvp_nondiff_env_vals: Dict[str, Union[torch.Tensor, FxConst]] = {}
        for nondiff_param_name, default_arg in dfb.other_params:
            raw_node_nondiff_arg = raw_normalized_kwargs[nondiff_param_name]

            if isinstance(raw_node_nondiff_arg, Sequence):
                raise NotImplementedError("Nested non-differentiable arg")
                # PyTorch unlikely has this.
            
            if isinstance(raw_node_nondiff_arg, Node):
                # Will result in GET_ATTR[tensor_contants0] Nodes in subgraph.
                jvp_nondiff_val = self._create_zero_val(raw_node_nondiff_arg)
                assert isinstance(jvp_nondiff_val, torch.Tensor)

                nondiff_val2node[jvp_nondiff_val] = raw_node_nondiff_arg

            else:
                assert raw_node_nondiff_arg is None \
                    or isinstance(raw_node_nondiff_arg, (int, float, str))
                jvp_nondiff_val = raw_node_nondiff_arg

            # TODO this takes effect even user specifies the value to be
            # explicitly None.
            # It seems ok that no common operators offer a default value
            # (especially when the argument is omitted at callsite)
            # that is not None.
            if jvp_nondiff_val is None:
                # Sometimes FX normalization will add omitted optional argument
                # which is inferred from op definition, effectively same as
                # `default_arg`. But if that arg appears in node.kwargs, it
                # means user has explicitly specified it.
                if nondiff_param_name in self.current_node.kwargs:
                    raise NotImplementedError(
                        "User explicitly specifies None arg in"
                        f" {self.current_node}"
                        ", may indicate that an internal assumption is broken"
                    )
                assert not isinstance(default_arg, RequiredParam)
                jvp_nondiff_val = default_arg
            
            jvp_nondiff_env_vals[
                nondiff_param_name
            ] = jvp_nondiff_val  # type: ignore
        
        
        # Differentiable Tensor-type parameters, must be passed via jvp()
        # API param list.
        # The names and the zero Tensors must be in the same order.
        diff_arg_names: List[str] = dfb.diffable_params
        diff_primal_vals = list(map(
            jvp_diff_primal_vals.__getitem__, diff_arg_names
        ))
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
        
        def _unflatten(
            positions: List[List[int]], items: Sequence[torch.Tensor]
        ):
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

            

        flattened_primal_tree, flattened_primal_vals = \
            _flatten(diff_primal_vals)
        flattened_tangent_tree, flattened_tangent_vals = \
            _flatten(tangent_vals)
        assert flattened_primal_tree == flattened_tangent_tree

        # Other non-differentiable parameters must NOT be passed via jvp()
        # API param list, but via function closure.

        is_aten_api = 'aten' in function.__module__

        def _primal_func(*flattened_primals: torch.Tensor):
            # There preparations are not part of FX Proxy and won't be traced
            unfltd_primals = _unflatten(
                flattened_primal_tree, flattened_primals
            )

            kw = {}
            for diff_arg_name, v in zip(diff_arg_names, unfltd_primals):

                if is_aten_api and diff_arg_name == 'input':
                    diff_arg_name = 'self'

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
        
        # NOTE depending on whether we are passing primals by a tuple like
        # `flattened_primals` or by individual parameter for each primal,
        # the output Node of `gm` from ** fx.experimental.make_fx **
        # will have nested structure or will not:
        #
        # If by individual parameter
        # `def _jvp(to_sort: Tensor, arg_tangent: Tensor)` then:
        # `return ((sorted, index), (gathered_tangent, zero))`
        #
        # If by a tuple like below
        # `def _jvp(flattened_primals: Tuple[torch.Tensor, ...],)` then:
        # return ([sorted, index, gathered_tangent, zero],)
        #
        # But when evaluate `y, jvp_res = gm(vals, tangents)` the results
        # will have proper structure.
        
        # FX Graph, including make_fx, does not respect nested inputs,
        # so we need to manually flatten any nested args and maintain the
        # mapping between before/after flattening.
        gm: GraphModule = make_fx(_jvp)(
            flattened_primal_vals, flattened_tangent_vals
        )
        y, jvp_res = gm(flattened_primal_vals, flattened_tangent_vals)

        ng = get_node_tensor_group(self.current_node)
        if ng is None:
            role = Role.REPLICATED
        else:
            role = Role.DISTRIBUTED

        from easier.core.runtime.jit_engine.jit_engine import \
            get_value_runtime_info
        y_meta = get_value_runtime_info(
            self.current_node, y, self._fake_eval_meta_ctor
        )
        jvp_meta = get_value_runtime_info(
            self.current_node, jvp_res, self._fake_eval_meta_ctor
        )
        assert y_meta == jvp_meta

        # Connect jvp() assigned _tensor_constantsN fields with nondiff Nodes
        nondiff_jvp_submod_attrnames = {}
        for const_name, nondiff_val in gm.named_buffers():
            nondiff_jvp_submod_attrnames[const_name] = nondiff_val2node[nondiff_val]

        # TODO return bijective position mapping
        return gm, y_meta, flattened_primal_tree, nondiff_jvp_submod_attrnames

class _TorchJvpSubGraphCopier(EasierInterpreter):
    def __init__(
        self, subgm: GraphModule, subg: Graph, jvp_graph: Graph,
        input_flatten_tree: List[List[int]],
        jvp_input_primals: List[Union[FxConst, Node, Sequence[Node]]],
        jvp_input_tangents: List[Union[FxConst, Node, Sequence[Node]]],
        n_flattened_primal_outputs: int,
        nondiff_jvp_submod_attrnames: Dict[str, Node]
    ) -> None:
        super().__init__([subgm], [subg])  # type: ignore

        self.jvp_graph = jvp_graph
        # self.diff_arg_names = diff_arg_names
        self.jvp_input_primals = jvp_input_primals
        self.jvp_input_tangents = jvp_input_tangents
        assert len(jvp_input_primals) == len(jvp_input_tangents)

        self.input_flatten_tree = input_flatten_tree
        self.n_flattened_primal_inputs = len(input_flatten_tree)

        self.n_flattened_primal_outputs =  n_flattened_primal_outputs

        self.nondiff_jvp_submod_attrnames = nondiff_jvp_submod_attrnames

        self.placeholder_i = 0  # totally 2*len(diff_arg_names)

        # self.nondiff_getattr_i = 0  # totally len(nondiff_inputs)

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
        is_primal = self.placeholder_i < self.n_flattened_primal_inputs

        def _from_flatten(
            jvp_inputs: List[Union[FxConst, Node, Sequence[Node]]],
            tree: List[int]
        ) -> Union[Node, FxConst]:
            if len(tree) == 1:
                [i] = tree
                return jvp_inputs[i]  # type: ignore
            else:
                [i, ii] = tree
                return jvp_inputs[i][ii] # type: ignore

        if is_primal:
            ph_pos = self.placeholder_i

            jvp_primal = _from_flatten(
                self.jvp_input_primals, self.input_flatten_tree[ph_pos]
            )
            self.nodemap_subg2jvp[self.current_node] = jvp_primal
        else:
            ph_pos = self.placeholder_i - self.n_flattened_primal_inputs

            jvp_tangent = _from_flatten(
                self.jvp_input_tangents, self.input_flatten_tree[ph_pos]
            )
            self.nodemap_subg2jvp[self.current_node] = jvp_tangent

        self.placeholder_i += 1
    
    def if_get_attr(self, submod_path: str, attr_name: str, attr_val):
        assert submod_path == ''

        self.nodemap_subg2jvp[self.current_node] = \
            self.nondiff_jvp_submod_attrnames[attr_name]
    
    def if_call_function(self, function):
        jvp_node = self.jvp_graph.node_copy(
            self.current_node, self.nodemap_subg2jvp.__getitem__
        )

        self.nodemap_subg2jvp[self.current_node] = jvp_node
    
    def if_output(self):
        """
        When without AD, a multi-res Node stands individually, it's from
        tensor metadata level can we know it's multi-res.

        However, in torch.func.jvp sub-Graph, the primal multi-res Node will
        first be unpacked, then all primal result items are put in OUTPUT
        Node's args[0] list, making it NO LONGER an individual Node.
        When copying the sub-Graph into jvp Graph, we need to handle this.
        """
        jvp_subgraph_out = cast(Sequence[Node], self.current_node.args[0])
        assert len(jvp_subgraph_out) == 2 * self.n_flattened_primal_outputs, \
            "With currently organization of _primal_func and _jvp above," \
            " We expect torch.func.jvp() sub-Graph flatten and concat all" \
            " primal and tangent result items"

        convert = self.nodemap_subg2jvp.__getitem__

        if self.n_flattened_primal_outputs == 1:
            self.output_primal = tree_map(jvp_subgraph_out[0], convert)
            self.output_tangent = tree_map(jvp_subgraph_out[1], convert)
        else:
            self.output_primal = tree_map(
                jvp_subgraph_out[:self.n_flattened_primal_outputs], convert
            )
            self.output_tangent = tree_map(
                jvp_subgraph_out[self.n_flattened_primal_outputs:], convert
            )
    
    def if_call_method(self, method_name: str):
        raise NotImplementedError(
            "It's unlikely that torch.func.jvp generates method calls"
        )
    
    def if_call_module(self, submod: Module):
        raise NotImplementedError(
            "It's unlikely that torch.func.jvp generates module calls"
        )


class Jvp(esr.Module):
    def __init__(
        self,
        inputs: Sequence[esr.Tensor],
        outputs: Sequence[esr.Tensor],
        vectors: Optional[Sequence[esr.Tensor]] = None
    ):
        super().__init__()

        # nn.ParamList is not actually a Sequence[Tensor] because it lacks
        # __contains__.
        self.inputs: Sequence[esr.Tensor] = \
            torch.nn.ParameterList(inputs)  # type: ignore
        self.outputs: Sequence[esr.Tensor] = \
            torch.nn.ParameterList(outputs)  # type: ignore

        if vectors is not None:
            self._check_args_nondup_and_dtype(
                list(inputs) + list(outputs) + list(vectors),
                'inputs + outputs + vectors'
            )

            if len(vectors) != len(inputs):
                raise ValueError(
                    f"The number of vectors {len(vectors)} does not match the"
                    f" number of inputs {len(inputs)}"
                )
            
            for i, v in zip(inputs, vectors):
                if i.dtype != v.dtype or i.shape != v.shape:
                    raise ValueError(
                        "Vector's dtype/shape does not match the input"
                    )

            self.vectors = cast(Sequence[esr.Tensor], torch.nn.ParameterList(
                vectors
            ))

        else:
            self._check_args_nondup_and_dtype(
                list(inputs) + list(outputs),
                'inputs + outputs'
            )

            self.vectors = cast(Sequence[esr.Tensor], torch.nn.ParameterList(
                esr.Tensor(
                    esr.zeros_like(input),
                    mode=('partition' if input.is_partition else 'replicate')
                )
                for input in inputs
            ))
        

        self.products = cast(Sequence[esr.Tensor], torch.nn.ParameterList(
            esr.Tensor(
                esr.zeros_like(output),
                mode=('partition' if output.is_partition else 'replicate')
            )
            for output in outputs
        ))

        # Type hint for local _Jvp class created by esr.jvp()
        self.graph_module: GraphModule

    
    def _check_args_nondup_and_dtype(
        self, args: Sequence[esr.Tensor], param_name: str
    ):
        for pos, arg in enumerate(args):
            if not arg.dtype.is_floating_point:
                raise ValueError(
                    f"The {pos}-th easier.Tensor does not have floating-point"
                    f" dtype in {param_name}"
                )

        if len(set(args)) != len(args):
            raise ValueError(
                f"Some easier.Tensor gets specified multiple times"
                f" in {param_name}"
            )
        

def jvp(
    module: esr.Module,
    inputs: Sequence[esr.Tensor],
    outputs: Sequence[esr.Tensor],
    vectors: Optional[Sequence[esr.Tensor]] = None
) -> Jvp:
    """
    Transform an easier.Module to a new easier.Module that evaluates the
    _Jacobian vector product_ (JVP) when invoking, with regards to
    `inputs/outputs`.

    Args:
    -   module: The easier.Module whose JVP will be evaluated.s

    -   inputs:

        The easier.Tensors which are expected to be paired with initial
        tangents, i.e. the _vector_ in _Jacobian vector product_.

    -   outputs:

        The easier.Tensors which are expected to store the result of evaluating
        `module`, and against which the JVP will be calculated.

        The initial or immediate values of `outputs` can be read in
        `module.forward()`.
    
    -   vectors:

        Users may provide their own easier.Tensors which stores initial
        tangents.

        Otherwise, zero-value easier.Tensors will be created by `easier.zeros`
        and users need extra code to set up initial tangents.

        However, when needed, the values of these easier.Tensors should be
        explicitly updated between multiple invocations of the resultant
        easier.Jvp module.
    
    Return:
    -   A new easier.Jvp (deriving easier.Module) module instance. With these
        attributes:

        -   Jvp.inputs (torch.nn.ParameterList): Equals to `inputs`
        -   Jvp.outputs (torch.nn.ParameterList): Equals to `outputs`

        -   Jvp.vectors (torch.nn.ParameterList):
            Equals to `vectors` if provided.
            Otherwise, many zero-value easier.Tensors will be created
            and have the same shapes/dtypes as `inputs`.

        -   Jvp.products (torch.nn.ParameterList):
            Many zero-value easier.Tensors will be created and have the
            same shapes/dtypes as `outputs`.
    
    Example:
    ```
    TODO
    ```

    Remarks:
    -   Input and output easier.Tensors are all mutable.
        Even, for example, `input[:] = f(output)` that `vector` tensor for
        `input` will store resultant tangent.

    -   In `inputs/outputs/vectors`, the same easier.Tensor instance cannot be
        specified twice.
    """

    # To make Jvp Module FX-traceable, where FX picks the entrance `forward`
    # method on the CLASS instead of an Jvp Module instance, we need to
    # create a local class for each jvp call to provide class-level `forward`.
    class _Jvp(Jvp):
        def forward(self):
            # Products are not overlapping with other esr.Tensors,
            # zero-initialize them every time, since uninvolved tensors will
            # have zero tangents.
            for p in self.products:
                p.zero_()

            # The JvpTransformer-generated Graph will be inlined here.
            self.graph_module()
            

    jvpm = _Jvp(inputs, outputs, vectors)
    jvp_transformer = JvpTransformer(module, jvpm).run()

    gm = GraphModule(jvpm, jvp_transformer.jvp_graph)
    jvpm.graph_module = gm

    return jvpm