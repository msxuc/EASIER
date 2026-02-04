# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import operator
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Type, TypeAlias, TypeVar, Union, cast

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
    FX, EasierInterpreter, SubmodNameAllocator, normalize_reducer_call_into_args, \
    fx_normalize_function_variant_into_kwargs, tree_map, \
    normalize_selector_call_into_args, get_attr_value

from easier.core.runtime.metadata import \
    Role, RuntimeTensorMeta, StructuredTensorMeta, \
    collect_meta, set_node_meta, get_node_meta
    
from easier.core.autodiff.jvp import Jvp

from easier.core.autodiff.autodiff_rule import \
    DiffRuleBase, Differentiability, RequiredParam, \
    diff_rule_registry, differentiabilities, getitem_aux_kw
from easier.core.autodiff.utils import PrimalMetaPropagator, create_zero_arg_val, simplify_torchfunc_fx_graph, FxConst, wrap_operator_specific_kwargs_normalizer
from easier.core.utils import EasierJitException, logger

_T = TypeVar('_T')


"""
About using torch.func.vjp to generate VJP Graph:

A code snippet:

```python
def myjvp(primals, cotangents):
    y, vjpfun = torch.func.vjp(torch.bmm, *primals)
    grad = vjpfun(*cotangents)
    y: torch.Tensor
    return y, grad

x = torch.rand(100, 2, 2)
gm = make_fx(myjvp)((x, x.clone()), (x.clone(), ))
```

will generate a Graph like:

```
graph():
    %primals_1 : [num_users=2] = placeholder[target=primals_1]
    %primals_2 : [num_users=2] = placeholder[target=primals_2]
    %cotangents_1 : [num_users=2] = placeholder[target=cotangents_1]

    %bmm : [num_users=1] = call_function[
            target=torch.ops.aten.bmm.default
        ](args = (%primals_1, %primals_2), kwargs = {})
    # ^^^ end of primal part

    %transpose : [num_users=1] = call_function[
            target=torch.ops.aten.transpose.int
        ](args = (%primals_1, 1, 2), kwargs = {})
    %bmm_1 : [num_users=1] = call_function[
            target=torch.ops.aten.bmm.default
        ](args = (%transpose, %cotangents_1), kwargs = {})
    %transpose_1 : [num_users=1] = call_function[
            target=torch.ops.aten.transpose.int
        ](args = (%primals_2, 1, 2), kwargs = {})
    %bmm_2 : [num_users=1] = call_function[
            target=torch.ops.aten.bmm.default
        ](args = (%cotangents_1, %transpose_1), kwargs = {})

    return [bmm, bmm_2, bmm_1] 
    #       ^^^
```

torch.vjp generates a Graph that contains both primal computation and
cotangent computation.

By returning the primal result `y`, the `bmm` in the graph, we can decide
which part is primal computation and which part is cotangent computation.


Then, we can wise the primal parts for each raw Node to build the primal part
of the VJP Graph.

For the cotangent part of the VJP Graph, we need to **reversely** wise the
cotangent computation part for each raw Node, with `cotangents` parameter
being replaced by the resultant cotangent of the previous VJP subgraph,
in a sense of backpropagation.
"""


class VjpTransformer(EasierInterpreter):
    """
    """
    def __init__(
        self,
        module: esr.Module,
        vjp_module: 'Vjp',
    ):
        # coll_init returns all nested esr.Modules, here we takes only the top
        (module, *_), (graph, *_) = collectively_initialize_and_validate([module])
        [module], [graph] = group_tensors([module], [graph])
        PrimalMetaPropagator([module], [graph]).run()

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

        class _ReadWriteValidator(EasierInterpreter):
            def if_call_function(self, function):
                if function is operator.setitem:
                    container, index, value = self.current_node.args
                    assert isinstance(container, Node)
                    assert isinstance(value, Node)

                    if container.op != FX.GET_ATTR:
                        raise NotImplementedError(
                            "EASIER VJP currently only supports setitem to"
                            " GET_ATTR new_value Tensors"
                        )
                    
                    if len(container.users) > 1:
                        raise EasierJitException(
                            "EASIER VJP currently only allows writing to"
                            " a Tensor without any readers and only once."
                        )
        _ReadWriteValidator([module], [graph]).run()

        super().__init__([module], [graph], reverse=True)

        self.raw_module = module
        self.vjp_module = vjp_module

        self.vjp_graph = Graph()
        self.nodemap_raw2primal: Dict[Node, Node] = {}

        # Some primal Nodes will be replaced if it's decided to carry cotangent
        # during reverse traversal.
        for raw_node in graph.nodes:
            if raw_node.op == FX.OUTPUT:
                break
            vjp_primal_node = self._strict_map_raw_to_vjp(raw_node)
            self.nodemap_raw2primal[raw_node] = vjp_primal_node
        
        # When the reverse traversal reaches the definition of a raw Node,
        # if that raw Node is not in this dict, it means that raw Node does not
        # have cotangent paired.
        self.nodemap_primal2sumcot: Dict[Node, Node] = {}

        self.tensormap_primal2cotangent: Dict[esr.Tensor, esr.Tensor] = {}
        
        # Primal inputs/outputs, tangent inputs/outputs have been setattr-ed
        # to Jvp Module in its `.inputs/.outputs/.vectors/.products`
        # torch.nn.ParamList fields, we need to rebind the attr paths.
        #
        # NOTE currently if an esr.Tensor/GET_ATTR-Node gets written with
        # tangent, we only 1) ensure its GET_ATTR Node is unique; 2) inject a
        # zero_like Node for GET_ATTR Node.
        # I.e. we don't allocate an esr.Tensor for such immediate tangents.
        self.vjp_tensors_attrpaths: Dict[esr.Tensor, str] = {}

        for i, (output, vector) in enumerate(zip(
            vjp_module.outputs, vjp_module.vectors
        )):
            self.tensormap_primal2cotangent[output] = vector

            self.vjp_tensors_attrpaths[output] = f'outputs.{i}'
            self.vjp_tensors_attrpaths[vector] = f'vectors.{i}'


        for i, (input, product) in enumerate(zip(
            vjp_module.inputs, vjp_module.products
        )):
            self.tensormap_primal2cotangent[input] = product

            self.vjp_tensors_attrpaths[input] = f'inputs.{i}'
            self.vjp_tensors_attrpaths[product] = f'products.{i}'
        

        self._setup_vjp_module_attrs()

        # Attr name for ()-shape constant Tensors created for literal Scalar
        # arguments, those constants are made attributes to make them aware
        # of AOT target device like CUDA.
        self.const_name_allocator = SubmodNameAllocator('const')

        self.bp_submod_name_allocator = SubmodNameAllocator('bp')

        self.sr_map: Dict[
            Union[esr.Selector, esr.Reducer],
            Tuple[Union[esr.Selector, esr.Reducer], str]
        ] = {}

    def _setup_vjp_module_attrs(self):
        """
        Copy other submod (no matterr esr.Module or common torch.nn.Module)
        and other parameters to Vjp module,
        this allows accesses to attrbutes like `vjpm.submod1.submod2.x`.
        """
        for path, obj in list(
            self.raw_module.named_modules()
        ) + list(
            self.raw_module.named_parameters(recurse=False)
        ):
            if '.' in path or path == '':
                # - submod children: inherited, since we re-assign the submod;
                # - ''-path: the root module itself
                continue

            if path in ['inputs', 'outputs', 'vectors', 'products']:
                raise EasierJitException(
                    f"{self.raw_module.__class__.__name__}.{path}"
                    " attribute name conflicts with easier.vjp()"
                )
            
            setattr(self.vjp_module, path, obj)
            

    def _ensure_vjp_const_attr(
        self, const: torch.Tensor, attrname_hint: str = ''
    ) -> str:
        primal_attrname = self.const_name_allocator.alloc_name(
            self.vjp_module, attrname_hint
        )
        setattr(self.vjp_module, primal_attrname, const)

        return primal_attrname
    
    def _ensure_bp_submod_attr(
        self, bp: Union[esr.Selector, esr.Reducer], attrname_hint: str = ''
    ) -> str:
        bp_attrname = self.bp_submod_name_allocator.alloc_name(
            self.vjp_module, attrname_hint
        )
        setattr(self.vjp_module, bp_attrname, bp)

        return bp_attrname

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
    

    def _create_zero_cotangent(
        self, raw_node: Node
    ) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        def _make(x):
            assert isinstance(x, RuntimeTensorMeta)
            return torch.zeros(x.shape, dtype=x.dtype)

        meta = get_node_meta(raw_node)
        return tree_map(meta, _make)
        



    def if_get_attr(self, submod_path: str, attr_name: str, attr_val) -> None:
        primal = self.nodemap_raw2primal[self.current_node]

        if primal in self.nodemap_primal2sumcot:
            cot = self.nodemap_primal2sumcot[primal]

            if attr_val in self.tensormap_primal2cotangent:
                grad_tensor = self.tensormap_primal2cotangent[attr_val]
                path = self.vjp_tensors_attrpaths[grad_tensor]

                grad_node = self.vjp_graph.get_attr(path)
                self.vjp_graph.call_function(operator.setitem, (grad_node, (slice(None),), cot))


    def if_call_method(self, method_name: str):
        function = getattr(torch.ops.aten, method_name)
        self._handle_operation(function)


    def if_call_function(self, function: Callable) -> None:
        if function is operator.setitem:
            # A special syntactic Node for VJP to write the resultant primal
            # Tensors.
            container, index, value = self.nodemap_raw2primal[self.current_node].args
            assert isinstance(container, Node)
            assert isinstance(value, Node)

            if index not in [slice(None), Ellipsis]:
                raise NotImplementedError(
                    "EASIER VJP currently only supports setitem with Ellipsis"
                )
            
            if container.op != FX.GET_ATTR:
                raise NotImplementedError(
                    "EASIER VJP currently only supports setitem to"
                    " GET_ATTR new_value Tensors"
                )
            
            container_val = get_attr_value(self.raw_module, container)
            assert isinstance(container_val, esr.Tensor)

            if container_val in self.tensormap_primal2cotangent:
                cot_tensor = self.tensormap_primal2cotangent[container_val]

                cot_attrname = self.vjp_tensors_attrpaths[cot_tensor]
                cot_node = self.vjp_graph.get_attr(cot_attrname)

                # Writes to `output` is the final use of the `value` Node.
                self.nodemap_primal2sumcot[value] = cot_node

            return

        if function is operator.getitem:
            container, index = self.current_node.args
            assert isinstance(container, Node)
            imeta = get_node_meta(container)

            if isinstance(imeta, Sequence):
                # This is the raw Node for unpacking a tuple, will be handled
                # when handling the multi-res raw Node in a few future steps.
                return

        self._handle_operation(function)
    

    def _try_prepare_get_cotangent(self, dfb: Union[DiffRuleBase, Differentiability]) -> Union[Node, Sequence[Node], None]:
        primal_node = self.nodemap_raw2primal[self.current_node]

        out_diff = dfb.output_differentiability

        if isinstance(out_diff, bool):
            assert out_diff == True

            return self.nodemap_primal2sumcot.get(primal_node, None)            
        
        else:
            assert isinstance(out_diff, Sequence)
            assert all(out_diff)

            comps = [None] * len(out_diff)
            for raw_getitem in self.current_node.users:
                assert raw_getitem.target is operator.getitem
                _, item_i = raw_getitem.args
                assert isinstance(item_i, int)

                cot_comp = self.nodemap_primal2sumcot.get(self.nodemap_raw2primal[raw_getitem], None)
                comps[item_i] = cot_comp # type: ignore
            
            missing = any(comp is None for comp in comps)

            if missing:
                if not all(comp is None for comp in comps):
                    raise NotImplementedError(
                        "For the call to multiple-result operator"
                        f" {self.current_node.target}, some of its result"
                        " components have cotangent while some do not"
                    )
                return None

            meta = get_node_meta(self.current_node)
            assert isinstance(meta, Sequence)

            # Must be exactly the same tuple type like `torch.types.svd`
            cot_coll = type(meta)(comps) # type: ignore

            return cot_coll # type: ignore



    def _prepare_diffable_primals(
        self,
        function: Callable,
        raw_node_diff_args: Dict[str, Union[FxConst, Node, Sequence[Node]]]
    ) -> List[Union[Node, Sequence[Node]]]:
        """
        Given the order of raw diff-able args in `raw_node_diff_args`,
        return a list of _prepared_ primals to
        achieve the VJP calculation of this raw Node.

        Being _prepared_ means scalar diff-able args will converted to
        ()-shape Tensors/Nodes in the VJP Graph.
        This can also be seen from the absence of FxConst in the result type.
        """
        # Must be in the same (whatever) order as `raw_node_diff_args`
        input_primal_nodes: List[Union[Node, Sequence[Node]]] = []

        diffable_roles = []

        for argname, raw_node_diff_arg in raw_node_diff_args.items():

            if not isinstance(raw_node_diff_arg, (Node, Sequence)):
                # const scalars
                assert isinstance(raw_node_diff_arg, (int, float, str)), \
                    "If a differentiable parameter is not a fx.Node" \
                    " or Node list, it must be a scalar"
                
                # Zero tangent for scalar
                #
                # Because when invoking torch.jvp(), we convert scalar-type
                # diffable arguments to ()-shape tensors.
                # Because of this assumption, torch.jvp() will geneerate calls
                # to operator overloading that requires args to be tensors.
                # If we keep the raw scalar arguments, TypeError will occur.
                #
                # Therefore, before we do sub-Graph inlining, we need to
                # convert raw scalar args to tensors/Nodes in the JVP graph.
                #
                num_hint = str(raw_node_diff_arg).replace('.', '_')[:10]
                primal_const_tensor_attrname = self._ensure_vjp_const_attr(
                    torch.full((), raw_node_diff_arg, dtype=torch.float32),
                    f"{function.__name__}_{argname}_{num_hint}"
                )
                input_primal_nodes.append(
                    self.vjp_graph.get_attr(primal_const_tensor_attrname)
                )

                diffable_roles.append(Role.REPLICATED)
            
            else:  # Node or Node list
                input_primal_nodes.append(tree_map(
                    raw_node_diff_arg, self.nodemap_raw2primal.__getitem__
                ))

                diffable_roles.extend(collect_meta(
                    collect_meta(
                        raw_node_diff_arg, get_node_meta, leaf_type=Node
                    ),
                    lambda meta: meta.role
                ))

        if len(set(diffable_roles)) >= 2:
            logger.warning(
                f"{self.current_node} takes both distributed and replicated"
                " arguments, use `expand_as` on the replicated arguments first"
                " and OUTSIDE OF esr.vjp scope"
            )
            # TODO during vjp we should detect broadcasting semantics from
            # replica to distributed tensors, expand_as it manually, and
            # generate an `esr.sum` for BP, otherwise torch.vjp will generate
            # a common torch.sum which breaks EASIER programming model.


        return input_primal_nodes
    

    def _handle_operation(self, function: Callable):
        if function in diff_rule_registry:
            rule_cls = diff_rule_registry[function]
            rule = rule_cls(
                self.current_node, function, self._fake_eval_meta_ctor
            )

            cotangent = self._try_prepare_get_cotangent(rule)
            if cotangent is None:
                return
            
            primal_node = self.nodemap_raw2primal[self.current_node]

            out_meta = get_node_meta(self.current_node)
            if isinstance(out_meta, Sequence):  # multi-res Node
                rule_jvp_result = []

                for user_i, raw_getitem in enumerate(self.current_node.users):
                    assert raw_getitem.target is operator.getitem
                    _, item_i = raw_getitem.args
                    assert isinstance(item_i, int)
                    assert item_i == user_i

                    primal_getitem = self.nodemap_raw2primal[raw_getitem]
                    rule_jvp_result.append(primal_getitem)
                
                rule_jvp_result = tuple(rule_jvp_result)
            else:
                rule_jvp_result = primal_node
            
            raw_node_diff_args = rule.input_differentiability(
                *self.current_node.args, **self.current_node.kwargs
            )
     
            input_primal_nodes = self._prepare_diffable_primals(
                function, raw_node_diff_args
            )

            # Append the VJP graph
            input_cotangents = rule.inject_vjp_subgraph(
                primal_node,
                list(raw_node_diff_args.keys()),
                self.nodemap_raw2primal,
                cotangent
            )

            for ip, ic in zip(input_primal_nodes, input_cotangents):
                if isinstance(ip, Node):
                    assert isinstance(ic, Node)
                    if ip in self.nodemap_primal2sumcot:
                        self.nodemap_primal2sumcot[ip] = self.vjp_graph.call_function(
                            torch.add, (self.nodemap_primal2sumcot[ip], ic)
                        )
                    else:
                        self.nodemap_primal2sumcot[ip] = ic
                
                else:
                    assert isinstance(ip, Sequence)
                    assert isinstance(ic, Sequence)
                    for sub_ip, sub_ic in zip(ip, ic):
                        if sub_ip in self.nodemap_primal2sumcot:
                            self.nodemap_primal2sumcot[sub_ip] = self.vjp_graph.call_function(
                                torch.add, (self.nodemap_primal2sumcot[sub_ip], sub_ic)
                            )
                        else:
                            self.nodemap_primal2sumcot[sub_ip] = sub_ic

        
        else: # not in diff_rule_registry
            wrap_normalizer = lambda f: f

            if getattr(operator, function.__name__, None) is function:
                if function is operator.getitem:
                    # Special path, see differentiabilities[operator.getitem]
                    function = getitem_aux_kw
                    
                elif function is operator.truediv:
                    function = torch.div  # torch does not have truediv
                    wrap_normalizer = wrap_operator_specific_kwargs_normalizer
                else:
                    function = getattr(torch, function.__name__)
                    wrap_normalizer = wrap_operator_specific_kwargs_normalizer
                
            if function not in differentiabilities:
                raise NotImplementedError(
                    f"Operator {function} is not registered in EASIER AutoDiff"
                )

            dfbs = differentiabilities[function]

            for dfb in dfbs:
                # TODO this is actually shared by all Differentiability
                # overloadings.
                raw_node_normalized_kwargs: Dict[str, FxArg] = \
                    (
                        wrap_normalizer(dfb.kwargs_normalizer)
                    )(
                        function,
                        self.current_node.args,
                        self.current_node.kwargs
                    )  # type: ignore
                if dfb.all_param_names() == set(
                    raw_node_normalized_kwargs.keys()
                ):
                    break
            else:
                assert False, \
                    "Failed to resolve overloading:" \
                    f" with Differentiabilities {dfbs}," \
                    f" got {raw_node_normalized_kwargs}"
            

            cotangent = self._try_prepare_get_cotangent(dfb)
            if cotangent is None:
                return


            raw_node_diff_args: Dict[
                str, Union[FxConst, Node, Sequence[Node]]
            ] = {
                p: raw_node_normalized_kwargs[p] for p in dfb.diffable_params
            }  # type: ignore
            
            y_meta = get_node_meta(self.current_node)

            # One result: positions == [0]
            # Multi results: positions == [[0,1], [0,2], ...]
            flattened_output_tree, _ = self._flatten([y_meta], leaf_type=RuntimeTensorMeta)

            gm, flattened_input_tree, nondiff_inputs_attrname2raw = \
                self._generate_vjp_subgraph_using_torchfunc(
                    function, dfb, raw_node_normalized_kwargs
                )
            subg = simplify_torchfunc_fx_graph(gm)

            out_diff = dfb.output_differentiability

            nondiff_inputs_attrname2vjp = {
                k: self.nodemap_raw2primal[v]
                for k, v in nondiff_inputs_attrname2raw.items()
            }

            primal_node = self.nodemap_raw2primal[self.current_node]
            
            # Edit VJP primal Node
            with self.vjp_graph.inserting_before(primal_node.next):
                input_primal_nodes = self._prepare_diffable_primals(
                    function, raw_node_diff_args
                )

                # Because when invoking torch.vjp we treat all scalars as
                # ()-shape tensors, here we also need to prepare a ()-shape
                # attribute for that scalar.
                #
                # This would make torch.vjp generate subgraph Node for that
                # ()-shape input, consequently lead to invalid operation like
                # summing cotangent when
                # e.g. X * r = Y when X is dist and r is replica.
                # We need to prune subgraph Nodes only contribute to $d_r$.
                prune_diff_args: List[int] = []
                for arg_i, (arg_name, raw_arg) in enumerate(raw_node_diff_args.items()):
                    if isinstance(raw_arg, (int, float, str)):
                        prune_diff_args.append(arg_i)

                primal_copier = _TorchVjpSubGraphPrimalPartCopier(
                    gm, subg, self.vjp_graph,
                    flattened_input_tree, input_primal_nodes,
                    len(flattened_output_tree),
                    self.current_node,
                    nondiff_inputs_attrname2vjp
                ).run()
            
            # TODO it seems unnecessary to have two Copiers, can be refactored
            # to just one and two sub-phases.

            # Resume to insert at the end of VJP graph
            cotangent_copier = _TorchVjpSubGraphCotangentPartCopier(
                gm, subg, self.vjp_graph, primal_copier,
                cotangent,
                flattened_output_tree,
                nondiff_inputs_attrname2vjp,
                self.nodemap_primal2sumcot,
                prune_diff_args
            ).run()

            if isinstance(out_diff, bool):
                assert isinstance(primal_copier.output_primal, Node)
                assert out_diff == True
                primal_node.replace_all_uses_with(primal_copier.output_primal)
                self.vjp_graph.erase_node(primal_node)

                self.nodemap_raw2primal[self.current_node] = \
                    primal_copier.output_primal
            
            else:
                assert isinstance(primal_copier.output_primal, Sequence)
                assert isinstance(out_diff, Sequence)

                # NOTE it's likely the RAW multi-res Node doesn't have
                # explicit primal/tangent counterpart Nodes, so we can only
                # set up binding on the unpacking getitem Nodes.

                for raw_getitem in self.current_node.users:
                    assert raw_getitem.target is operator.getitem
                    _, item_i = raw_getitem.args
                    assert isinstance(item_i, int)

                    primal_getitem = self.nodemap_raw2primal[raw_getitem]
                    primal_getitem.replace_all_uses_with(
                        primal_copier.output_primal[item_i]
                    )
                    self.vjp_graph.erase_node(primal_getitem)

                    self.nodemap_raw2primal[raw_getitem] = \
                        primal_copier.output_primal[item_i]
                
                self.vjp_graph.erase_node(primal_node)

                self.nodemap_primal2sumcot.pop(primal_node)
            # endif out_diff


    def _strict_map_raw_to_vjp(self, raw: Node):
        """
        fx.Graph.node_copy tends to silently hide raw-not-existing error in
        nodemap.__getitem__ arg transformation (in torch C++ level),
        silently returning a None constant in the VJP Graph.

        This aux method validates the consistency of both nodemaps
        in this class.
        """
        for raw_input in raw.all_input_nodes:
            assert raw_input in self.nodemap_raw2primal, \
                "Raw Node's Node input must be in raw-primal Node map"
        
        vjp_node = self.vjp_graph.node_copy(
            raw, arg_transform=self.nodemap_raw2primal.__getitem__
        )
        vjp_node.meta = {}

        return vjp_node


    def if_call_module(self, submod: Module):
        primal_node = self.nodemap_raw2primal[self.current_node]

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
            input_primal = self.nodemap_raw2primal[input]

            # Don't use meta as it has fake 1000 bs
            co_reducer_n = get_node_tensor_group(input).n

            if primal_node in self.nodemap_primal2sumcot:
                vjp_out_cot = self.nodemap_primal2sumcot[primal_node]

                if submod not in self.sr_map:
                    reducer = esr.Reducer(submod.easier_data_loader, n=co_reducer_n)
                    reducer_name = self._ensure_bp_submod_attr(reducer, submod.easier_hint_name)
                    self.sr_map[submod] = (reducer, reducer_name)
                else:
                    _, reducer_name = self.sr_map[submod]

                cot_node = self.vjp_graph.call_module(reducer_name, (vjp_out_cot,))
        
                self.nodemap_primal2sumcot[input_primal] = cot_node


        elif isinstance(submod, esr.Reducer):
            if submod.reduce != 'sum':
                raise NotImplementedError()

            input_primal, out = normalize_reducer_call_into_args(
                *primal_node.args, **primal_node.kwargs
            )
            assert isinstance(input_primal, Node)
            if out is not None:
                raise EasierJitException("cannot use Reducer(out=...)")

            if primal_node in self.nodemap_primal2sumcot:
                vjp_out_cot = self.nodemap_primal2sumcot[primal_node]

                if submod not in self.sr_map:
                    selector = esr.Selector(submod.easier_data_loader)
                    selector_name = self._ensure_bp_submod_attr(selector, submod.easier_hint_name)
                    self.sr_map[submod] = (selector, selector_name)
                else:
                    _, selector_name = self.sr_map[submod]

                cot_node = self.vjp_graph.call_module(selector_name, (vjp_out_cot,))
        
                self.nodemap_primal2sumcot[input_primal] = cot_node

        else:
            assert False, 'unreachable'

    def _flatten(
        self,
        vals: List[Union[_T, Sequence[_T]]],
        leaf_type: Type = torch.Tensor
    ):
        """
        Always take a parameter list, therefore containing
        the positions of parameters.
        """
        def _flatten_pos(
            i_val: Tuple[int, Union[_T, Sequence[_T]]]
        ) -> List[Tuple[List[int], _T]]:
            i, val = i_val
            if isinstance(val, leaf_type):
                return [([i], val)]  # type: ignore
            else:
                i_ii_vals = []
                for ii, item in enumerate(val):  # type: ignore
                    i_ii_vals.append(([i, ii], item))
                return i_ii_vals

        i_ii_vals = list(
            more_itertools.flatten(map(_flatten_pos, enumerate(vals)))
        )
        positions = list(map(lambda tp: tp[0], i_ii_vals))
        items = tuple(map(lambda tp: tp[1], i_ii_vals))
        return positions, items
        
    def _unflatten(
        self, positions: List[List[int]], items: Sequence[torch.Tensor]
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

    def _generate_vjp_subgraph_using_torchfunc(
        self,
        function: Callable,
        dfb: Differentiability,
        raw_normalized_kwargs: Dict[str, FxArg],
    ) -> Tuple[
        GraphModule,
        List[List[int]],
        Dict[str, Node]
    ]:
        jvp_diff_primal_vals: \
            Dict[str, Union[torch.Tensor, Sequence[torch.Tensor]]] = {}

        for diff_param in dfb.diffable_params:
            diff_param: str
            raw_node_diff_arg = raw_normalized_kwargs[diff_param]

            if not isinstance(raw_node_diff_arg, (Node, Sequence)):
                # const scalars
                assert isinstance(raw_node_diff_arg, (int, float, str))
                
                # torch.vjp requires all positional inputs are Tensors
                diff_argval = torch.zeros([], dtype=torch.float32)

                jvp_diff_primal_vals[diff_param] = diff_argval

            else:  # Node or Node list

                # for a list-typed arg, we assume no mix of Node and scalar.
                jvp_diff_primal_vals[diff_param] = \
                    create_zero_arg_val(raw_node_diff_arg)  # type: ignore
                
        nondiff_val2raw: Dict[torch.Tensor, Node] = {}

        jvp_nondiff_env_vals: Dict[str, Union[torch.Tensor, FxConst]] = {}
        for nondiff_param_name, default_arg in dfb.other_params:
            raw_node_nondiff_arg = raw_normalized_kwargs[nondiff_param_name]

            if isinstance(raw_node_nondiff_arg, Sequence):
                if function is getitem_aux_kw:
                    index_nodes = collect_meta(raw_node_nondiff_arg, leaf_type=Node)
                    if len(index_nodes) != 0:
                        raise NotImplementedError(
                            "Using indices like `:self.i` in getitem is not supported"
                        )
                    jvp_nondiff_val = raw_node_nondiff_arg

                else:
                    raise NotImplementedError("Nested non-differentiable arg")
                    # PyTorch unlikely has this.
            
            elif isinstance(raw_node_nondiff_arg, Node):
                # Will result in GET_ATTR[tensor_contants0] Nodes in subgraph,
                # we can rely on the identity of this nondiff_val and the
                # attrname like "_tensor_constants0" to connect JVP-graph
                # nondiff-arg Nodes and the GET_ATTR Nodes in the subgraph.
                jvp_nondiff_val = create_zero_arg_val(raw_node_nondiff_arg)
                assert isinstance(jvp_nondiff_val, torch.Tensor)

                nondiff_val2raw[jvp_nondiff_val] = raw_node_nondiff_arg

            else:
                assert raw_node_nondiff_arg is None \
                    or isinstance(raw_node_nondiff_arg, (int, float, str, slice))
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

        # TODO if some outputs are not diff-able, like the index of torch.sort
        # when creating the _primal_func below, an extra getitem will be added
        # to the primal result.
        # TODO Additionally, we need to pick all primal getitem Nodes in the
        # torch.vjp subgraph.
        if isinstance(dfb.output_differentiability, Sequence):
            if not all(dfb.output_differentiability):
                raise NotImplementedError(
                    "EASIER VJP currently does not support"
                    " partially non-differentiable outputs like torch.sort"
                )

        # NOTE strictly speaking a cotangent is not the exactly same as an arg,
        # because a cotangent is isomorphic to the result of a Node, which
        # may be NESTED, but an arg will never be nested -- but many args may
        # form a nested structure in the arg list.
        cotangent_val = self._create_zero_cotangent(self.current_node)

        flattened_primal_tree, flattened_primal_vals = \
            self._flatten(diff_primal_vals)


        # Other non-differentiable parameters must NOT be passed via jvp()
        # API param list, but via function closure.

        is_aten_api = 'aten' in function.__module__

        def _primal_func(*flattened_primals: torch.Tensor):
            # There preparations are not part of FX Proxy and won't be traced
            unfltd_primals = self._unflatten(
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
        
        def _vjp(
            flattened_primals: Tuple[torch.Tensor, ...],
            cotangent: Union[torch.Tensor, Sequence[torch.Tensor]]
        ):
            y, vjpfun = torch.func.vjp(_primal_func, *flattened_primals) # type: ignore
            res_cot = vjpfun(cotangent) # type: ignore
            return y, res_cot
        
        gm: GraphModule = make_fx(_vjp)(
            flattened_primal_vals, cotangent_val
        )

        jvp_nondiff_attrname2raw: Dict[str, Node] = {}
        for const_name, nondiff_val in gm.named_buffers():
            jvp_nondiff_attrname2raw[const_name] = \
                nondiff_val2raw[nondiff_val]
        
        # TODO return bijective position mapping
        return gm, flattened_primal_tree, jvp_nondiff_attrname2raw
    

class _TorchVjpSubGraphPrimalPartCopier(EasierInterpreter):
    def __init__(
        self, subgm: GraphModule, subg: Graph, vjp_graph: Graph,
        #
        # Structural info about diff-able inputs
        #
        input_flatten_tree: List[List[int]],
        vjp_input_primals: List[Union[Node, Sequence[Node]]],
        
        # jvp_input_tangents: List[Union[Node, Sequence[Node]]],

        #
        # Structural info about outputs
        #
        n_flattened_primal_outputs: int,
        raw_node: Node,

        #
        # Structural info about non-diff-able inputs
        #
        vjp_input_nondiff_attrname2node: Dict[str, Node]
    ) -> None:
        super().__init__([subgm], [subg])  # type: ignore

        self.vjp_graph = vjp_graph

        self.input_flatten_tree = input_flatten_tree
        self.vjp_input_primals = vjp_input_primals

        # self.jvp_input_tangents = jvp_input_tangents

        self.raw_node = raw_node


        self.n_flattened_primal_inputs = len(input_flatten_tree)

        self.n_flattened_primal_outputs =  n_flattened_primal_outputs

        # Keys look like _tensor_constant0
        self.vjp_input_nondiff_attr2node = vjp_input_nondiff_attrname2node


        self._placeholder_i = 0  # totally 2*len(diff_arg_names)

        # TODO as JvpTransformer._prepare_diffable_primals_and_tangents, we
        # allow constants, but the subgraph assumes all inputs are Nodes,
        # making it erroneous to call torch function/method on constants.
        self.nodemap_subg2vjp: Dict[Node, Union[Node, FxConst]] = {}

        self.output_primal: Union[Node, Sequence[Node]]

        self.cotangent_placeholders: List[Node] = []
        self.cotangent_part_start: Node


    def run(self):
        for i, (root, graph) in enumerate(zip(self.modules, self.graphs)):
            self.current_module = root
            self.current_graph = graph
            self.current_module_index = i

            # Before traversing, we fix the nodes by copying them into a list,
            # in case the customized handler modifies the `graph.nodes` view.
            nodes = list(graph.nodes)

            assert not self.reverse
            
            has_met_primal_output = False
            output_i = 0

            for node in nodes:
                self.current_node = node
                self.for_each_node()

                for user in node.users:
                    if user.op == FX.OUTPUT:
                        if user.args[0][0] is node:
                            # Must be 0-th, because sometimes in vjp subgraph
                            # an argument cotangent will be directly returned,
                            # e.g. A + B => dA = d; dB = d
                            has_met_primal_output = True

                if has_met_primal_output:
                    output_i += 1

                if output_i >= self.n_flattened_primal_outputs:
                    break
            
            self.cotangent_part_start = node.next

            # Because sequential handling has been broken, we need to
            # manually handle the OUTPUT Node here.
            self._record_output(self.current_graph.output_node())

        return self
        
    def if_placeholder(self, param_name: str):
        """
        Same as torch.func.jvp requirements, the PLACEHOLDER Nodes only differ
        in their positions and the order.

        All placeholders are formed by first flattening the args of the raw
        Node, e.g. torch.cat([A,B,C]) will have 3 placeholders A, B, C.
        """
        # param_name would be "primals_1" "tangents_2" (from `def _jvp` above)
        # and not usable.
        is_primal = self._placeholder_i < self.n_flattened_primal_inputs

        def _from_flatten(
            jvp_inputs: List[Union[Node, Sequence[Node]]],
            tree: List[int]
        ) -> Union[Node, FxConst]:
            if len(tree) == 1:
                [i] = tree
                return jvp_inputs[i]  # type: ignore
            else:
                [i, ii] = tree
                return jvp_inputs[i][ii] # type: ignore

        if is_primal:
            ph_pos = self._placeholder_i

            vjp_primal = _from_flatten(
                self.vjp_input_primals, self.input_flatten_tree[ph_pos]
            )
            self.nodemap_subg2vjp[self.current_node] = vjp_primal
        else:
            ph_pos = self._placeholder_i - self.n_flattened_primal_inputs

            self.cotangent_placeholders.append(self.current_node)

        self._placeholder_i += 1
    
    def if_get_attr(self, submod_path: str, attr_name: str, attr_val):
        assert submod_path == ''
        # attr_name looks like _tensor_constants0

        self.nodemap_subg2vjp[self.current_node] = \
            self.vjp_input_nondiff_attr2node[attr_name]
    
    def _strict_copy_subg_to_jvp(self, subg_node: Node):
        for subg_input in subg_node.all_input_nodes:
            assert subg_input in self.nodemap_subg2vjp, \
                "Sub-Graph Node's Node input must be in subg2jvp Node map"
        
        jvp_subgraph_out = cast(
            Sequence[Node], self.current_graph.output_node().args[0]
        )
        is_output = subg_node in \
            jvp_subgraph_out[:self.n_flattened_primal_outputs]

        if is_output:
            name = f'{self.raw_node.name}_vjp'
        else:
            name = f'{self.raw_node.name}_vjp_{subg_node.name}'
        
        ng = get_node_tensor_group(self.raw_node)
        if ng is not None:
            # Sometimes literal batch size 1000 (we picked for fake vjp run)
            # appears in the arg list.
            def _pick_dist_input(x: Node):
                if get_node_meta(x).role == Role.DISTRIBUTED:
                    return x
                else:
                    return None
            dist_primal_input = collect_meta(
                self.vjp_input_primals, leaf_type=Node
            )[0]

            bs_cache = []

            def _map_with_bs(x):
                if isinstance(x, Node):
                    return self.nodemap_subg2vjp[x]
                elif x == 1000:
                    if len(bs_cache) > 0:
                        return bs_cache[0]

                    # NOTE get the symbolic size int from a dist input to support
                    # partitioning, because each rank will have different shape[0].
                    # NOTE this value should be treated as REPLICA but is not the same
                    # across ranks!
                    bs = self.vjp_graph.call_method(
                        'size', (dist_primal_input, 0,)
                    )
                    bs_cache.append(bs)

                    return bs
                else:
                    return x
            
            args = tree_map(subg_node.args, _map_with_bs)
            kwargs = { k: tree_map(v, _map_with_bs) for k,v in subg_node.kwargs.items() }

            vjp_node = self.vjp_graph.create_node(
                subg_node.op, subg_node.target, args, kwargs,
                name,
                subg_node.type
            )
        
        else:
            args = tree_map(subg_node.args, self.nodemap_subg2vjp.__getitem__)
            kwargs = {
                k: tree_map(v, self.nodemap_subg2vjp.__getitem__)
                for k,v in subg_node.kwargs.items()
            }

            vjp_node = self.vjp_graph.create_node(
                subg_node.op, subg_node.target, args, kwargs,
                name,
                subg_node.type
            )

        return vjp_node
    
    def if_call_function(self, function):
        jvp_node = self._strict_copy_subg_to_jvp(
            self.current_node
        )

        self.nodemap_subg2vjp[self.current_node] = jvp_node
    

    def _record_output(self, subg_output: Node):
        jvp_subgraph_out = cast(Sequence[Node], subg_output.args[0])

        convert = self.nodemap_subg2vjp.__getitem__

        if self.n_flattened_primal_outputs == 1:
            self.output_primal = tree_map(jvp_subgraph_out[0], convert)
        else:
            self.output_primal = tree_map(
                jvp_subgraph_out[:self.n_flattened_primal_outputs], convert
            )
    
    def if_output(self):
        raise EasierJitException(
            "Primal part copier should not meet OUTPUT Node"
        )
    
    def if_call_method(self, method_name: str):
        raise NotImplementedError(
            "It's unlikely that torch.func.jvp generates method calls"
        )
    
    def if_call_module(self, submod: Module):
        raise NotImplementedError(
            "It's unlikely that torch.func.jvp generates module calls"
        )




class _TorchVjpSubGraphCotangentPartCopier(EasierInterpreter):
    """
    CotangentPartCopiers are run in the reversed order.
    """

    def __init__(
        self, subgm: GraphModule, subg: Graph, vjp_graph: Graph,
        primal_copier: _TorchVjpSubGraphPrimalPartCopier,
        #
        # Structural info about diff-able inputs
        #
        # input_flatten_tree: List[List[int]],
        # vjp_input_primals: List[Union[Node, Sequence[Node]]],

        # jvp_input_tangents: List[Union[Node, Sequence[Node]]],
        input_cotangent: Union[Node, Sequence[Node]],
        
        output_flatten_tree: List[List[int]],

        #
        # Structural info about outputs
        #
        # n_flattened_primal_outputs: int,
        #
        # Structural info about non-diff-able inputs
        #
        vjp_input_nondiff_attrname2node: Dict[str, Node],

        nodemap_primal2sumcot: Dict[Node, Node],

        prune_diff_args: List[int]
    ) -> None:
        super().__init__([subgm], [subg])  # type: ignore

        self.primal_copier = primal_copier

        self.vjp_graph = vjp_graph

        # self.input_flatten_tree = input_flatten_tree
        # self.vjp_input_primals = vjp_input_primals

        # self.jvp_input_tangents = jvp_input_tangents
        self.input_cotangent = input_cotangent

        self.output_flatten_tree = output_flatten_tree

        # self.n_flattened_primal_inputs = len(input_flatten_tree)

        self.n_flattened_primal_outputs =  len(output_flatten_tree)

        # Keys look like _tensor_constant0
        self.vjp_input_nondiff_attr2node = vjp_input_nondiff_attrname2node

        self.nodemap_primal2sumcot: Dict[Node, Node] = nodemap_primal2sumcot

        self.prune_diff_args: List[int] = prune_diff_args
        self.only_for_pruned_cots: Set[Node] = set()

    def _from_flatten(
        self,
        jvp_inputs: List[Union[Node, Sequence[Node]]],
        tree: List[int]
    ) -> Node:
        if len(tree) == 1:
            [i] = tree
            return jvp_inputs[i]  # type: ignore
        else:
            [i, ii] = tree
            return jvp_inputs[i][ii] # type: ignore
    
    def _prune_subg(self):
        """
        TODO there will ultimately be no such pruning process, because:

        -   Currently we only respect the differentiability of parameters
            on API-level, we didn't take the real arguments into consideration
            e.g. the argument may be a const scalar or torch.Tensor.
            So to each diff-able parameter torch.vjp will generate cotangent
            computations.

            The better approach is to filter those arguments by the value,
            send to _primal_func as closure.
            `_generate_subgraph_using_torch_vjp` already has similar
            functionalities.

        -   Currently, also as a bad side-effect of the #1 point, e.g.
            `dist * replica` will lead torch.vjp to generate a torch.aten.sum
            to aggregate cotangent for `replica`.
            In the current setting this can be pruned.

            However, pruning is not the correct way to handle this, as
            dist+replica operations are ubiquitous.

            The better approach is to handle torch's broadcasting semantics
            ourselves:
            1.  Insert an explicit `torch.expand_as` call to replica arguments;
            2.  For BP, insert an `esr.sum` aggregator for that `expand_as`.
        """
        vjp_subgraph_out = cast(Sequence[Node], self.current_graph.output_node().args[0])
        out_cots = vjp_subgraph_out[self.n_flattened_primal_outputs:]

        for arg_i in self.prune_diff_args:
            self.only_for_pruned_cots.add(out_cots[arg_i])
        
        for node in reversed(list(self.current_graph.nodes)):
            node: Node
            if node.op == FX.OUTPUT:
                continue
            
            if all(user in self.only_for_pruned_cots for user in node.users):
                self.only_for_pruned_cots.add(node)

    def run(self):
        for cot_ph_pos, cot_ph in enumerate(self.primal_copier.cotangent_placeholders):
            vjp_in_cot = self._from_flatten([self.input_cotangent], self.output_flatten_tree[cot_ph_pos])
            self.primal_copier.nodemap_subg2vjp[cot_ph] = vjp_in_cot

        for i, (root, graph) in enumerate(zip(self.modules, self.graphs)):
            self.current_module = root
            self.current_graph = graph
            self.current_module_index = i

            self._prune_subg()

            # Before traversing, we fix the nodes by copying them into a list,
            # in case the customized handler modifies the `graph.nodes` view.
            nodes = list(graph.nodes)

            assert not self.reverse
            
            start_pos = nodes.index(self.primal_copier.cotangent_part_start)
            nodes = nodes[start_pos:]

            for node in nodes:
                self.current_node = node

                # Only prune for cotangent part, not for primal part.
                if node not in self.only_for_pruned_cots:
                    self.for_each_node()
        
        return self
    

    def if_get_attr(self, submod_path: str, attr_name: str, attr_val):
        assert submod_path == ''
        # attr_name looks like _tensor_constants0

        self.primal_copier.nodemap_subg2vjp[self.current_node] = \
            self.vjp_input_nondiff_attr2node[attr_name]

    def if_call_function(self, function):
        jvp_node = self.primal_copier._strict_copy_subg_to_jvp(
            self.current_node
        )

        self.primal_copier.nodemap_subg2vjp[self.current_node] = jvp_node
    
    def if_output(self):
        """
        After traversing and copying all cotangent Nodes, we need to handle:

        -   The aggregation of components of cotangents for each primal value.
        -   Maintain the primal-Node-cotangent-Node
            and primal-Tensor-cotangent-Tensor relationships in VJP module.
        """
        vjp_subgraph_out = cast(Sequence[Node], self.current_node.args[0])
        def _convert(subg_node: Node):
            if subg_node in self.primal_copier.nodemap_subg2vjp:
                return self.primal_copier.nodemap_subg2vjp[subg_node]
            else:
                if subg_node not in self.only_for_pruned_cots:
                    raise EasierJitException(
                        "Expected a torch.vjp Node that has been either copied"
                        " or determined as pruned"
                    )
                return None
        
        vjp_out_cots = tree_map(
            vjp_subgraph_out[self.n_flattened_primal_outputs:],
            _convert
        )

        for i in range(self.primal_copier.n_flattened_primal_inputs):
            vjp_primal = self._from_flatten(
                self.primal_copier.vjp_input_primals,
                self.primal_copier.input_flatten_tree[i]
            )
            vjp_out_cot = vjp_out_cots[i]

            if vjp_out_cot is None:  # has been pruned
                continue

            if vjp_primal in self.nodemap_primal2sumcot:
                vjp_out_cot_comp = self.nodemap_primal2sumcot[vjp_primal]
                vjp_out_cot = self.vjp_graph.call_function(
                    torch.add,
                    (vjp_out_cot_comp, vjp_out_cot)
                )
        
            self.nodemap_primal2sumcot[vjp_primal] = vjp_out_cot

    
    def if_call_method(self, method_name: str):
        raise NotImplementedError(
            "It's unlikely that torch.func.jvp generates method calls"
        )
    
    def if_call_module(self, submod: Module):
        raise NotImplementedError(
            "It's unlikely that torch.func.jvp generates module calls"
        )
    

class Vjp(esr.Module):
    # TODO the resultant module Vjp and Jvp are basically the same,
    # we may have a common base class like _VpBase
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

            if len(vectors) != len(outputs):
                raise ValueError(
                    f"The number of vectors {len(vectors)} does not match the"
                    f" number of outputs {len(outputs)}"
                )
            
            for i, v in zip(outputs, vectors):
                if i.dtype != v.dtype or i.shape != v.shape:
                    raise ValueError(
                        "Vector's dtype/shape does not match the output"
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
                    esr.zeros_like(output),
                    mode=('partition' if output.is_partition else 'replicate')
                )
                for output in outputs
            ))
        

        self.products = cast(Sequence[esr.Tensor], torch.nn.ParameterList(
            esr.Tensor(
                esr.zeros_like(input),
                mode=('partition' if input.is_partition else 'replicate')
            )
            for input in inputs
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
        

def vjp(
    module: esr.Module,
    inputs: Sequence[esr.Tensor],
    outputs: Sequence[esr.Tensor],
    vectors: Optional[Sequence[esr.Tensor]] = None
) -> Vjp:
    """
    Args:
    -   inputs: primal input Tensors of `module`, cannot be written.
    -   outputs: primal output Tensors of `module`, can only be written and written once.
    -   vectors: User-defined cotangent Tensors for primal Tensors `outputs`.
    """

    class _Vjp(Vjp):
        _raw_module_class = module.__class__.__qualname__

        def forward(self):
            # This (as well as the graph_copy in coll_init) unconditionally
            # initializes products to 0. However, some products actually have
            # results so they are written twice, breaking the requirement of
            # repeated VJP.
            # TODO make only those products that are not bound to
            # primal computation zeros.
            for p in self.products:
                p[:] = 0
            
            self.graph_module()
    
    vjpm = _Vjp(inputs, outputs, vectors)
    vjp_transformer = VjpTransformer(module, vjpm).run()

    gm = GraphModule(vjpm, vjp_transformer.vjp_graph)
    vjpm.graph_module = gm

    return vjpm
