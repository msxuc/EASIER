# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import operator
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, TypeAlias, TypeVar, Union, cast

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
    diff_rule_registry, differentiabilities
from easier.core.autodiff.utils import PrimalMetaPropagator, simplify_torchfunc_fx_graph, FxConst
from easier.core.utils import EasierJitException

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
        [module], [graph] = collectively_initialize_and_validate([module])
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
        primal_attrname = self.const_name_allocator.alloc_name(
            self.vjp_module, attrname_hint
        )
        setattr(self.vjp_module, primal_attrname, bp)

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

        Always on CPU.
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
        primal = self.nodemap_raw2primal[self.current_node]
        if primal in self.nodemap_primal2sumcot:
            cot = self.nodemap_primal2sumcot[primal]
            if attr_val in self.tensormap_primal2cotangent:
                grad_tensor = self.tensormap_primal2cotangent[attr_val]
                path = self.vjp_tensors_attrpaths[grad_tensor]
                grad_node = self.vjp_graph.get_attr(path)
                self.vjp_graph.call_function(operator.setitem, (grad_node, (slice(None),), cot))



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

            if container_val not in self.vjp_module.new_values:
                # outputs are subset of new_values, not all new_values are
                # bound with cotangents.
                raise EasierJitException(
                    "Primal setitem can only write to new_values Tensors"
                )
            
            if container_val in self.vjp_module.outputs:
                cot_tensor = self.tensormap_primal2cotangent[container_val]

                cot_attrname = self.vjp_tensors_attrpaths[cot_tensor]
                cot_node = self.vjp_graph.get_attr(cot_attrname)

                # Writes to `output` is the final use of the `value` Node.
                self.nodemap_primal2sumcot[value] = cot_node

            return

        if function is operator.getitem:
            container, index = self.nodemap_raw2primal[self.current_node].args
            assert isinstance(container, Node)
            imeta = get_node_meta(container)

            if isinstance(imeta, Sequence):
                # This is the raw Node for unpacking a tuple, has been handled
                # when handling the multi-res raw Node.
                return

        self._handle_operation(function)
    

    def _try_prepare_get_cotangent(self, dfb: Union[DiffRuleBase, Differentiability]) -> Union[Node, Sequence[Node], None]:
        primal_node = self.nodemap_raw2primal[self.current_node]

        if isinstance(dfb, Differentiability):
            out_diff = dfb.output_differentiability
        else:
            dfb.input_differentiability()
            out_diff = dfb.output_differentiability()

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
                assert all(comp is None for comp in comps), "TODO"
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
                primal_const_tensor_attrname = self._ensure_vjp_const_attr(
                    torch.full((), raw_node_diff_arg, dtype=torch.float32),
                    f"{function.__name__}_{argname}"
                )
                input_primal_nodes.append(
                    self.vjp_graph.get_attr(primal_const_tensor_attrname)
                )
            
            else:  # Node or Node list
                input_primal_nodes.append(tree_map(
                    raw_node_diff_arg, self.nodemap_raw2primal.__getitem__
                ))


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
            
            raw_node_diff_args = rule.input_differentiability(
                *self.current_node.args, **self.current_node.kwargs
            )
     
            input_primal_nodes = self._prepare_diffable_primals(
                function, raw_node_diff_args
            )

            rule.inject_vjp_subgraph()

        
        else: # not in diff_rule_registry
            if getattr(operator, function.__name__, None) is function:
                if function is operator.truediv:
                    function = torch.div  # torch does not have truediv
                else:
                    function = getattr(torch, function.__name__)
                
                # Unlike JVP, we do not need to care literal args in VJP.
            
            if function not in differentiabilities:
                raise NotImplementedError(
                    f"Operator {function} is not registered in EASIER AutoDiff"
                )

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

            primal = self.nodemap_raw2primal[self.current_node]
            
            # Edit VJP primal Node
            with self.vjp_graph.inserting_after(primal):
                input_primal_nodes = self._prepare_diffable_primals(
                    function, raw_node_diff_args
                )

                primal_copier = _TorchVjpSubGraphPrimalPartCopier(
                    gm, subg, self.vjp_graph,
                    flattened_input_tree, input_primal_nodes,
                    len(flattened_output_tree),
                    self.current_node,
                    nondiff_inputs_attrname2vjp
                ).run()

                # TODO replace IO Nodes
                raise NotImplementedError("replace primal IO Node")

            # Resume to insert at the end of VJP graph
            cotangent_copier = _TorchVjpSubGraphCotangentPartCopier(
                gm, subg, self.vjp_graph, primal_copier,
                cotangent,
                flattened_output_tree,
                nondiff_inputs_attrname2vjp,
                self.nodemap_primal2sumcot
            ).run()

            # if isinstance(out_diff, bool):
            #     assert isinstance(primal_copier.output_primal, Node)
            #     assert out_diff == True
            #     self.nodemap_raw2primal[self.current_node] = \
            #         primal_copier.output_primal
            
            # else:
            #     assert isinstance(primal_copier.output_primal, Sequence)
            #     assert isinstance(out_diff, Sequence)

            #     # NOTE it's likely the RAW multi-res Node doesn't have
            #     # explicit primal/tangent counterpart Nodes, so we can only
            #     # set up binding on the unpacking getitem Nodes.

            #     for raw_getitem in self.current_node.users:
            #         assert raw_getitem.target is operator.getitem
            #         _, item_i = raw_getitem.args
            #         assert isinstance(item_i, int)

            #         self.nodemap_raw2primal[raw_getitem] = \
            #             primal_copier.output_primal[item_i]
            # # endif out_diff

            # self.cotangent_part_generators.append(cotangent_copier)


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
                    self._create_zero_val(raw_node_diff_arg)  # type: ignore
                
        nondiff_val2raw: Dict[torch.Tensor, Node] = {}

        jvp_nondiff_env_vals: Dict[str, Union[torch.Tensor, FxConst]] = {}
        for nondiff_param_name, default_arg in dfb.other_params:
            raw_node_nondiff_arg = raw_normalized_kwargs[nondiff_param_name]

            if isinstance(raw_node_nondiff_arg, Sequence):
                raise NotImplementedError("Nested non-differentiable arg")
                # PyTorch unlikely has this.
            
            if isinstance(raw_node_nondiff_arg, Node):
                # Will result in GET_ATTR[tensor_contants0] Nodes in subgraph,
                # we can rely on the identity of this nondiff_val and the
                # attrname like "_tensor_constants0" to connect JVP-graph
                # nondiff-arg Nodes and the GET_ATTR Nodes in the subgraph.
                jvp_nondiff_val = self._create_zero_val(raw_node_nondiff_arg)
                assert isinstance(jvp_nondiff_val, torch.Tensor)

                nondiff_val2raw[jvp_nondiff_val] = raw_node_nondiff_arg

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

        # y_meta = get_node_meta(self.current_node)
        # y_positions, y_meta_items = self._flatten([y_meta], leaf_type=RuntimeTensorMeta)

        cotangent_vals = [self._create_zero_val(self.current_node)]

        flattened_primal_tree, flattened_primal_vals = \
            self._flatten(diff_primal_vals)

        # flattened_tangent_tree, flattened_tangent_vals = \
        #     _flatten(tangent_vals)
        # assert flattened_primal_tree == flattened_tangent_tree

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
            cotangents: Tuple[torch.Tensor, ...]
        ):
            y, vjpfun = torch.func.vjp(_primal_func, *flattened_primals) # type: ignore
            res_cot = vjpfun(*cotangents) # type: ignore
            return y, res_cot
        
        gm: GraphModule = make_fx(_vjp)(
            flattened_primal_vals, cotangent_vals
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
                    has_met_primal_output = has_met_primal_output or user.op == FX.OUTPUT

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
    
    def _strict_map_subg_to_jvp(self, subg_node: Node):
        for subg_input in subg_node.all_input_nodes:
            assert subg_input in self.nodemap_subg2vjp, \
                "Sub-Graph Node's Node input must be in subg2jvp Node map"
        
        ng = get_node_tensor_group(self.raw_node)
        if ng is not None:
            # Sometimes literal batch size 1000 (we picked for fake vjp run)
            # appears in the arg list.
            bs = ng.n

            def _map_with_bs(x):
                if isinstance(x, Node):
                    return self.nodemap_subg2vjp[x]
                elif x == 1000:
                    return bs
                else:
                    return x
            
            args = tree_map(subg_node.args, _map_with_bs)
            kwargs = { k: tree_map(v, _map_with_bs) for k,v in subg_node.kwargs.items() }
            vjp_node = self.vjp_graph.create_node(
                subg_node.op, subg_node.target, args, kwargs, subg_node.name, subg_node.type
            )
        
        else:
            vjp_node = self.vjp_graph.node_copy(
                subg_node,
                arg_transform=self.nodemap_subg2vjp.__getitem__  # type: ignore
            )

        return vjp_node
    
    def if_call_function(self, function):
        jvp_node = self._strict_map_subg_to_jvp(
            self.current_node
        )

        self.nodemap_subg2vjp[self.current_node] = jvp_node
    
    def _record_output(self, output: Node):
        jvp_subgraph_out = cast(Sequence[Node], output.args[0])

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

        nodemap_primal2sumcot: Dict[Node, Node]
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

    def run(self):
        
        for cot_ph_pos, cot_ph in enumerate(self.primal_copier.cotangent_placeholders):
            vjp_in_cot = self._from_flatten([self.input_cotangent], self.output_flatten_tree[cot_ph_pos])
            self.primal_copier.nodemap_subg2vjp[cot_ph] = vjp_in_cot

        for i, (root, graph) in enumerate(zip(self.modules, self.graphs)):
            self.current_module = root
            self.current_graph = graph
            self.current_module_index = i

            # Before traversing, we fix the nodes by copying them into a list,
            # in case the customized handler modifies the `graph.nodes` view.
            nodes = list(graph.nodes)

            assert not self.reverse
            
            start_pos = nodes.index(self.primal_copier.cotangent_part_start)
            nodes = nodes[start_pos:]

            for node in nodes:
                self.current_node = node
                self.for_each_node()
        
        return self
    

    def if_get_attr(self, submod_path: str, attr_name: str, attr_val):
        assert submod_path == ''
        # attr_name looks like _tensor_constants0

        self.primal_copier.nodemap_subg2vjp[self.current_node] = \
            self.vjp_input_nondiff_attr2node[attr_name]

    def if_call_function(self, function):
        jvp_node = self.primal_copier._strict_map_subg_to_jvp(
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
        convert = self.primal_copier.nodemap_subg2vjp.__getitem__
        
        vjp_out_cots = tree_map(vjp_subgraph_out[self.n_flattened_primal_outputs:], convert)

        for i in range(self.primal_copier.n_flattened_primal_inputs):
            vjp_primal = self._from_flatten(
                self.primal_copier.vjp_input_primals, self.primal_copier.input_flatten_tree[i]
            )
            vjp_out_cot = vjp_out_cots[i]

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
        new_values: Sequence[esr.Tensor],
        inputs: Sequence[esr.Tensor],
        outputs: Sequence[esr.Tensor],
        vectors: Optional[Sequence[esr.Tensor]] = None
    ):
        super().__init__()
        
        self.new_values = new_values

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
    new_values: Sequence[esr.Tensor],
    inputs: Sequence[esr.Tensor],
    outputs: Sequence[esr.Tensor],
    vectors: Optional[Sequence[esr.Tensor]] = None
) -> Vjp:
    """
    Args:
    -   inputs: primal input Tensors of `module`, cannot be written.
    -   new_values:
            output Tensors to store the primal output values,
            cannot be overlapping with `inputs`.
    -   outputs:
            subset of `new_values`, cotangents for this subset of outputs
            will be passed-in in the resultant VJP module.
    -   vectors: User-defined cotangent Tensors for primal Tensors `outputs`.
    """

    class _Vjp(Vjp):
        _raw_module_class = module.__class__.__qualname__

        def forward(self):
            for p in self.products:
                p.zero_()
            
            self.graph_module()
    
    vjpm = _Vjp(new_values, inputs, outputs, vectors)
    vjp_transformer = VjpTransformer(module, vjpm).run()

    gm = GraphModule(vjpm, vjp_transformer.vjp_graph)
    vjpm.graph_module = gm

    return vjpm
