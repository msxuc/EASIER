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
    FX, EasierInterpreter, SubmodNameAllocator, normalize_reducer_call_into_args, \
    fx_normalize_function_variant_into_kwargs, tree_map, \
    normalize_selector_call_into_args, get_attr_value

from easier.core.runtime.metadata import \
    Role, RuntimeTensorMeta, StructuredTensorMeta, \
    collect_meta, set_node_meta, get_node_meta
    
from easier.core.autodiff.jvp import Jvp

from easier.core.autodiff.autodiff_rule import \
    Differentiability, RequiredParam, \
    diff_rule_registry, differentiabilities
from easier.core.autodiff.utils import PrimalMetaPropagator, simplify_torchfunc_fx_graph, FxConst
from easier.core.utils import EasierJitException


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
        self.jvp_tensors_attrpaths: Dict[esr.Tensor, str] = {}

        for i, (input, vector) in enumerate(zip(
            vjp_module.inputs, vjp_module.vectors
        )):
            self.tensormap_primal2tangent[input] = vector

            self.jvp_tensors_attrpaths[input] = f'inputs.{i}'
            self.jvp_tensors_attrpaths[vector] = f'vectors.{i}'


        for i, (output, product) in enumerate(zip(
            vjp_module.outputs, vjp_module.products
        )):
            self.tensormap_primal2tangent[output] = product

            self.jvp_tensors_attrpaths[output] = f'outputs.{i}'
            self.jvp_tensors_attrpaths[product] = f'products.{i}'
        

        self._setup_vjp_module_attrs()

        # Attr name for ()-shape constant Tensors created for literal Scalar
        # arguments, those constants are made attributes to make them aware
        # of AOT target device like CUDA.
        self.const_name_allocator = SubmodNameAllocator('const')

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
        pass


    def if_call_function(self, function: Callable) -> None:
        if function is operator.setitem:
            # A special syntactic Node for VJP to write the resultant primal
            # Tensors.
            container, index, value = self.current_node.args
            assert isinstance(container, Node)

            if index != Ellipsis:
                raise NotImplementedError(
                    "EASIER VJP currently only supports setitem with Ellipsis"
                )
            
            if container.op != FX.GET_ATTR:
                raise NotImplementedError(
                    "EASIER VJP currently only supports setitem to"
                    " GET_ATTR new_value Tensors"
                )
            
            container_val = get_attr_value(self.raw_module, container)
            if container_val not in self.vjp_module.outputs:
                # outputs are subset of new_values, not all new_values are
                # bound with cotangents.
                raise EasierJitException(
                    "Primal setitem can only write to output Tensors"
                )
                

            return

        if function is operator.getitem:
            container = self.current_node.args[0]
            assert isinstance(container, Node)
            imeta = get_node_meta(container)

            if isinstance(imeta, Sequence):
                # This is the raw Node for unpacking a tuple, has been handled
                # when handling the multi-res raw Node.
                return

        self._handle_operation(function)
    

    def _is_cotangent_involved(self) -> bool:
        1

    

    def _handle_operation(self, function: Callable):
        if function in diff_rule_registry:
            rule_cls = diff_rule_registry[function]
            rule = rule_cls(
                self.current_node, function, self._fake_eval_meta_ctor
            )

            #
            # primal
            #
        
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

            raw_node_diff_res: Sequence[
                Union[FxConst, Node, Sequence[Node]]
            ] = [
                1 for p in dfb.output_differentiability
            ]


            cotangent_involved = self._is_cotangent_involved(raw_node_diff_args)

            raw_node_diff_args: Dict[
                str, Union[FxConst, Node, Sequence[Node]]
            ] = {
                p: raw_node_normalized_kwargs[p] for p in dfb.diffable_params
            }  # type: ignore


    def _strict_map_raw_to_vjp(self, raw: Node, nodemap: Dict[Node, Node]):
        """
        fx.Graph.node_copy tends to silently hide raw-not-existing error in
        nodemap.__getitem__ arg transformation (in torch C++ level),
        silently returning a None constant in the VJP Graph.

        This aux method validates the consistency of both nodemaps
        in this class.
        """
        assert nodemap is self.nodemap_raw2primal \
            or nodemap is self.nodemap_raw2tangent

        for raw_input in raw.all_input_nodes:
            assert raw_input in self.nodemap_raw2primal, \
                "Raw Node's Node input must be in raw-primal Node map"
        
        vjp_node = self.vjp_graph.node_copy(
            raw, arg_transform=nodemap.__getitem__
        )

        return vjp_node   

class Vjp(Jvp):
    # TODO the resultant module Vjp and Jvp are basically the same,
    # we may have a common base class like _VpBase
    pass

def vjp(
    module: esr.Module,
    inputs: Sequence[esr.Tensor],
    new_values: Sequence[esr.Tensor],
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
    
    vjpm = _Vjp(inputs, outputs, vectors)
    vjp_transformer = VjpTransformer(module, vjpm).run()

    gm = GraphModule(vjpm, vjp_transformer.vjp_graph)
    vjpm.graph_module = gm

    return vjpm
