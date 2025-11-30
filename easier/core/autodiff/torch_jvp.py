# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.




import dataclasses
import operator
from typing import Callable, Dict, List, Literal, Optional, Sequence, Set, Tuple, Type, Union
from typing_extensions import OrderedDict

import torch
from torch.fx import Node, Graph

import easier as esr
from easier.core.autodiff.autodiff import FxConst, FxArg
from easier.core.passes.utils import FX, fx_normalize_function_variant_into_kwargs, tree_map
from easier.core.runtime.metadata import Role, RuntimeTensorMeta, collect_meta, get_node_meta, set_node_meta


class RequiredParam:
    pass
required = RequiredParam()

@dataclasses.dataclass
class Differentiability:
    # TODO currently we assume the elements of an overloading lattice only
    # differ between parameter type Tensor/Scalar.
    # And each overloading lattice corresponds to a certain number of params.

    # The names for params that are differentiable.
    # Generally the param names are not ordered within this dataclass.
    diffable_params: List[str]

    # The second field is the default value.
    # NOTE A mandatory param without default value will still have the second
    # field be None, and will be overwritten by callsite non-None value.
    # TODO ensure no common ops whose default argument of a non-scalar param
    # (e.g. `float?`, `Tensor`) is not None.
    #
    # Generally the param names are not ordered within this dataclass.
    other_params: List[Tuple[str, Union[RequiredParam, FxConst]]] = dataclasses.field(default_factory=list)

    # TODO certain ops like aten::_to_copy has this field a function rather
    # than a constant, e.g.
    # `output_differentiability: ["!dtype || isDifferentiableType(*dtype)"]`
    output_differentiability: Union[Literal[True], List[bool]] = True

    kwargs_normalizer: Callable[[Callable, tuple, dict], Dict[str, FxArg]] = \
        fx_normalize_function_variant_into_kwargs

    def all_param_names(self) -> Set[str]:
        # For current handling of overloading resolution, a Set[str] suffices.
        params = set()
        params.update(self.diffable_params)
        params.update(k for k, v in self.other_params)
        return params
    

class DiffRuleBase:
    fx_normalize_to_kwargs_only: bool = True
    needs_result: bool = False

    # Only if normalize_to_kwargs_only.
    # Especially for 'self' param in torch yaml, use 'input' instead.
    diffable_params: Optional[List[str]] = None


    def input_differentiability(self) -> Dict[str, Union[FxConst, Node, Sequence[Node]]]:
        """
        Resolve inputs at the callsite, decide which inputs are differentiable.

        Unlike generating jvp sub-Graph using torch.func.jvp, EASIER DiffRule
        is purely symbolic, so derived Rule class does not need to convert
        FX scalars/constants to ()-shape tensor values.
        """
        # Default implementation:
        # - Resolve overloading via FX normalization;
        # - Decide input differentiability using simple param names list.
        assert self.diffable_params is not None
        assert self.fx_normalize_to_kwargs_only, \
            "Only FX-normalizable torch-Python/torch.ops.aten operators can" \
            " simply use `diff_params` field for differentiability of inputs"

        raw_kwargs = self.raw_normalized_kwargs

        assert len(set(self.diffable_params) - raw_kwargs.keys()) == 0, \
            "All `diff_params` must be present in callsite arguments"

        raw_diff_kwargs = raw_kwargs.fromkeys(self.diffable_params)
        assert list(raw_diff_kwargs.keys()) == self.diffable_params

        return raw_diff_kwargs  # type: ignore
            
    
    # def output_differentiability(self)->Union[Literal[True], List[bool]]:
    #     # Default implmentation:
    #     # - Single output and differentiable
        
    #     # TODO cover torch.to/aten._to_copy whose output diff is dynamic
    #     return True

    def output_meta(self, *args, **kwargs):
        """
        Calculate the metadata for raw/primal result since we only symbolically
        handle the Node without evaluate it.

        Tangent, if present, will share the same metadata.

        Inputs:
        -   RAW input Nodes or constants to the RAW Node

        Returns:
        -   RuntimeTensorMeta or a nested one
        """
        raise NotImplementedError("Derived class should implement this")

    
    def jvp(self, *args, **kwargs):
        """
        Inputs:
        -   EasierProxy for input Nodes
        -   Constants remain constants

        Returns:
        -   Tensor/Proxy: A single result item with tangent
        -   List[None | Tensor/Proxy]: Multiple items, some don't have tangent
        """
        raise NotImplementedError("Derived class should implement this")
    
    def __init__(self, node: Node, op: Callable, raw_meta_ctor: Callable) -> None:
        self.raw_node = node
        self.op = op
        self.raw_meta_ctor = raw_meta_ctor

        if self.fx_normalize_to_kwargs_only:
            self.raw_normalized_kwargs = \
                fx_normalize_function_variant_into_kwargs(
                    self.op, self.raw_node.args, self.raw_node.kwargs
                )
    
    def invoke_output_meta(self):
        if self.fx_normalize_to_kwargs_only:
            ometa = self.output_meta(**self.raw_normalized_kwargs)

        else:
            ometa = self.output_meta(*self.raw_node.args, **self.raw_node.kwargs)
        
        return ometa
        
    
    def inject_jvp_subgraph(
        self,
        # If multi-res op, this param including non-diffable result item.
        primal_result: Union[Node, Sequence[Node]],
        diff_input_names: List[str],
        raw2primal: Dict[Node, Node],
        tangents: List[Union[FxConst, Node, Sequence[Node]]]
    ) -> Union[Node, Sequence[Node]]:
        assert self.raw_node.op == FX.CALL_FUNCTION

        from easier.core.jit import EasierProxy, EasierTracer

        jvp_graph: Graph = collect_meta(
            primal_result, lambda n: n.graph, leaf_type=Node
        )[0]

        tracer = EasierTracer()
        tracer.graph = jvp_graph

        primal_result_proxies = []
        if self.needs_result:
            primal_proxy = tree_map(primal_result, lambda n: tracer.proxy(n))
            primal_result_proxies = [primal_proxy]

        kw_tangents = dict(zip(diff_input_names, tangents))

        if self.fx_normalize_to_kwargs_only:
            norm_kw_proxies = {
                k: tree_map(raw, tracer.proxy)
                for k, raw in self.raw_normalized_kwargs.items()
            }
            res_tangent = self.jvp(*primal_result_proxies, **norm_kw_proxies, **kw_tangents)

        else:
            args_proxies = tree_map(self.raw_node.args, tracer.proxy)
            kw_proxies = {
                k: tree_map(raw, tracer.proxy)
                for k, raw in self.raw_node.kwargs.items()
            }
            res_tangent = self.jvp(*primal_result_proxies, *args_proxies, **kw_proxies, **kw_tangents)
        
        assert get_node_meta(self.raw_node), \
            f"Rule {self} should set metadata on the raw Node"
        
        res_tangent: Union[EasierProxy, Sequence[Union[None, EasierProxy]]]
        return tree_map(res_tangent, lambda p: p.node)  # type: ignore


tangent_rule_registry: Dict[Callable, Type[DiffRuleBase]] = {}

class SetitemRule(DiffRuleBase):
    normalize_to_kwargs_only = False

    def output_meta(self, target, index, input):
        return get_node_meta(target)

    def jvp(self, target, index, input, target_t, input_t):
        set_node_meta(self.raw_node, get_node_meta(target))

        target_t[index] = input_t
        return target_t

tangent_rule_registry[operator.setitem] = SetitemRule


class EsrSumRule(DiffRuleBase):
    normalize_to_kwargs_only = False

    def output_meta(self, input):
        imeta: RuntimeTensorMeta = get_node_meta(input)  # type: ignore
        return RuntimeTensorMeta(
            Role.REPLICATED, (1,) + imeta.shape[1:], imeta.dtype
        )

    def jvp(
        self,
        input, input_t
    ):

        return esr.sum(input_t)

tangent_rule_registry[esr.sum] = EsrSumRule



differentiabilities: Dict[Callable, List[Differentiability]] = {}

#
# GENERATED by tools/autodiff/gen_autodiff.py
#

differentiabilities[torch.mul] = [
    Differentiability(
        ['input', 'other']
    )
]

differentiabilities[torch.add] = [
    Differentiability(
        ['input', 'other'],
        [('alpha', 1)]
    )
]

differentiabilities[torch.lt] = [
    Differentiability(
        [],
        [('input', required), ('other', required), ('alpha', 1)]
    )
]

def _normalize_einsum_kwargs(op, args, kwargs):
    def _match(equation, tensors):
        return { 'equation': equation, 'tensors': tensors }
    return _match(*args, **kwargs)

differentiabilities[torch.einsum] = [
    Differentiability(
        ['tensors'],
        [('equation', required)],
        kwargs_normalizer=_normalize_einsum_kwargs
    )
]