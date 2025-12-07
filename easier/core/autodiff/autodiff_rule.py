# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import dataclasses
import operator
from typing import Callable, Dict, List, Literal, Optional, Sequence, \
    Set, Tuple, Type, Union, TYPE_CHECKING

import torch
from torch.fx import Node, Graph
from torch.fx.node import Argument as FxArg

import easier as esr
from easier.core.passes.utils import \
    FX, fx_normalize_function_variant_into_kwargs, tree_map
from easier.core.runtime.metadata import \
    Role, RuntimeTensorMeta, collect_meta, get_node_meta
from easier.core.autodiff.utils import FxConst
from easier.core.utils import EasierJitException


if TYPE_CHECKING:
    from easier.core.jit import EasierProxy


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
    other_params: List[Tuple[str, Union[RequiredParam, FxConst, None]]] = \
        dataclasses.field(default_factory=list)

    # TODO certain ops like aten::_to_copy has this field a function rather
    # than a constant, e.g.
    # `output_differentiability: ["!dtype || isDifferentiableType(*dtype)"]`
    # and `torch.split` or `torch.ops.aten.split_with_sizes` return as many
    # items as input items.
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


    def input_differentiability(
        self, *args, **kwargs
    ) -> Dict[str, Union[FxConst, Node, Sequence[Node]]]:
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

        raw_diff_kwargs = { p: raw_kwargs[p] for p in self.diffable_params }
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
        roles = []
        def _make(x):
            if isinstance(x, Node):
                meta = get_node_meta(x)
                assert isinstance(meta, RuntimeTensorMeta), \
                    "Value of arg Node cannot be nested structure"

                roles.append(meta.role)

                return torch.zeros(meta.shape, dtype=meta.dtype)
            else:
                return x

        vals = tree_map(args, _make)
        kwvals = { k: tree_map(v,  _make) for k, v in kwargs.items() }
        res = self.op(*vals, **kwvals)
        assert isinstance(res, torch.Tensor), \
            'Default impl supports single-res op only'

        role = \
            Role.DISTRIBUTED if Role.DISTRIBUTED in roles else Role.REPLICATED
        return RuntimeTensorMeta(role, tuple(res.shape), res.dtype)

    
    def jvp(self, *args, **kwargs):
        """
        Inputs:
        -   Optional Union[EasierProxy, Tuple[EasierProxy]] for primal result
            if needed.

            NOTE: for multi-res op, this param for result is an explicit tuple,
            not a single Node such that indexing it will result in extra
            getitem Nodes.

        -   EasierProxy for primal input Nodes
        -   Constants remain constants

        Returns:
        -   Tensor/Proxy: A single result item for tangent

        -   List[None | Tensor/Proxy]:
            Resultant tangent items for a multiple-result operator.
            For primal item that doesn't have a tangent, the result item should
            be None.
        """
        raise NotImplementedError("Derived class should implement this")
    
    def __init__(self, node: Node, op: Callable, raw_meta_ctor: Callable):
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
            ometa = self.output_meta(
                **self.raw_normalized_kwargs
            )

        else:
            ometa = self.output_meta(
                *self.raw_node.args, **self.raw_node.kwargs
            )
        
        return ometa
        
    
    def inject_jvp_subgraph(
        self,
        # If multi-res op, this param including non-diffable result item.
        primal_result: Union[Node, Sequence[Node]],
        diff_input_names: List[str],
        raw2primal: Dict[Node, Node],
        # tangents are currently Tensors/Nodes, and might be ()-shape.
        tangents: List[Union[Node, Sequence[Node]]]
    ) -> Union[Node, Sequence[Node]]:
        assert self.raw_node.op == FX.CALL_FUNCTION

        from easier.core.jit import EasierProxy, EasierTracer
        from easier.core.autodiff.autodiff import FxConst

        jvp_graphs: List[Graph] = collect_meta(
            primal_result, lambda n: n.graph, leaf_type=Node
        )
        assert len(set(jvp_graphs)) == 1
        jvp_graph = jvp_graphs[0]

        tracer = EasierTracer()
        tracer.graph = jvp_graph

        primal_result_proxies = []
        if self.needs_result:
            primal_proxy = tree_map(primal_result, lambda n: tracer.proxy(n))
            primal_result_proxies = [primal_proxy]
        

        def _raw_arg_proxy(raw_arg):
            if isinstance(raw_arg, FxConst.__args__):
                return raw_arg
            else:
                assert raw_arg.graph is not jvp_graph

                primal_arg = raw2primal[raw_arg]
                # Including Node and nested structure -- will result in
                # explicit getitem Nodes
                return tracer.proxy(primal_arg)


        # Tangent parameters are suffixed by _t .e.g input_t, other_t
        kw_tangents_proxies = dict(zip(
            (n + '_t' for n in diff_input_names),
            tree_map(tangents, tracer.proxy)
        ))

        if self.fx_normalize_to_kwargs_only:
            norm_kw_proxies = {
                k: tree_map(raw, _raw_arg_proxy)
                for k, raw in self.raw_normalized_kwargs.items()
            }
            res_tangent = self.jvp(
                *primal_result_proxies,
                **norm_kw_proxies, **kw_tangents_proxies
            )

        else:
            args_proxies = tree_map(self.raw_node.args, _raw_arg_proxy)
            kwargs_proxies = {
                k: tree_map(raw, _raw_arg_proxy)
                for k, raw in self.raw_node.kwargs.items()
            }
            res_tangent = self.jvp(
                *primal_result_proxies, *args_proxies,
                **kwargs_proxies, **kw_tangents_proxies
            )
        
        assert get_node_meta(self.raw_node), \
            f"Rule {self} should set metadata on the raw Node"
        
        res_tangent: Union[EasierProxy, Sequence[Union[None, EasierProxy]]]
        return tree_map(
            res_tangent,
            lambda p: None if p is None else p.node
        )  # type: ignore


tangent_rule_registry: Dict[Callable, Type[DiffRuleBase]] = {}
differentiabilities: Dict[Callable, List[Differentiability]] = {}

#
# Custom DiffRule
#

class SetitemRule(DiffRuleBase):
    fx_normalize_to_kwargs_only = False

    def input_differentiability(
        self, target, index, input
    ) -> Dict[str, Union[FxConst, Node, Sequence[Node]]]:
        return {'target': target, 'input': input}

    def output_meta(self, target, index, input):
        return get_node_meta(target)

    def jvp(
        self,
        target, index, input,
        target_t: 'EasierProxy', input_t: 'EasierProxy'
    ):
        target_t.node.graph.call_function(
            operator.setitem, (target_t.node, index, input_t.node)
        )
        return target_t

tangent_rule_registry[operator.setitem] = SetitemRule


class GetitemRule(DiffRuleBase):
    fx_normalize_to_kwargs_only = False

    def input_differentiability(
        self, input, index
    ) -> Dict[str, Union[FxConst, Node, Sequence[Node]]]:
        return {'input': input}

    def output_meta(self, input, index):
        imeta = get_node_meta(input)
        assert isinstance(imeta, RuntimeTensorMeta), \
            "Tuple unpacking is handled elsewhere"
        out_shp = tuple(torch.zeros(imeta.shape)[*index].shape)
        return RuntimeTensorMeta(imeta.role, out_shp, imeta.dtype)

    def jvp(self, input, index, input_t):
        return input_t[index]

tangent_rule_registry[operator.getitem] = GetitemRule


class EsrSumRule(DiffRuleBase):
    fx_normalize_to_kwargs_only = False

    def input_differentiability(
        self, input
    ) -> Dict[str, Union[FxConst, Node, Sequence[Node]]]:
        return {'input': input}

    def output_meta(self, input):
        imeta: RuntimeTensorMeta = get_node_meta(input)  # type: ignore
        return RuntimeTensorMeta(
            Role.REPLICATED, (1,) + imeta.shape[1:], imeta.dtype
        )

    def jvp(self, input, input_t):
        return esr.sum(input_t)

tangent_rule_registry[esr.sum] = EsrSumRule


class PowRule(DiffRuleBase):
    fx_normalize_to_kwargs_only = True
    
    def __init__(self, node: Node, op: Callable, raw_meta_ctor: Callable):
        super().__init__(
            node,
            torch.pow,  # unify operator.pow
            raw_meta_ctor
        )

    def input_differentiability(
        self, input, exponent
    ) -> Dict[str, Union[FxConst, Node, Sequence[Node]]]:
        d = {}
        if isinstance(input, Node):
            d['input'] = input
        if isinstance(exponent, Node):
            d['exponent'] = exponent

        self.diffable_choice = d
        if list(d.keys()) != ['input']:
            raise NotImplementedError(
                "torch.pow with two Tensors has special semantics"
            )

        return d

    def jvp(self, input, exponent, input_t):
        # TODO Current exponent is constant Scalar
        # TODO For optional tangents, take kwargs like **tangents and check
        # 'input_t, expoenent_t' and their primal types.
        if exponent == 0:
            return torch.zeros_like(input)
        else:
            return input_t * (exponent * torch.pow(input, exponent - 1))


tangent_rule_registry[operator.pow] \
    = tangent_rule_registry[torch.pow] \
    = PowRule


class _NonDiffable(DiffRuleBase):
    # no matter normalizable or not, we don't need to do normalization
    fx_normalize_to_kwargs_only = False  

    def input_differentiability(self, *args, **kwargs):
        return {}

    def jvp(self, *args, **kwargs):
        raise EasierJitException("unreachable")

tangent_rule_registry[operator.lt] = _NonDiffable

#
# Custom torch op Differentiability for torch.func.jvp()
#
# Mainly for:
# - Composed torch.aten ops not in derivatives.yaml
#   (i.e. in native_functions.yaml they don't have 'core' tag)
#
#   But if they are accepted by torch.jvp(), we can simply add
#   Differentiability entries for them.
#
# - torch.aten ops accepted by torch.jvp(), but minor preprocess is needed,
#   e.g. torch.einsum.
#
# - torch.aten ops in derivatives.yaml but requiring special parsing.
#   It's easier to manual define Differentiability for them.
#

differentiabilities[torch.ops.aten.add_] = [
    Differentiability(
        ['input', 'other'],
        [('alpha', 1)]
    )
]

differentiabilities[torch.concat] = [
    Differentiability(
        ['tensors'],
        [('dim', 0)]
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

differentiabilities[torch.lt] = [
    Differentiability(
        [],
        [('input', required), ('other', required)]
    )
]

differentiabilities[torch.ops.aten.sub_] = [
    Differentiability(
        ['input', 'other'],
        [('alpha', 1)]
    )
]

differentiabilities[torch.sort] = [
    Differentiability(
        ['input'],
        [('dim', -1), ('descending', False)],
        output_differentiability=[True, False]
    ),
    # Differentiability(
    #     ['input'],
    #     [('stable', None), ('dim', -1), ('descending', False)]
    #     output_differentiability=[True, False]
    # )
]
