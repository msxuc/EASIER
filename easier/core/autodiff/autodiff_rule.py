# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import dataclasses
import operator
from typing import Callable, Dict, List, Literal, Optional, Sequence, \
    Set, Tuple, Type, Union, TYPE_CHECKING
import typing

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

    # TODO this is actually shared by all Differentiability overloadings.
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

    output_differentiability: Union[Literal[True], List[bool]] = True

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

        is_aten_api = 'aten' in self.op.__module__
        if is_aten_api:
            if 'input' in kwvals:
                kwval_input = kwvals.pop('input')
                kwvals['self'] = kwval_input

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

        -   EasierProxies for primal input Nodes
        -   EasierProxies for tangent input Nodes

            A primal or tangent input Node may have structure e.g. tuple
            if the raw arg is a nested structure.

        -   Constants remain constants

        Returns:
        -   Tensor/Proxy: A single result item for tangent

        -   List[None | Tensor/Proxy]:
            Resultant tangent items for a multiple-result operator.
            For primal item that doesn't have a tangent, the result item should
            be None.
        """
        raise NotImplementedError("Derived class should implement this")
    
    def vjp(self, *args, **kwargs):
        """
        Inputs:
        -   Optional Union[EasierProxy, Tuple[EasierProxy]] for primal result
            if needed.

            NOTE: for multi-res op, this param for result is an explicit tuple,
            not a single Node such that indexing it will result in extra
            getitem Nodes.

        -   EasierProxies for primal input Nodes
        -   One EasierProxy for cotangent input Node

            A primal or the cotangent input Node may have structure e.g. tuple
            if the raw arg/result is a nested structure.

        -   Constants remain constants

        Returns:
        -   List[Tensor/Proxy | Tuple[Tensor/Proxy]]:
            Resultant cotangent items for each differentiable input.
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

        from easier.core.jit import EasierProxy, EasierTracer
        from easier.core.autodiff.jvp import FxConst

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
            res_tangent_proxy = self.jvp(
                *primal_result_proxies,
                **norm_kw_proxies, **kw_tangents_proxies
            )

        else:
            args_proxies = tree_map(self.raw_node.args, _raw_arg_proxy)
            kwargs_proxies = {
                k: tree_map(raw, _raw_arg_proxy)
                for k, raw in self.raw_node.kwargs.items()
            }
            res_tangent_proxy = self.jvp(
                *primal_result_proxies, *args_proxies,
                **kwargs_proxies, **kw_tangents_proxies
            )
        
        assert get_node_meta(self.raw_node), \
            f"Rule {self} should set metadata on the raw Node"
        
        res_tangent_proxy: Union[EasierProxy, Sequence[Union[None, EasierProxy]]]
        return tree_map(
            res_tangent_proxy,
            lambda p: None if p is None else p.node
        )  # type: ignore

    def inject_vjp_subgraph(
        self,
        primal_result: Union[Node, Sequence[Node]],
        diff_input_names: List[str],
        raw2primal: Dict[Node, Node],
        cotangent: Union[Node, Sequence[Node]]
    ) -> List[Union[Node, Sequence[Node]]]:
        from easier.core.jit import EasierProxy, EasierTracer
        from easier.core.autodiff.jvp import FxConst

        vjp_graphs: List[Graph] = collect_meta(
            primal_result, lambda n: n.graph, leaf_type=Node
        )
        assert len(set(vjp_graphs)) == 1
        vjp_graph = vjp_graphs[0]

        tracer = EasierTracer()
        tracer.graph = vjp_graph

        primal_result_proxies = []
        if self.needs_result:
            primal_proxy = tree_map(primal_result, lambda n: tracer.proxy(n))
            primal_result_proxies = [primal_proxy]
        

        def _raw_arg_proxy(raw_arg):
            if isinstance(raw_arg, FxConst.__args__):
                return raw_arg
            else:
                assert raw_arg.graph is not vjp_graph

                primal_arg = raw2primal[raw_arg]
                # Including Node and nested structure -- will result in
                # explicit getitem Nodes
                return tracer.proxy(primal_arg)


        cotangent_proxy = tree_map(cotangent, tracer.proxy)

        if self.fx_normalize_to_kwargs_only:
            norm_kw_proxies = {
                k: tree_map(raw, _raw_arg_proxy)
                for k, raw in self.raw_normalized_kwargs.items()
            }
            res_cotangent_proxy = self.vjp(
                *primal_result_proxies,
                cotangent=cotangent_proxy,
                **norm_kw_proxies
            )

        else:
            args_proxies = tree_map(self.raw_node.args, _raw_arg_proxy)
            kwargs_proxies = {
                k: tree_map(raw, _raw_arg_proxy)
                for k, raw in self.raw_node.kwargs.items()
            }
            res_cotangent_proxy = self.vjp(
                *primal_result_proxies, *args_proxies,
                cotangent=cotangent_proxy,
                **kwargs_proxies
            )
        
        assert get_node_meta(self.raw_node), \
            f"Rule {self} should set metadata on the raw Node"
        
        res_cotangent_proxy: List[Union[
            EasierProxy, Sequence[EasierProxy]
        ]]
        return tree_map(
            res_cotangent_proxy,
            lambda p: None if p is None else p.node
        )  # type: ignore



diff_rule_registry: Dict[Callable, Type[DiffRuleBase]] = {}
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
        # By simply invoking a setitem operation on tangent Tensors/Proxies
        # we can inject Nodes to jvp Graph, all nested proxies will be
        # converted to Nodes, e.g. in `X[:, self.i:self.j]`
        target_t[index] = input_t

        return target_t
    
    def vjp(self, *args, **kwargs):
        raise EasierJitException("Setitem does not support VJP")

diff_rule_registry[operator.setitem] = SetitemRule


class EsrSumRule(DiffRuleBase):
    fx_normalize_to_kwargs_only = False

    def input_differentiability(
        self, input
    ) -> Dict[str, Union[FxConst, Node, Sequence[Node]]]:
        return {'input': input}

    def jvp(self, input, input_t: 'EasierProxy'):
        # NOTE without fully fleged fx.symbolic_trace(), custom wrapped
        # functions like esr.aggregators are not handled but inlined to the
        # underlying torch.sum, which will lead to error when validating
        # Roles.
        esr_sum = input_t.node.graph.call_function(esr.sum, (input_t.node,))
        return input_t.tracer.proxy(esr_sum)
    
    def vjp(self, input, cotangent: 'EasierProxy'):
        return [cotangent.expand_as(input)]

diff_rule_registry[esr.sum] = EsrSumRule


class EsrNormRule(DiffRuleBase):
    fx_normalize_to_kwargs_only = False
    needs_result = True

    def input_differentiability(
        self, input, p=2
    ) -> Dict[str, Union[FxConst, Node, Sequence[Node]]]:
        return {'input': input}

    # NOTE because all arguments are actually passed in as keywords, so we
    # can put primal parameter p=2 to the last if it has default value.
    def jvp(self, norm_result, input, input_t, p=2):
        if p != 2:
            raise NotImplementedError(f"esr.norm p = {p} and != 2")
        
        mul = input_t * input

        # NOTE without fully fleged fx.symbolic_trace(), custom wrapped
        # functions like esr.aggregators are not handled but inlined to the
        # underlying torch.sum, which will lead to error when validating
        # Roles.
        mul: 'EasierProxy'
        esr_sum = mul.node.graph.call_function(esr.sum, (mul.node,))
        sum = mul.tracer.proxy(esr_sum)

        d = sum / norm_result
        return torch.where(norm_result == 0, 0, d)

    def vjp(self, norm_result, input, cotangent, p=2):
        if p != 2:
            raise NotImplementedError(f"esr.norm p = {p} and != 2")
        
        d = cotangent * (input / norm_result)
        return [torch.where(norm_result == 0, 0, d)]

diff_rule_registry[esr.norm] = EsrNormRule


class TorchNormRule(DiffRuleBase):
    """
    PyTorch derivatives.yaml and FunctionsManuals.cpp has bugs that overloading
    of torch.norm without dim/keepdim parameters are dispatched to the
    fully fledged version with keepdim=True, causing wrong shapes and this
    would make torch.jvp reject calls like `x.norm()`.
    """
    fx_normalize_to_kwargs_only = True
    diffable_params = ['input']

    needs_result = True

    def jvp(self, norm_result, input, input_t, p=2):
        if p != 2:
            raise NotImplementedError(f"torch.norm p = {p} and != 2")
        
        d = torch.sum(input_t * input) / norm_result
        return torch.where(norm_result == 0, 0, d)

    def vjp(self, norm_result, input, cotangent, p=2):
        if p != 2:
            raise NotImplementedError(f"esr.norm p = {p} and != 2")
        
        d = cotangent * (input / norm_result)
        return [torch.where(norm_result == 0, 0, d)]

diff_rule_registry[torch.norm] = \
diff_rule_registry[torch.ops.aten.norm] = \
    TorchNormRule


class ClampRule(DiffRuleBase):
    fx_normalize_to_kwargs_only = True

    def input_differentiability(
        self, input, min=None, max=None
    ) -> Dict[str, Union[FxConst, Node, Sequence[Node]]]:
        d = {'input': input}

        # but may be Scalar, will result in ()-shape zero tangent tensor
        if min is not None:  
            d['min'] = min
        if max is not None:
            d['max'] = max
        
        self.input_diff = d

        return d

    @typing.no_type_check
    def jvp(
        self,
        input,
        min: Union['EasierProxy', None]=None, max=None,
        input_t='MUST_EXIST_MAYBE_ZEROS',
        min_t: Union['EasierProxy', None]=None, max_t=None
    ):
        if min is not None and max is not None:
            return torch.where(
                min > max,
                max_t,
                torch.where(
                    input < min,
                    min_t,
                    torch.where(
                        input > max,
                        max_t,
                        input_t
                    )
                )
            )
        elif min is not None:
            return torch.where(input > min, input_t, min_t)
        elif max is not None:
            return torch.where(input < max, input_t, max_t)
        else:
            return input_t
    
diff_rule_registry[torch.clamp] = \
diff_rule_registry[torch.ops.aten.clamp] = \
    ClampRule


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
    

    def vjp(self, input, exponent, cotangent):
        if exponent == 0:
            return [torch.zeros_like(input)]
        else:
            d = cotangent * (exponent * torch.pow(input, exponent - 1))
            return [d]


diff_rule_registry[operator.pow] = \
diff_rule_registry[torch.pow] = \
    PowRule


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


#
# Python-syntax getitem, depending on index being scalars, slices, tensors,
# will lead to a lot of backprop aten::op calls for each kind of indexing.
#
# We add a aux, kwargs-style method to go into the torch.vjp pipeline.
#
def getitem_aux_kw(input, index):
    return input[index]
def _normalize_operator_getitem(op, args, kwargs):
    input, index = args
    return { 'input': input, 'index': index }
differentiabilities[operator.getitem] = \
differentiabilities[getitem_aux_kw] = [
    Differentiability(
        ['input'],
        [('index', required)],
        kwargs_normalizer=_normalize_operator_getitem
    )
]


#
# Ops not in derivatives.yaml
#

differentiabilities[torch.ops.aten.add_] = [
    Differentiability(
        ['input', 'other'],
        [('alpha', 1)]
    )
]

differentiabilities[torch.aminmax] = [
    Differentiability(
        ['input'],
        [('dim', None), ('keepdim', None)],

        # TODO VJP not supported by torch yet
        output_differentiability=[True, True]
    )
]

differentiabilities[torch.concat] = [
    Differentiability(
        ['tensors'],
        [('dim', 0)]
    )
]

differentiabilities[torch.frexp] = [
    Differentiability(
        ['input'],
        output_differentiability=[True, True]
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

differentiabilities[torch.linalg.svd] = [
    Differentiability(
        ['A',],
        [('full_matrices', True), ('driver', None)],
        output_differentiability=[True, True, True]
    )
]

differentiabilities[torch.ops.aten.sub_] = [
    Differentiability(
        ['input', 'other'],
        [('alpha', 1)]
    )
]

#
# Ops to manually handle, but via Differentiability mechanism
#

differentiabilities[torch.matmul] = [
    Differentiability(
        ['input', 'other'],
    )
]

# TODO let gen_autodiff handles out_differentiability
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

differentiabilities[torch.squeeze] = \
differentiabilities[torch.ops.aten.squeeze] = [
    Differentiability(
        ['input'],
    ),
    Differentiability(
        ['input'],
        [('dim', required)]  # int or int[]
    )
]

differentiabilities[torch.lt] = [
    Differentiability(
        [],
        [('input', required), ('other', required)]
    )
]
