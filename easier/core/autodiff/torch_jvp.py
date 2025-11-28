# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.




import dataclasses
import operator
from typing import Callable, Dict, List, Literal, Optional, Sequence, Set, Tuple, Type, Union
from typing_extensions import OrderedDict

import torch
from torch.fx import Node

import easier as esr
from easier.core.autodiff.autodiff import FxConst, FxArg
from easier.core.passes.utils import FX, fx_normalize_function_variant_into_kwargs
from easier.core.runtime.metadata import Role, RuntimeTensorMeta, get_node_meta, set_node_meta

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
    other_params: List[Tuple[str, FxConst]] = dataclasses.field(default_factory=list)

    # TODO certain ops like aten::_to_copy has this field a function rather
    # than a constant, e.g.
    # `output_differentiability: ["!dtype || isDifferentiableType(*dtype)"]`
    output: Union[Literal[True], List[bool]] = True

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
            
    
    def output_differentiability(self)->Union[Literal[True], List[bool]]:
        # Default implmentation:
        # - Single output and differentiable
        
        # TODO cover torch.to/aten._to_copy whose output diff is dynamic
        return True
    
    def pushforward(self, *args, **kwargs):
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
    
    def invoke(
        self,
        # If multi-res op, this param including non-diffable result item.
        primal_result: Union[Node, Sequence[Node]],
        diff_input_names: List[str],
        tangents: List[Union[FxConst, Node, Sequence[Node]]]
    ):
        assert self.raw_node.op == FX.CALL_FUNCTION

        args = []
        if self.needs_result:
            args = [primal_result]

        kw_tangents = dict(zip(diff_input_names, tangents))

        if self.fx_normalize_to_kwargs_only:
            pf_res = self.pushforward(*args, **self.raw_normalized_kwargs, **kw_tangents)

        else:
            pf_res = self.pushforward(*args, *self.raw_node.args, **self.raw_node.kwargs, **kw_tangents)
        


tangent_rule_registry: Dict[Callable, Type[DiffRuleBase]] = {}

class SetitemRule(DiffRuleBase):
    normalize_to_kwargs_only = False

    def pushforward(self, target, index, input, target_t, input_t):
        set_node_meta(self.raw_node, get_node_meta(target))

        target_t[index] = input_t
        return target_t

tangent_rule_registry[operator.setitem] = SetitemRule


class EsrSumRule(DiffRuleBase):
    normalize_to_kwargs_only = False

    def pushforward(
        self,
        input, input_t
    ):
        imeta: RuntimeTensorMeta = get_node_meta(input)  # type: ignore
        set_node_meta(self.raw_node, RuntimeTensorMeta(
            Role.REPLICATED, (1,) + imeta.shape[1:], imeta.dtype
        ))

        return esr.sum(input_t)

tangent_rule_registry[esr.sum] = EsrSumRule

class DivRule(DiffRuleBase):
    needs_primal_result = True

    def pushforward(
        self,
        result,  # TODO needs result
        input, other,
        input_t, other_t
    ):
        return (input_t - other_t * result) / other

tangent_rule_registry[torch.div] = DivRule
# tangent_rule_registry[torch.Tensor.div_] = DivRule




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


def _normalize_einsum_kwargs(op, args, kwargs):
    def _match(equation, tensors):
        return { 'equation': equation, 'tensors': tensors }
    return _match(*args, **kwargs)

differentiabilities[torch.einsum] = [
    Differentiability(
        ['tensors'],
        [('equation', None)],
        kwargs_normalizer=_normalize_einsum_kwargs
    )
]