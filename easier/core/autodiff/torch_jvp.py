# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.




import dataclasses
import operator
from typing import Callable, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union

import torch
from torch.fx import Node

import easier as esr
from easier.core.autodiff.autodiff import FxConst
from easier.core.passes.utils import FX, fx_normalize_function_variant_into_kwargs

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
    #
    # Generally the param names are not ordered within this dataclass.
    other_params: List[Tuple[str, Union[int, float, str]]] = dataclasses.field(default_factory=list)

    # TODO certain ops like aten::_to_copy has this field a function rather
    # than a constant, e.g.
    # `output_differentiability: ["!dtype || isDifferentiableType(*dtype)"]`
    output: Union[Literal[True], List[bool]] = True

    def all_param_names(self) -> Set[str]:
        # For current handling of overloading resolution, a Set[str] suffices.
        params = set()
        params.update(self.diffable_params)
        params.update(k for k, v in self.other_params)
        return params


class DiffRuleBase:
    normalize_to_kwargs_only: bool = True

    diff_params: Optional[List[str]] = None

    def input_differentiability(self, *args, **kwargs) -> Dict[str, Union[FxConst, Node, Sequence[Node]]]:
        """
        Resolve inputs at the callsite, decide which inputs are differentiable.

        Unlike generating jvp sub-Graph using torch.func.jvp, EASIER DiffRule
        is purely symbolic, so derived Rule class does not need to convert
        FX scalars/constants to ()-shape tensor values.
        """
        # Default implementation:
        # - Resolve overloading via FX normalization;
        # - Decide input differentiability using simple param names list.
        assert self.diff_params is not None
        assert self.normalize_to_kwargs_only, \
            "Only FX-normalizable torch-Python/torch.ops.aten operators can" \
            " simply use `diff_params` field for differentiability of inputs"
        assert len(args) == 0  # effectively by normalize-only

        assert len(set(self.diff_params) - kwargs.keys()) == 0, \
            "All `diff_params` must be present in callsite arguments"

        diff_kwargs = kwargs.fromkeys(self.diff_params)
        return diff_kwargs  # type: ignore
            
    
    def output_differentiability(self)->Union[Literal[True], List[bool]]:
        # Default implmentation:
        # - Single output and differentiable
        
        # TODO cover torch.to/aten._to_copy whose output diff is dynamic
        return True
    
    def pushforward(self, *args, **kwargs):
        raise NotImplementedError("Derived class should implement this")
    
    def __init__(self, node: Node, op: Callable) -> None:
        self.node = node
        self.op = op
    
    def invoke(
        self,
        handle_inputs_prepare_tangents: Callable[
            [
                Dict[str, Union[FxConst, Node, Sequence[Node]]],
            ],
            Tuple[
                Dict[str, Union[FxConst, Node, Sequence[Node]]],
                Dict[str, Union[FxConst, Node, Sequence[Node]]]
            ]
        ]
    ):
        assert self.node.op == FX.CALL_FUNCTION

        if self.normalize_to_kwargs_only:
            kwargs = fx_normalize_function_variant_into_kwargs(self.op, self.node.args, self.node.kwargs)
            diff_inputs = self.input_differentiability(self, **kwargs)
        else:
            diff_inputs = self.input_differentiability(*self.node.args, **self.node.kwargs)


        diff_inputs, _ = handle_inputs_prepare_tangents(diff_inputs)
        tangents: Dict[str, Union[FxConst, Node, Sequence[Node]]] = {}
        for param_name, primal in diff_inputs.items():
            tangent_param = param_name + '_t'
            assert tangent_param not in diff_inputs
            assert tangent_param not in tangents


        
        if self.normalize_to_kwargs_only:
            pf_res = self.pushforward(**kwargs, **tangents)
        else:
            pf_res = self.pushforward(*self.node.args, **self.node.kwargs, **tangents)
        



#
# TODO before dispatch to Differentiability + torch.func.jvp,
# certain ops must be handled specifically and manually
#

tangent_rules: Dict[Callable, List[Tuple[Differentiability, Callable]]] = {}

# def tangent(
#     target_op: Callable,
#     diff_params: List[str], other_params: List[Tuple[str, Union[int, float, str]]] = [],
#     output_differentiability: Union[Literal[True], List[bool]] = True
# ):
#     def wrapper(rule_func):
#         tangent_rules.setdefault(target_op, []).append((
#             Differentiability(
#                 diff_params, other_params, output_differentiability
#             ), rule_func
#         ))
#     return wrapper

class SetitemRule(DiffRuleBase):
    normalize_to_kwargs_only = False

    def pushforward(self, input, index, value):

# @tangent(operator.setitem)
def setitem(
    self_p, index, value_p,
    value_t
):
    return value_t


def esr_sum(
    self_p, self_t
):
    return esr.sum(self_t)

# TODO add decorator on functions for these cases
# tangent_rules[torch.div] = []

def div(
    result,  # TODO needs result
    self_p, other_p,
    self_t, other_t
):
    return (self_t - other_t * result) / other_p




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