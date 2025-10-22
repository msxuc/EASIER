# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Manually registered derivatives rules for some special PyTorch operators.

Currently only about the forward propagation / tangent / pushforward rules.

The sources of the all rules are:
- github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml
- github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/FunctionsManual.cpp


This manual registry includes:

-   Forbidded operators in EASIER programming model
    e.g. torch.unique

    TODO forbidden as distributed op, but ok as replicated op
    (need tensor_grouping AOT pass to tell)
    (however even as replicated op, stableness and consistency are required)
    TODO maybe we need a universal pass to check this like syntax_check pass

-   Rules declared in derivatives.yaml but hard to parse
    e.g. TODO ANY?

-   Operators whose derivatives are specially handled by PyTorch
    e.g. torch.matmul

-   Auxiliary functions in FunctionsManual.cpp that are used by auto-transpiled
    derivative rules from derivatives.yaml
    e.g. maybe_multiply


Both manually registered and auto-generated derivative rules have these
properties and requirements:

-   Generally only the non-inplace operators are included.

    Other derived operators like the inplace versions
    e.g. `add_` or `add(out=...)`
    are generally handled by EASIER autodiff framework and do not have rules.

-   PyTorch operator overloadings must be merged.

    E.g. using `passes.utils.fx_normalize_function_variant_into_kwargs`
    (certain operators in e.g. `torch.functional` are excluded)

    Merging overloadings increase the number of tangent component terms in
    $\\sum_i \\frac{\\partial{f}}{\\partial{X_i}}(P_i) * T_i$
    for multi-arg operators, and lead to unnecessary zero tangent computations,
    therefore rule definition need to do extra work to separate the tangent
    component terms for arguments.

-   The rule implementations must be value-independent, i.e. not depending on
    shapes or element values.
    TODO certain torch rules do use shapes, may be we need to leave a special
    conditioning mark for NodeEvaluator to mimic control flow
    (Graph-level), or encapsulate the conditioning into an EASIER-provided op.


"""

import operator
from typing import TypeAlias, Union
import torch

from easier.core.autodiff.pushforward import aux, pushforward

Scalar: TypeAlias = Union[int, float]

@aux
def maybe_multiply(t: torch.Tensor, s: Scalar):
    if s == 1:
        return t
    else:
        return t * s


# @pushforward(operator.setitem)
# def setitem(
#     target: torch.Tensor, index, value: torch.Tensor,
#     value_t: torch.Tensor
# ):
#     """
#     Although being inplace, setitem is a typical example of inplace operator
#     (and not defined in PyTorch derivatives.yaml,
#     however, torch.fill_/copy_ are defined there).

#     The data container `target`, an esr.Tensor instance or an immediate Tensor,
#     may start to carry tangent only after this write

#     P.S. `target` may also be shadowed into a non-carrier if `value` is not a
#     tangent carrier, but for such cases this pushforward function won't be
#     activated or called.  TODO will it? How is add_ handled then?
#     """
#     # TODO how is target_t storage ever involved? especially, broadcasting may be needed
#     return value_t

@pushforward(torch.add, torch.Tensor.add)
def add(
    input: torch.Tensor, other: torch.Tensor,
    input_t: torch.Tensor, other_t: torch.Tensor,
    *,
    alpha: Scalar
):
    # torch has an overloading with `other` being Scalar and not involved in pushforward,
    # for such cases we can escalate Scalars to Tensors, and go with other_t==0
    # -- it's AD framework to detect Scalar (always non-diff-able) and allocate replicated zero tangent.
    # TODO can we? store a constant (0,)-shape zero replica in JVP esr.Module?
    return input_t + maybe_multiply(other_t, alpha)

@pushforward(torch.Tensor.add_)
def add_(target, value, target_t, value_t):
    """
    TODO
    1.  `add_` may not need a rule at all, as it's derived from non-inplace version of `add`
        NOTE only the non-inplace version has rule defined (manually or auto-gen-ed)
            can be derived to its inplace version.

    2.  there are 2*2=4 combinations of target/value carries tangent or not.
        the case e.g. value does not carry tangent equals value_t==0,
        but can AD framework decide value_t==0 equal no need to call pushforward?
        Can we decide this out of linearity from chain rule?

    3.  similar to setitem, how target_t/result_t can be involved, with Tensor broadcasting behavior for free?
    """
    return target_t + value_t

# TODO temporarily for demo, can be auto-gen-ed
@pushforward(torch.addmv)
def addmv(
    input: torch.Tensor, mat: torch.Tensor, vec: torch.Tensor,
    input_t: torch.Tensor, mat_t: torch.Tensor, vec_t: torch.Tensor,
    *,
    beta: Scalar, alpha: Scalar
):
    return maybe_multiply(input_t, beta) + maybe_multiply(mat_t.mv(vec), alpha) + maybe_multiply(mat.mv(vec_t), alpha)