# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import functools
import operator
from typing import Callable, List, Literal, Sequence, TypeAlias, Union
import torch

Scalar: TypeAlias = Union[int, float]

# TODO codegen: use fixed version of derviatives.yaml, may be not suitable for different versions of pytorch.

# TODO make these lists only accessible by codegen for pushforwards.
aux_funcs: List[Callable] = []

def aux(func):
    """
    Some definitions of pushforward in PyTorch `derivatives.yaml` config file
    use torch.autograd-internal auxiliary functions (not even torch operators!)
    defined in `torch/csrc/autograd/FunctionsManual.cpp`.

    To ease the code-generation from `derivatives.yaml`, we can define such
    auxiliary functions with `@aux`, then the code-generator will gently handle
    those pushfoward definitions.
    """
    aux_funcs.append(func)
    return func

def pushforward(*primal_func):
    """
    Define a pushforward function for a primal operator.

    The scheme for pushforward function:
    -   All parameters for the primal operator must be included:
        -   Positional parameters must be in the same order;
        -   Keyword parameters can remain as keyword parameters too.

    -   An optional parameter named 'result' can be included as a positional
        parameter, in whatever position (but recommended to be the first).
        It represents the primal result of the primal operator.
    
    -   If a primal parameter is involved in the pushforward of the tangent
        value, a parameter for its tangent should be added:
        -   For a positional primal parameter, the tangent parameter should
            be a positional parameter too, and recommended to be after primal
            parameters;
        -   For a keyword primal parameter, the tangent parameter should
            be a keyword parameter too.
        -   The name of the tangent parameter should be named with a suffix
            '_t', for example, the tangent parameter for the primal parameter
            'mat' should be named 'mat_t'.

    -   If a tangent parameter is declared, it must be used.
        Don't declare tangent parameters for primal parameters that are not
        involved in the pushforward!

    For example:
    ```
    @pushforward(torch.addmv)
    def addmv(
        # result,               # this pushforward does not use primal result
        input, mat, vec,        # primal params in the same order
        input_t, mat_t, vec_t   # tangent params, positional
        *,
        beta: Scalar, alpha: Scalar  # primal keywords, none has tangent
    ):
        ...
    ```
    """
    def pf_decorator(pf_def):
        # the raw function of pushforward is not changed.
        return pf_def  
    return pf_decorator

def output_differentiability(differentiability: Union[Literal[False], Sequence[bool]]):
    """
    For example:
    ```
    @pushforward(torch.count_nonzero)
    @output_differentiability(False)
    def count_nonzero(input, dim):
        raise EasierJitException("not differentiable")
    
    @pushforward(torch.sort)
    @output_differentiability([True, False])  # returns (sorted, pos)
    def sort(
        result,                     # primal result: (sorted, pos)
        input, dim, descending,
        input_t                     # only one param can have tangent
    ):
        (_sorted, pos) = result
        res_t = input_t[pos]

        # return only 1 item -- len(filter(is_True, differentiability))
        return [res_t]  
    ```
    """
    def pf_decorator(pf_def):
        # the raw function of pushforward is not changed.
        return pf_def  
    return pf_decorator

@aux
def maybe_multiply(t: torch.Tensor, s: Scalar):
    if s == 1:
        return t
    else:
        return t * s


@pushforward(operator.setitem)
def setitem(
    target, index, value,
    value_t
):
    """
    setitem is a typical example of inplace operator
    (and not defined in PyTorch derivatives.yaml,
    however, torch.fill_/copy_ are defined there).

    The data container `target`, an esr.Tensor instance or an immediate Tensor,
    may start to carry tangent only after this write

    P.S. `target` may also be shadowed into a non-carrier if `value` is not a
    tangent carrier, but for such cases this pushforward function won't be
    activated or called.  TODO will it? How is add_ handled then?
    """
    # TODO how is target_t storage ever involved? especially, broadcasting may be needed
    return value_t

@pushforward(torch.Tensor.add_)
def add_(target, value, target_t, value_t):
    """
    TODO
    1.  `add_` may not need a rule at all, as it's derived from non-inplace version of `add`
    2.  there are 2*2=4 combinations of target/value carries tangent or not.
        the case e.g. value does not carry tangent equals value_t==0,
        but can AD framework decide value_t==0 equal no need to call pushforward? Can we decide this out of linearity from chain rule?
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