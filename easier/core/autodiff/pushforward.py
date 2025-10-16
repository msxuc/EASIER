# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import dataclasses
import functools
import operator
from typing import Callable, Dict, List, Literal, Optional, Sequence, TypeAlias, Union
import torch

from easier.core.passes.utils import EasierInterpreter

Scalar: TypeAlias = Union[int, float]
KEY__PUSHFORWARD_META = 'easier_autodiff_pushforwardMeta'


# Keys are torch operators. Many operators may share the same pushforward.
pushforward_registry: Dict[Callable, Callable] = {}


@dataclasses.dataclass
class PushforwardMeta:

    param_names: List[str]

    primal_result_param_pos: int = -1
    
    # result_tangent_param_pos: Optional[int] = None

    # 
    input_differentiablity: int = 1

    # - False: for single-result op, the result is not differentiable.
    # - Sequence[bool]: mark each result if differentiable or not.
    # - 'default': no matter single- or multi-result, all are differentiable.
    output_differentiability: Union[
        Literal['default', False], Sequence[bool]
    ] = 'default'


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

    Must be applied to pushforward first -- at the **bottom** of decorators.

    The scheme for pushforward function:
    -   All parameters for the primal operator must be included:
        -   Positional parameters must be in the same order;
        -   Keyword parameters can remain as keyword parameters too.

        -   All parameters must have the exactly same names as those of the
            PyTorch operator.
            With one extra enforcement: `self` must be renamed to `input`.

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
    def pf_decorator(pushforward_func: Callable):
        
        for f in primal_func:
            assert f not in pushforward_registry
            pushforward_registry[f] = pushforward_func

        meta = _parse(pushforward_func)
        pushforward_func.__dict__[KEY__PUSHFORWARD_META] = meta

        # the raw function of pushforward is not changed.
        return pushforward_func  
    return pf_decorator

def output_differentiability(
    # For a single `True`, no need to call this decorator.
    differentiability: Union[Literal[False], Sequence[bool]]
):
    """
    For example:
    ```
    @output_differentiability(False)
    @pushforward(torch.count_nonzero)
    def count_nonzero(input, dim):
        raise EasierJitException("not differentiable")
    
    @output_differentiability([True, False])  # returns (sorted, pos)
    @pushforward(torch.sort)
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

    The "default" case is, no matter it's multi-result or not (which can be
    told by the Node and subsequent getitem Nodes),
    the result or all result items are differentiable.
    We don't need to call with `True` or `[True, ...]` for the default case.
    """
    def pf_decorator(pushforward: Callable):

        meta: PushforwardMeta = pushforward.__dict__[KEY__PUSHFORWARD_META]
        meta = dataclasses.replace(
            meta, output_differentiability=output_differentiability
        )
        pushforward.__dict__[KEY__PUSHFORWARD_META] = meta

        # the raw function of pushforward is not changed.
        return pushforward  
    return pf_decorator


def _parse(pushforward: Callable) -> PushforwardMeta:
    """
    Trace merely the pushforward function, which results in special PLACEHOLDER
    Nodes for parameters.
    These Nodes are in the same order as the pushforward Python function and
    can tell the parameter names, but cannot tell they're keyword param or not.

    TODO primal op callsites may use positional-as-keyword, how to bind?
    """
    gm = torch.fx.symbolic_trace(pushforward)
    param_names: List[str] = []

    primal_param_names: List[str] = []

    class _ParamGetter(EasierInterpreter):
        def if_placeholder(self, param_name: str):
            param_names.append(param_name)
    # TODO EasierInterpreter requires (but not strictly) esr.Module, but
    # GraphModule is merely a torch.nn.Module. Can we relax that requirement?
    _ParamGetter([gm], [gm.graph]).run()  # type: ignore

    for param_name in param_names:
        if param_name.endswith('_t') and param_name != 'result_t':
            assert param_name[:-2] in param_names, \
                f'Bad parameter name {param_name} for tangent of primal' \
                f' parameter in pushfoward {pushforward.__name__}'
            # `result_t` can appear solely without `result`.
        
        if (not param_name.endswith('_t')) and param_name not in ['result', 'result_t']:
            primal_param_names.append(param_name)
        
    # TODO when we have hundreds of operators, this tracing-based subprocess
    # may be slow, but for real AD usage we may only need the param info
    # for the involved operators, no need to load all. Measure and make it lazy
    # if it has to be run during easier.autodiff module loading time.

    meta = PushforwardMeta(
        primal_param_names=primal_param_names
    )
    return meta

@aux
def maybe_multiply(t: torch.Tensor, s: Scalar):
    if s == 1:
        return t
    else:
        return t * s


@pushforward(operator.setitem)
def setitem(
    target: torch.Tensor, index, value: torch.Tensor,
    value_t: torch.Tensor
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