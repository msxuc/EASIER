# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import itertools
from typing import \
    Any, List, Optional, Tuple, Type, TypeVar, Union, cast, overload, \
    Sequence

import torch
import torch.overrides
from torch.fx.node import Node

from .metadata import \
    EasierTensorMeta, Role, StructuredTensorMeta, \
    get_meta_from_ir_literal, promote_scalar_types, ScalarType
from easier.core.utils import EasierJitException


# May throw if the corresponding method variant does not exist.
def get_method_variant(function_variant):
    return getattr(torch.Tensor, function_variant.__name__)


# May throw if the corresponding function variant does not exist.
def get_function_variant(method_variant):
    return getattr(torch, method_variant.__name__)


_T = TypeVar("_T")


class Validation:
    @overload
    @staticmethod
    def assert_non_structured(arg: Union[Node, StructuredTensorMeta]
                              ) -> EasierTensorMeta: ...

    @overload
    @staticmethod
    def assert_non_structured(arg: Union[Node, StructuredTensorMeta],
                              *args: Union[Node, StructuredTensorMeta]
                              ) -> List[EasierTensorMeta]: ...

    @staticmethod
    def assert_non_structured(arg, *args):
        metas = []
        for x in itertools.chain([arg], args):
            if isinstance(x, (Node, int, float)):
                meta = get_meta_from_ir_literal(x)
            else:
                meta = x

            if not isinstance(meta, EasierTensorMeta):
                raise EasierJitException(f'{x} is structured')

            metas.append(meta)

        if len(args) == 0:
            return metas[0]
        else:
            return metas

    @staticmethod
    def require(truth: bool):
        if not truth:
            raise EasierJitException()

    @staticmethod
    def equals(a: _T, b: _T) -> _T:  # TODO vararg
        if a == b:
            return a
        else:
            raise EasierJitException(f'{a} != {b}')

    @staticmethod
    def must_of(obj, t: Type[_T]) -> _T:
        if isinstance(obj, t):
            return obj
        else:
            raise EasierJitException(f'{obj} is not of type {t.__name__}')


V = Validation


def promote_and_validate_roles(*roles: Role) -> Role:
    # Cannot have two non-REPLICA roles.
    dist_roles = set(roles) - set([Role.REPLICA])
    if len(dist_roles) == 0:
        return Role.REPLICA
    elif len(dist_roles) == 1:
        return dist_roles.pop()
    else:
        raise EasierJitException()


def broadcast_and_validate_shapes(*shps: Tuple[int, ...]) -> Tuple[int, ...]:
    assert all(type(s) is tuple for s in shps), \
        'All input shapes to broadcast must be of int tuples'

    try:
        return tuple(torch.functional.broadcast_shapes(*shps))
    except RuntimeError:
        raise EasierJitException()


def broadcast_args(*args: Union[Node, StructuredTensorMeta],
                   dtype: Optional[ScalarType] = None
                   ) -> EasierTensorMeta:
    """
    Calculate resultant metadata of broadcasting `*args`.

    The resultant tensor is always newly allocated.

    This auxiliary function also provides:
    -   promotion of dtypes and roles of `*args`
    -   detect erroneous "mix batch and non-batch dimensions" broadcast usage
    """
    assert len(args) > 0, "Must specify at least one argument to broadcast"

    # Detect erroneous "mix batch and non-batch dimensions" broadcast usage:
    metas = [V.assert_non_structured(x) for x in args]
    max_ndim = max(len(m.shape) for m in metas)
    for i, meta in enumerate(metas):
        if meta.role != Role.REPLICA:
            V.equals(len(meta.shape), max_ndim)
            # ... otherwise we are broadcasting among batch and non-batch dims.

    shape = broadcast_and_validate_shapes(*(m.shape for m in metas))
    dtype = dtype or promote_scalar_types(*(m.dtype for m in metas))
    role = promote_and_validate_roles(*(m.role for m in metas))
    return EasierTensorMeta(shape, dtype, role)


def extract_shape_from_varargs(
        arg: Union[int, Sequence[int]], varargs: Tuple[int, ...]
) -> Tuple[int, ...]:
    def _validate_dims(dims):
        V.require(all(type(d) is int for d in dims))
        return tuple(dims)

    if len(varargs) == 0:
        if type(arg) in [list, tuple]:
            return _validate_dims(arg)
        else:
            d = V.must_of(arg, int)
            return (d,)
    else:
        allargs = (arg,) + varargs
        return _validate_dims(allargs)


def split_list_by_indexes(
    lst: Sequence[Any], indexes: Sequence[int]
) -> Tuple[List[Any], List[Any], List[int]]:
    """
    Args:
    -   lst: commonly are List[int], which is for dim lengths.
    -   indexes: commonly for operator 'dim:List[int]' argument,
                 can contain negatives, but cannot point to the same element.

    Returns:
    -   res[0]: specified elements by `indexes`, in the original order
    -   res[1]: not specified elements, in the original order
    -   res[2]: positive-only version of `indexes`, sorted ascendingly.
    """
    length = len(lst)
    offsets = set()
    for idx in indexes:
        V.require(-length <= idx < length)
        offset = idx % length
        V.require(offset not in offsets)
        offsets.add(offset)

    by_idxes = []
    not_by_idxes = []
    for i in range(length):
        if i in offsets:
            by_idxes.append(lst[i])
        else:
            not_by_idxes.append(lst[i])

    return by_idxes, not_by_idxes, sorted(offsets)
